"""Local dataset benchmark for GGUF models on llama.cpp: accuracy (--score),
per-request mean acceptance length (speculative decoding), or throughput over
GSM8K, MATH-500, HumanEval, MBPP and MT-Bench, rendered as a dark table PNG.

The runner launches llama-server itself for each model, waits for /health,
sends the prompts, and reads llama.cpp's `timings` from every response —
no external server or management app required.

Speculative decoding is configured server-side; pass the flags through:
    --server-args "--spec-type draft-mtp --spec-draft-n-max 10 --spec-draft-p-min 0.5"
With lossless rejection sampling every verification pass emits accepted+1
tokens, so:

    mean acceptance length = predicted_n / (predicted_n - draft_n_accepted)

Examples:
    python bench.py --model path\\to\\model.gguf --samples 10
    python bench.py --model a.gguf,b.gguf --datasets GSM8K,MATH-500 --greedy --score
    python bench.py --model a.gguf --server-args "--spec-type draft-mtp --spec-draft-n-max 10"
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import requests

from datasets_io import DATASET_NAMES, load_prompts, load_qa, grade
import render_table

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# progress must be visible live even when stdout is piped/redirected
# (a multi-hour run with fully buffered output looks hung)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Default sampling settings
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)

# subprocess text-mode kwargs: on Windows, bare text=True decodes child output
# as the ANSI codepage (cp1252) and a UTF-8 byte in a child's output kills the
# reader thread mid-communicate(), wedging the run forever
_TEXT = dict(text=True, encoding="utf-8", errors="replace")


def find_server(explicit=None):
    """Locate llama-server: --server-bin, $LLAMA_SERVER, PATH, known installs."""
    candidates = [explicit, os.environ.get("LLAMA_SERVER"),
                  shutil.which("llama-server"),
                  r"E:\AI\llama.cpp\llama-server.exe",
                  r"E:\AI\llama.cpp-dflash\build\bin\llama-server.exe"]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    sys.exit("llama-server not found: pass --server-bin, set LLAMA_SERVER, "
             "or put llama-server on PATH")


def machine_info(server_bin):
    """Fingerprint of this machine/backend, stored in every result file."""
    info = {
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor(),
    }
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"], capture_output=True,
                           **_TEXT, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            info["gpu"] = r.stdout.strip().splitlines()[0]
    except OSError:
        pass
    try:
        r = subprocess.run([server_bin, "--version"], capture_output=True,
                           **_TEXT, timeout=15)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if line.strip().startswith("version"):
                info["llama_cpp"] = line.strip()
                break
    except OSError:
        pass
    return info


def _suite_hash(prompts_by_ds):
    h = hashlib.sha256()
    for ds in sorted(prompts_by_ds):
        for p in prompts_by_ds[ds]:
            h.update(ds.encode())
            h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def freeze_suite(path, datasets, samples, seed, max_tokens):
    """Snapshot the exact prompts + settings into a portable suite file."""
    qa_by_ds = {ds: load_qa(ds, samples) for ds in datasets}
    prompts_by_ds = {ds: [p for p, _ in qa] for ds, qa in qa_by_ds.items()}
    suite = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "settings": {**SAMPLING, "samples": samples, "max_tokens": max_tokens,
                     "seed": seed},
        "prompts": prompts_by_ds,
        "answers": {ds: [a for _, a in qa] for ds, qa in qa_by_ds.items()},
        "hash": _suite_hash(prompts_by_ds),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2, ensure_ascii=False)
    print(f"froze suite -> {path} (hash {suite['hash']}, "
          f"{sum(len(v) for v in prompts_by_ds.values())} prompts). "
          f"Copy this file to other machines and run with --suite.")


def load_suite(path):
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)
    actual = _suite_hash(suite["prompts"])
    if actual != suite.get("hash"):
        sys.exit(f"suite file {path} is corrupted: hash {actual} != {suite.get('hash')}")
    return suite


class Server:
    """One llama-server process hosting one model."""

    def __init__(self, server_bin, model_path, port, ctx, server_args):
        self.base_url = f"http://127.0.0.1:{port}"
        cmd = [server_bin, "-m", model_path, "-c", str(ctx), "-ngl", "99",
               "--parallel", "1", "--jinja",
               "--host", "127.0.0.1", "--port", str(port)] + server_args
        print(f"starting llama-server for {os.path.basename(model_path)} ...")
        self.log = open(os.path.join(RESULTS_DIR, "llama-server.log"), "w",
                        encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=self.log)

    def wait_ready(self, timeout_s=600):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.proc.returncode} "
                    f"(see results/llama-server.log)")
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.ok and r.json().get("status") == "ok":
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become healthy in time")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run_one(base_url, prompt, max_tokens, seed=None, sampling=None,
            timeout=1800):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        **(sampling or SAMPLING),
    }
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    body = r.json()
    t = body.get("timings", {}) or {}
    try:
        msg = body["choices"][0]["message"]
        # with --jinja llama-server splits thinking into reasoning_content;
        # the final answer (####, \boxed{}) lives in content
        content = msg.get("content") or ""
    except (KeyError, IndexError):
        content = ""
    completion = int(t.get("predicted_n") or 0)
    rec = {
        "tokens": completion,
        "tok_s": t.get("predicted_per_second", 0.0) or 0.0,
        "ttft": (t.get("prompt_ms", 0.0) or 0.0) / 1000.0,
        "text": content,
    }
    accepted = t.get("draft_n_accepted")
    total_draft = t.get("draft_n")
    if accepted is not None and completion and completion > accepted:
        rec["accept_len"] = completion / (completion - accepted)
        if total_draft:
            rec["accept_rate"] = accepted / total_draft
    return rec


def bench_model(label, base_url, prompts_by_ds, max_tokens, seed, sampling,
                answers_by_ds=None):
    results = {}
    speculative = False
    for ds, prompts in prompts_by_ds.items():
        answers = (answers_by_ds or {}).get(ds) or [None] * len(prompts)
        recs = []
        print(f"[{label}] {ds}: {len(prompts)} prompts")
        for i, p in enumerate(prompts):
            try:
                rec = run_one(base_url, p, max_tokens, seed, sampling)
            except (requests.RequestException, RuntimeError) as e:
                print(f"    prompt {i+1}/{len(prompts)} FAILED: {e}")
                continue
            truncated = rec["tokens"] >= max_tokens
            # a truncated response never reached its final answer: count it
            # wrong without the last-number fallback, which can luckily match
            # a mid-thinking number and score a spurious CORRECT
            text = rec.pop("text")
            verdict = False if truncated else grade(ds, text, answers[i])
            if answers[i] is not None:
                rec["correct"] = bool(verdict)
                rec["truncated"] = truncated
            recs.append(rec)
            al = f", accept_len {rec['accept_len']:.2f}" if "accept_len" in rec else ""
            sc = ("" if answers[i] is None else
                  ", CORRECT" if verdict else
                  ", wrong (truncated)" if truncated else ", wrong")
            print(f"    prompt {i+1}/{len(prompts)}: {rec['tokens']} tok, "
                  f"{rec['tok_s']:.1f} tok/s{al}{sc}")
        if not recs:
            print(f"    {ds}: all prompts failed, skipping dataset")
            continue
        agg = {
            "n": len(recs),
            "tokens": _mean(recs, "tokens"),
            "tok_s": _mean(recs, "tok_s"),
            "ttft": _mean(recs, "ttft"),
        }
        graded = [r for r in recs if "correct" in r]
        if graded:
            agg["accuracy"] = sum(r["correct"] for r in graded) / len(graded)
            agg["graded_n"] = len(graded)
            agg["truncated_n"] = sum(r.get("truncated", False) for r in graded)
        with_spec = [r for r in recs if "accept_len" in r]
        if with_spec:
            speculative = True
            agg["accept_len"] = _mean(with_spec, "accept_len")
            agg["accept_rate"] = _mean([r for r in with_spec if "accept_rate" in r],
                                       "accept_rate") if any("accept_rate" in r for r in with_spec) else 0.0
        results[ds] = agg
    return results, speculative


def _mean(recs, key):
    vals = [r[key] for r in recs]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to a GGUF file, or comma-separated paths")
    ap.add_argument("--server-bin", help="path to llama-server (default: "
                    "$LLAMA_SERVER, PATH, or known local installs)")
    ap.add_argument("--server-args", default="",
                    help="extra llama-server flags, e.g. "
                         "\"--spec-type draft-mtp --spec-draft-n-max 10\" "
                         "(whitespace-split; avoid paths with spaces)")
    ap.add_argument("--port", type=int, default=1236,
                    help="port for the spawned llama-server (default 1236)")
    ap.add_argument("--ctx", type=int, default=8192,
                    help="context size -c for the spawned server (default 8192)")
    ap.add_argument("--datasets", default=",".join(DATASET_NAMES),
                    help="comma-separated subset of: " + ", ".join(DATASET_NAMES))
    ap.add_argument("--samples", type=int, default=10, help="prompts per dataset (default 10)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="max completion tokens (default 1024)")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampler seed sent with every request (default 42)")
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0 / top-k 1: deterministic decoding for "
                         "quality comparisons (overrides default sampling)")
    ap.add_argument("--score", action="store_true",
                    help="grade answers for accuracy on gradeable datasets "
                         "(GSM8K final number, MATH-500 boxed answer); "
                         "combine with --greedy for reproducible scores")
    ap.add_argument("--freeze-suite", metavar="FILE",
                    help="write the exact prompts+settings to FILE and exit; "
                         "copy it to other machines for identical runs")
    ap.add_argument("--suite", metavar="FILE",
                    help="run the prompts/settings frozen in FILE instead of "
                         "sampling datasets locally")
    ap.add_argument("--no-spawn", action="store_true",
                    help="don't launch llama-server; use whatever is already "
                         "listening on --port")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in datasets if d not in DATASET_NAMES]
    if bad:
        ap.error(f"unknown dataset(s): {bad}; choose from {DATASET_NAMES}")

    if args.freeze_suite:
        freeze_suite(args.freeze_suite, datasets, args.samples, args.seed,
                     args.max_tokens)
        return

    if not args.model:
        ap.error("--model is required (or use --freeze-suite)")

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    missing = [m for m in models if not os.path.exists(m)]
    if missing and not args.no_spawn:
        ap.error(f"model file(s) not found: {missing}")

    server_bin = find_server(args.server_bin)
    server_args = args.server_args.split()

    answers_by_ds = None
    if args.suite:
        suite = load_suite(args.suite)
        prompts_by_ds = suite["prompts"]
        s = suite["settings"]
        args.samples, args.max_tokens = s["samples"], s["max_tokens"]
        args.seed = s.get("seed", args.seed)
        sampling = {k: s[k] for k in SAMPLING}
        suite_hash = suite["hash"]
        if args.score:
            answers_by_ds = suite.get("answers")
            if not answers_by_ds:
                sys.exit("--score with a suite needs answers in the suite file; "
                         "re-freeze it with the current bench.py")
        print(f"using suite {args.suite} (hash {suite_hash})")
    else:
        qa_by_ds = {ds: load_qa(ds, args.samples) for ds in datasets}
        prompts_by_ds = {ds: [p for p, _ in qa] for ds, qa in qa_by_ds.items()}
        if args.score:
            answers_by_ds = {ds: [a for _, a in qa] for ds, qa in qa_by_ds.items()}
        sampling = dict(SAMPLING)
        suite_hash = _suite_hash(prompts_by_ds)

    if args.greedy:
        sampling.update(temperature=0.0, top_k=1, top_p=1.0, presence_penalty=0.0)

    machine = machine_info(server_bin)
    print(f"machine: {machine.get('host')} | {machine.get('gpu', machine.get('cpu'))} "
          f"| {machine.get('llama_cpp', 'llama.cpp version unknown')}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_files = []
    for model in models:
        label = os.path.splitext(os.path.basename(model))[0]
        server = None
        base_url = f"http://127.0.0.1:{args.port}"
        try:
            if not args.no_spawn:
                server = Server(server_bin, model, args.port, args.ctx, server_args)
                server.wait_ready()
        except RuntimeError as e:
            print(f"skipping {label}: {e}")
            if server:
                server.stop()
            continue
        t0 = time.time()
        try:
            results, speculative = bench_model(label, base_url, prompts_by_ds,
                                               args.max_tokens, args.seed,
                                               sampling, answers_by_ds)
        finally:
            if server:
                server.stop()
        if not results:
            print(f"no results for {label}, skipping")
            continue
        run = {
            "model_label": label,
            "model_key": model,
            "speculative": speculative,
            "scored": bool(args.score),
            "backend": {"engine": "llama.cpp (llama-server)",
                        "server_bin": server_bin,
                        "version": machine.get("llama_cpp"),
                        "server_args": args.server_args},
            "machine": machine,
            "suite_hash": suite_hash,
            "datasets": [d for d in prompts_by_ds if d in results],
            "results": results,
            "settings": {**sampling, "samples": args.samples,
                         "max_tokens": args.max_tokens, "seed": args.seed},
            "elapsed_s": round(time.time() - t0, 1),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"{render_table._slug(run['model_label'])}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        print(f"saved {path}")
        run_files.append(path)

    if not run_files:
        sys.exit("nothing benchmarked")

    runs = render_table.load_runs(run_files)
    for rf, run in zip(run_files, runs):
        render_table.render_runs([run], out_path=rf.replace(".json", ".png"))
    if len(runs) > 1:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        render_table.render_runs(runs, out_path=os.path.join(RESULTS_DIR,
                                                             f"comparison_{stamp}.png"))


if __name__ == "__main__":
    main()
