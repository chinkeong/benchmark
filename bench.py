"""Local dataset benchmark for LM Studio models: per-request mean acceptance
length (speculative decoding) or throughput over GSM8K, MATH-500, HumanEval,
MBPP and MT-Bench, rendered as a dark table PNG.

Mean acceptance length needs speculative decoding: attach a draft model to the
target model in LM Studio (My Models -> gear icon -> Speculative Decoding), or
pass --draft <model-key> (forwarded to the API as "draft_model"). With lossless
rejection sampling every verification pass emits accepted+1 tokens, so:

    mean acceptance length = completion_tokens / (completion_tokens - accepted_draft_tokens)

Without a draft model the run still works and reports tokens/sec instead.

Examples:
    python bench.py --list
    python bench.py --model qwen/qwen3.8-27b --samples 10
    python bench.py --model qwen/qwen3.8-27b --draft qwen/qwen3.8-0.6b
    python bench.py --model all --samples 5 --datasets GSM8K,MT-Bench
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import requests

from datasets_io import DATASET_NAMES, load_prompts, load_qa, grade
import render_table

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
BASE_URL = "http://localhost:1234"

# progress must be visible live even when stdout is piped/redirected
# (a multi-hour run with fully buffered output looks hung)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Default sampling settings
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)

# subprocess text-mode kwargs: on Windows, bare text=True decodes child output
# as the ANSI codepage (cp1252) and a UTF-8 byte from lms's progress output
# kills the reader thread mid-communicate(), wedging the run forever
_TEXT = dict(text=True, encoding="utf-8", errors="replace")


def machine_info():
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
        r = subprocess.run(["lms", "version"], capture_output=True, **_TEXT,
                           timeout=15)
        if r.returncode == 0:
            info["lms"] = r.stdout.strip().splitlines()[0]
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


def list_models():
    """Downloaded LLM model keys, via `lms ls --json` with API fallback."""
    try:
        out = subprocess.run(["lms", "ls", "--json"], capture_output=True,
                             **_TEXT, timeout=30)
        if out.returncode == 0:
            models = json.loads(out.stdout)
            return [m["modelKey"] for m in models if m.get("type") not in ("embedding",)
                    and "embed" not in m.get("modelKey", "")]
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        pass
    r = requests.get(f"{BASE_URL}/api/v0/models", timeout=10)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"] if m.get("type") != "embeddings"]


def load_model(key):
    print(f"loading {key} ...")
    subprocess.run(["lms", "unload", "--all"], capture_output=True, **_TEXT)
    r = subprocess.run(["lms", "load", key, "--gpu", "max", "-y"],
                       capture_output=True, **_TEXT, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"lms load failed for {key}:\n{r.stderr or r.stdout}")


def run_one(model, prompt, max_tokens, draft_model=None, seed=None,
            sampling=None, timeout=1800):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        **(sampling or SAMPLING),
    }
    if seed is not None:
        payload["seed"] = seed
    if draft_model:
        payload["draft_model"] = draft_model
    r = requests.post(f"{BASE_URL}/api/v0/chat/completions", json=payload,
                      timeout=timeout)
    r.raise_for_status()
    body = r.json()
    stats = body.get("stats", {}) or {}
    usage = body.get("usage", {}) or {}
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        content = ""
    completion = usage.get("completion_tokens") or stats.get("predicted_tokens_count") or 0
    accepted = stats.get("accepted_draft_tokens_count")
    total_draft = stats.get("total_draft_tokens_count")
    rec = {
        "tokens": completion,
        "tok_s": stats.get("tokens_per_second", 0.0),
        "ttft": stats.get("time_to_first_token", 0.0),
    }
    if accepted is not None and completion and completion > accepted:
        rec["accept_len"] = completion / (completion - accepted)
        if total_draft:
            rec["accept_rate"] = accepted / total_draft
    rec["text"] = content
    return rec


def bench_model(model, prompts_by_ds, max_tokens, draft_model, seed, sampling,
                answers_by_ds=None):
    results = {}
    speculative = False
    for ds, prompts in prompts_by_ds.items():
        answers = (answers_by_ds or {}).get(ds) or [None] * len(prompts)
        recs = []
        print(f"[{model}] {ds}: {len(prompts)} prompts")
        for i, p in enumerate(prompts):
            try:
                rec = run_one(model, p, max_tokens, draft_model, seed, sampling)
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
    ap.add_argument("--list", action="store_true", help="list available models and exit")
    ap.add_argument("--model", help="model key, comma-separated keys, or 'all'")
    ap.add_argument("--draft", help="draft model key for speculative decoding")
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
    ap.add_argument("--no-load", action="store_true",
                    help="don't lms load/unload; rely on JIT loading or an already-loaded model")
    args = ap.parse_args()

    if args.list:
        for m in list_models():
            print(m)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in datasets if d not in DATASET_NAMES]
    if bad:
        ap.error(f"unknown dataset(s): {bad}; choose from {DATASET_NAMES}")

    if args.freeze_suite:
        freeze_suite(args.freeze_suite, datasets, args.samples, args.seed,
                     args.max_tokens)
        return

    if not args.model:
        ap.error("--model is required (or use --list / --freeze-suite)")

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

    models = list_models() if args.model == "all" else \
        [m.strip() for m in args.model.split(",") if m.strip()]
    machine = machine_info()
    print(f"machine: {machine.get('host')} | {machine.get('gpu', machine.get('cpu'))}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_files = []
    for model in models:
        if not args.no_load:
            try:
                load_model(model)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"skipping {model}: {e}")
                continue
        t0 = time.time()
        results, speculative = bench_model(model, prompts_by_ds,
                                           args.max_tokens, args.draft,
                                           args.seed, sampling, answers_by_ds)
        if not results:
            print(f"no results for {model}, skipping")
            continue
        run = {
            "model_label": model.split("/")[-1],
            "model_key": model,
            "draft_model": args.draft,
            "speculative": speculative,
            "scored": bool(args.score),
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
    if not args.no_load:
        subprocess.run(["lms", "unload", "--all"], capture_output=True, **_TEXT)

    runs = render_table.load_runs(run_files)
    for rf, run in zip(run_files, runs):
        render_table.render_runs([run], out_path=rf.replace(".json", ".png"))
    if len(runs) > 1:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        render_table.render_runs(runs, out_path=os.path.join(RESULTS_DIR,
                                                             f"comparison_{stamp}.png"))


if __name__ == "__main__":
    main()
