"""Download, cache, and build prompts for the five benchmark datasets:
GSM8K, MATH-500, HumanEval, MBPP, MT-Bench.
Each is fetched once from its canonical public source and cached under ./datasets/.
"""

import gzip
import io
import json
import os
import re

import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

SOURCES = {
    "GSM8K": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "file": "gsm8k_test.jsonl",
    },
    "MATH-500": {
        "url": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "file": "math500_test.jsonl",
    },
    "HumanEval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "file": "humaneval.jsonl",
        "gzip": True,
    },
    "MBPP": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
        "file": "mbpp.jsonl",
    },
    "MT-Bench": {
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        "file": "mtbench_questions.jsonl",
    },
}

DATASET_NAMES = list(SOURCES.keys())


def _download(name):
    src = SOURCES[name]
    path = os.path.join(CACHE_DIR, src["file"])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  downloading {name} from {src['url']} ...")
    r = requests.get(src["url"], timeout=120)
    r.raise_for_status()
    data = r.content
    if src.get("gzip"):
        data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
    # write-then-rename so an interrupted download can't leave a partial
    # file that passes the size>0 cache check
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(name, row):
    """Return the user-turn prompt string for one dataset row."""
    if name == "GSM8K":
        return row["question"] + "\n\nPlease reason step by step, and put your final answer after \"####\"."
    if name == "MATH-500":
        return row["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    if name == "HumanEval":
        return ("Complete the following Python function. Return the full function "
                "implementation in a single code block.\n\n```python\n" + row["prompt"] + "\n```")
    if name == "MBPP":
        tests = "\n".join(row.get("test_list", [])[:1])
        return (row["text"] + "\n\nYour code should satisfy this test:\n" + tests +
                "\n\nReturn the solution in a single Python code block.")
    if name == "MT-Bench":
        return row["turns"][0]
    raise ValueError(f"unknown dataset {name}")


def reference_answer(name, row):
    """Gradeable ground-truth answer for one row, or None if the dataset
    has no mechanically-checkable answer (code datasets, MT-Bench)."""
    if name == "GSM8K":
        return row["answer"].split("####")[-1].strip()
    if name == "MATH-500":
        return str(row.get("answer", "")).strip() or None
    return None


def load_prompts(name, n_samples):
    """Return up to n_samples prompt strings, deterministically spread over the dataset.

    Selection is a pure function of (dataset file, n_samples): evenly spaced
    indices. Same file + same n_samples -> byte-identical prompts on any machine.
    For guaranteed cross-machine fairness, freeze them into a suite file
    (bench.py --freeze-suite) and share that file instead.
    """
    return [p for p, _ in load_qa(name, n_samples)]


def load_qa(name, n_samples):
    """Like load_prompts but returns (prompt, reference_answer) pairs;
    the answer is None where the dataset isn't mechanically gradeable."""
    path = _download(name)
    rows = _read_jsonl(path)
    if n_samples >= len(rows):
        picked = rows
    else:
        step = len(rows) / n_samples
        picked = [rows[int(i * step)] for i in range(n_samples)]
    return [(build_prompt(name, r), reference_answer(name, r)) for r in picked]


def _last_boxed(text):
    """Content of the last \\boxed{...} in text, brace-balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    out = []
    while j < len(text) and depth:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                break
        out.append(ch)
        j += 1
    return "".join(out) if not depth else None


def _norm_answer(s):
    s = s.strip().replace("$", "").replace(",", "").replace(" ", "")
    for junk in ("\\left", "\\right", "\\!", "\\,", "\\;"):
        s = s.replace(junk, "")
    return s.rstrip(".")


def grade(name, response, ref):
    """True/False: does the model's response match the reference answer?"""
    if ref is None:
        return None
    if name == "GSM8K":
        if "####" in response:
            pred = response.rsplit("####", 1)[-1].strip().splitlines()[0]
        else:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", response)
            pred = nums[-1] if nums else ""
    elif name == "MATH-500":
        pred = _last_boxed(response) or ""
    else:
        return None
    a, b = _norm_answer(pred), _norm_answer(ref)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    for ds in DATASET_NAMES:
        prompts = load_prompts(ds, 3)
        print(f"{ds}: ok, sample prompt: {prompts[0][:80]!r}")
