# Local LM Studio dataset benchmark

Benchmarks any model in your local LM Studio library over five standard
datasets — GSM8K, MATH-500, HumanEval, MBPP, MT-Bench — measuring per-request
mean acceptance length (with speculative decoding) or generation throughput,
and renders the results as a dark table PNG.

## Requirements

- LM Studio with the local server running on `http://localhost:1234`
  (Developer tab -> Start Server), plus the `lms` CLI on PATH.
- Python 3.10+ with `requests` and `Pillow`.

## What it measures

**Mean acceptance length** exists only with speculative decoding. Attach a
draft model to the target model in LM Studio (My Models -> gear ->
Speculative Decoding -> pick a small draft model), or try `--draft <model-key>`.
LM Studio's native REST API (`/api/v0/chat/completions`) then reports
`accepted_draft_tokens_count`; since lossless rejection sampling emits
`accepted + 1` tokens per verification pass:

```
mean acceptance length = completion_tokens / (completion_tokens - accepted_draft_tokens)
```

This is a *speed* metric, not a quality score — lossless rejection sampling
means the output distribution is identical to the base model's; a higher
number only means more tokens per verification pass.

Without a draft model the run still works and the table shows **tokens/sec,
TTFT, and mean output tokens** instead.

Default sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5.

## Usage

```powershell
python bench.py --list                              # show available model keys
python bench.py --model qwen/qwen3.8-27b            # full 5-dataset run, 10 samples each
python bench.py --model qwen/qwen3.8-27b --samples 25 --max-tokens 2048
python bench.py --model a,b,c                       # several models -> also a comparison PNG
python bench.py --model all                         # every downloaded LLM (slow!)
python bench.py --model X --datasets GSM8K,MT-Bench --samples 5   # quick check
python bench.py --model X --no-load                 # use whatever is already loaded (JIT)
```

Each run writes `results/<model>_<timestamp>.json` and a matching `.png`.
Re-render or combine old runs any time:

```powershell
python render_table.py --latest
python render_table.py results\run1.json results\run2.json   # comparison table
```

## Fair comparisons across machines / quants

Prompt selection is already deterministic (evenly spaced indices, no RNG), but
for guaranteed apples-to-apples runs freeze the exact prompts + settings into a
portable **suite file** and use that everywhere:

```powershell
python bench.py --freeze-suite suite_v1.json --samples 10        # once, on any machine
# copy suite_v1.json to each machine, then on every machine:
python bench.py --model <model-key> --suite suite_v1.json
```

The suite pins the prompts (SHA-256 verified — a corrupted/edited file refuses
to run), sampler settings, seed, and max_tokens; `--samples/--max-tokens` are
taken from the suite so nobody can accidentally diverge. Every result JSON and
PNG caption records the suite hash plus the machine fingerprint (host, GPU,
driver, OS), so two tables are comparable iff their suite hashes match.

A fixed `--seed` (default 42) is sent with every request. Note what this does
and doesn't buy you: given identical logits it makes sampling reproducible, but
different backends (CUDA/Metal/SYCL) produce slightly different logits from the
same weights, so generations can still diverge across hardware — that wobble is
inherent, not a flaw in the test. Interpretation guide:

- **Different quants of the same model** (Q4_K_M vs NVFP4 vs ...): systematic
  differences — separate columns in a comparison table.
- **Same file on different hardware**: noise, not quality — report as one cell
  with a ± range; speed metrics (tok/s, acceptance length) are the real signal.
- For **quality-style comparisons** where you want decoding itself
  deterministic per machine, add `--greedy` (temperature 0, top-k 1).

## Files

- `bench.py` — runner: loads each model via `lms`, sends prompts, collects stats
- `datasets_io.py` — downloads/caches the 5 datasets into `datasets/` and builds prompts
- `render_table.py` — renders result JSON into a dark table PNG
- `results/` — run JSONs and PNGs

## Notes

- Runtime estimate: samples x datasets x (max_tokens / tok_s). 10 samples x 5
  datasets x ~25 s/prompt is roughly 20 minutes per model.
- Prompts are deterministically spread across each dataset, so runs with the
  same `--samples` are comparable across models.
