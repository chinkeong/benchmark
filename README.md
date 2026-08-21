# Local dataset benchmark

Benchmarks local models over five standard datasets — GSM8K, MATH-500,
HumanEval, MBPP, MT-Bench — measuring accuracy (`--score`), per-request mean
acceptance length (with speculative decoding), or generation throughput, and
renders the results as a dark table PNG. The runner launches **llama-server**
(llama.cpp) itself for each model, waits for it to become healthy, and reads
llama.cpp's `timings` from every response — no external server or management
app involved.

## Requirements

- A llama.cpp build with `llama-server` — found via `--server-bin`, the
  `LLAMA_SERVER` env var, or PATH.
- Python 3.10+ with `requests` and `Pillow`.

## What it measures

**Mean acceptance length** exists only with speculative decoding, which is
configured server-side and passed through:

```powershell
--server-args "--spec-type draft-mtp --spec-draft-n-max 10 --spec-draft-p-min 0.5"
# external drafter (e.g. DFlash2): --server-args "-md path\to\drafter.gguf --spec-type draft-dflash --spec-draft-n-max 4"
```

llama-server reports `draft_n` / `draft_n_accepted` in its `timings`; since
lossless rejection sampling emits `accepted + 1` tokens per verification pass:

```
mean acceptance length = predicted_n / (predicted_n - draft_n_accepted)
```

This is a *speed* metric, not a quality score — lossless rejection sampling
means the output distribution is identical to the base model's; a higher
number only means more tokens per verification pass.

Without a draft model the run still works and the table shows **tokens/sec,
TTFT, and mean output tokens** instead.

**Accuracy** (`--score`) grades answers on the mechanically-checkable
datasets: GSM8K (final number after `####`, numeric-tolerant) and MATH-500
(last `\boxed{}` answer, LaTeX-normalized). Responses that hit `--max-tokens`
are counted wrong and reported separately as truncations — raise the cap for
thinking models (4096+) or the score measures your budget, not the model.
Combine with `--greedy` for reproducible per-machine scores:

```powershell
python bench.py --model a,b --datasets GSM8K,MATH-500 --samples 20 --max-tokens 4096 --greedy --score
```

Default sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5.

## Usage

`--model` takes GGUF file paths (comma-separated for several -> also a
comparison PNG). The spawned server defaults: `-ngl 99 --parallel 1 --jinja
-c 8192` on port 1236; add anything else via `--server-args`.

```powershell
python bench.py --model C:\models\a.gguf                          # full 5-dataset run
python bench.py --model a.gguf,b.gguf --samples 25 --max-tokens 2048
python bench.py --model a.gguf --datasets GSM8K,MT-Bench --samples 5   # quick check
python bench.py --model a.gguf --server-args "--spec-type draft-mtp --spec-draft-n-max 10"
python bench.py --model a.gguf --no-spawn --port 1234             # benchmark an already-running server
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

- `bench.py` — runner: spawns llama-server per model, sends prompts, collects stats
- `datasets_io.py` — downloads/caches the 5 datasets into `datasets/` and builds prompts
- `render_table.py` — renders result JSON into a dark table PNG
- `results/` — run JSONs and PNGs

## Notes

- Runtime estimate: samples x datasets x (max_tokens / tok_s). 10 samples x 5
  datasets x ~25 s/prompt is roughly 20 minutes per model.
- Prompts are deterministically spread across each dataset, so runs with the
  same `--samples` are comparable across models.
