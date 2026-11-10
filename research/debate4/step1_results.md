# Step 1 results: audit and references (2026-11-09)

Checkpoint: v4 final (`v4_vq_ent2/model.pt`). Search: exhaustive, 16 nearest symbols per slot incl. the writer's own (a slot can return), up to 20 rounds, keyed batch-invariant views (verified: batch 4 vs 2+2 give identical codes), visual guard. All readings through the 16-bit PNG path and the eyes. 256 random examples per panel, seed 0. Modal H100, ~25 min.

| panel | plain exact / words | search exact / words | search, unseen perturbation exact | shuffled exact | moves | s/sentence |
| --- | --- | --- | --- | --- | --- | --- |
| Tatoeba dev | 68.8% / 91.3% | **96.5% / 98.9%** | 94.5% | 1.2% | 8.6 | 0.44 |
| dev_random (3-10 weighted words) | 9.0% / 44.6% | **52.3% / 79.6%** | 49.2% | 0.8% | 16.5 | 0.87 |
| dev_uniform10 (10 uniform words) | 0.0% / 11.4% | **0.0% / 27.0%** | 0.0% | 0.0% | 17.5 | 1.18 |
| EWT fresh_natural | 20.7% / 61.7% | **66.8% / 84.7%** | 61.7% | 1.2% | 16.0 | 1.12 |
| fresh_random | 11.7% / 44.7% | **54.3% / 78.0%** | 48.8% | 2.0% | 16.3 | 0.90 |

## References (evaluation only)
Source bits (ideal arithmetic-coding length incl. EOS + 2 bits):
- Tatoeba dev: GPT-2 median 59.5 / p99 110 / max 156; Qwen2.5-1.5B median 59.9 / p99 105 / max 145.
- EWT: GPT-2 median 72.4 / p99 192; Qwen2.5-1.5B median 74.3 / p99 181 / max 217 (99.6% fit in 202 bits).
- Random 3-10 weighted (exact sampler distribution): median ~94, max ~157. Weighted 10 words: median 142. Uniform 10 words: 149.8.
Channel (eyes, 16-bit PNG): uniform random codewords 3.2% symbol errors clean / 5.4% perturbed; writer codes 1.4-2.0% / 2.7-3.8%.
RS over GF(81) block success on uniform codewords: RS(36,32) [202 bits] 87.7% clean / 68.1% perturbed; RS(36,28) [177 bits] 99.2% / 94.8%.
Classical estimate (fits x decodes), RS(36,28): Tatoeba 99.2% clean / 94.8% perturbed; EWT ~98% / 93.6%; all random sets incl. uniform 10 words 99.2% / 94.8%.
Caveat: source bits use ideal code length, not an implemented byte-exact arithmetic coder.

## Reading
- Natural, predictable text: learned codec + search is close to the classical reference on Tatoeba (96.5% vs ~99%), far on EWT (67% vs ~98%).
- High-entropy text is the gap: uniform 10-word combos need ~150 bits, well inside the classical 177-bit RS budget (99% decodable), yet the learned code gets 0% exact (27% words with search). The learned code's effective rate is far below the channel.
- The eyes read the writer's codes better than arbitrary codewords (1.4% vs 3.2% symbol error): the writer prefers readable marks.
