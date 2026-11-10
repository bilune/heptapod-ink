# Recommendation: learn the hard code first, then its visual reader

Use **T5-small on both sides of a learned categorical bottleneck**, followed by supervised training of polar eyes on the frozen code. This makes language coding and ink recognition separately measurable. Neither Claude option establishes arbitrary-English reconstruction within this budget. All quality forecasts below are guesses, not transferred COGS results.

The local evidence is encouraging but narrower: COGS reaches 78.3% exact, 93.2% with verification; structural generalization remains near zero. Thirty-six slots have a nominal ceiling of approximately `36 log2(33×84)=412` bits, but overlapping marks make the usable channel much worse. The probe’s approximately 105-bit estimate is not measured mutual information or a sentence-capacity guarantee.

**A buys a pretrained output vocabulary and language prior while retaining the proven writer architecture.** MiniLM adaptation may improve lexical fidelity, but pretrained T5 cross-attention has never seen polar features. Project and normalize visual memory to 512 dimensions; preserve T5’s decoder, tied embedding/output weights and tokenizer. Fine-tune pretrained weights slowly and new modules faster. Otherwise catastrophic forgetting and fluent image-independent guesses are plausible. A is the safer incremental experiment.

**B removes about 4M new writer parameters**, letting pretrained attention produce slot representations. Append learned embeddings through `inputs_embeds`, with distinct slot identities, positions and valid attention masks; do not tokenize repeated literal `[SLOT]` strings into unknown tokens. Sentence-dependent slot positions are another distribution shift. B can be cheaper than A, but longer MiniLM self-attention and backpropagation prevent assuming a speedup. Its pretrained writer changes more directly, increasing forgetting risk. Feeding polar features through T5’s encoder adds approximately 19M parameters and computation; it does not make those features linguistic automatically. Start with direct decoder memory instead.

Both require removing `read_text`’s `no_grad`, enabling MiniLM gradients, adding it to the optimizer and saving its weights. Current targets are BERT WordPieces: **never pass their IDs to T5**. Retokenize target strings with T5, use its decoder-start/padding/EOS conventions, and mask label padding with −100. MiniLM is uncased, so A/B cannot reliably preserve original casing. Existing normalized exact match hides this loss. Twelve words also need not fit forty T5 tokens.

Both retain the same discrete-channel problem: soft ink mixtures can transmit information unavailable to argmax drawings. ST supplies biased surrogate gradients, not differentiation through argmax. The renderer’s multiplicative ink union also suppresses gradients where marks overlap. Neither a pretrained decoder nor LoRA fixes this.

## Verified building blocks and hardware

- [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): 22.7M, Apache-2.0.
- [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small): approximately 60M, Apache-2.0. Decoder plus shared embedding/output weights is approximately 42M, calculated from its [configuration](https://huggingface.co/google-t5/t5-small/blob/main/config.json), not another checkpoint.
- [`microsoft/trocr-small-printed`](https://huggingface.co/microsoft/trocr-small-printed): pretrained DeiT/UniLM, then SROIE OCR tuning; [Microsoft reports 62M](https://github.com/microsoft/unilm/blob/master/trocr/README.md). The upstream project specifies [MIT](https://github.com/microsoft/unilm/blob/master/LICENSE); the Hub card lacks an explicit license tag, so distinguish upstream licensing from checkpoint metadata.

These standard PyTorch models fit H100 memory comfortably. **Inference and small-batch training on a 24GB M4 Pro are plausible engineering estimates**, using eager attention/FP32 initially and microbatches of 4–8. [Transformers documents M-series MPS support](https://huggingface.co/docs/transformers/main/apple_silicon), not certification of these complete pipelines. This session reports MPS unavailable: no M4 execution verified. Smoke-test forward, generation and backward, including `grid_sample`; use CPU fallback for unsupported operations. Do not assume the context’s 8× slowdown applies to every model. Use BF16 on H100, FP32 renderer/quantizer arithmetic; no CUDA-only quantization dependency.

## Concrete alternatives

**OCR reader:** freeze a learned writer, unwrap its actual ring image, invert ink to black-on-white, repeat grayscale into RGB, resize the strip to TrOCR’s configured 384×384, and train image-to-sentence pairs with its own tokenizer. This deliberately stretches the strip while preserving angular order; padding a 48-pixel-high strip wastes patches. Fine-tune the vision encoder as well as cross-attention. Edge/stroke features might transfer; alien-mark identity and sentence-global codes are new. That benefit is a guess, not OCR zero-shot capability.

**T5 throughout:** contextual T5 token states → learned slot-query attention → two categorical heads; code embeddings or visual features → new memory adapter → the pretrained T5 decoder. There is no direct text-state bypass around the code.

**Staged learning:** train that decoder on categorical code embeddings first. Freeze writer and symbol embeddings; train polar eyes to predict slot distributions, then pass their expected embeddings through the same memory adapter. Slot labels are generated automatically. This avoids requiring a language loss to discover both the code and visual recognition simultaneously.

**VQ/FSQ:** replace categorical heads with nearest-codebook quantization, or bounded scalar rounding. For example, FSQ levels `[5,5,5,5]` give 625 codes mapped injectively to 625 ring/mark pairs selected for visual separation. FSQ uses rounding STE; VQ uses nearest-neighbor STE plus commitment/codebook losses. The integer ink lookup itself supplies no gradient. Train vision afterward. [FSQ](https://arxiv.org/abs/2309.15505) avoids learning a VQ codebook, but neither method solves visually confusable ink; VQ additionally risks dead codes.

**Two rings:** split at a word boundary into chunks of at most six words, encode independently with the same 24-slot model, concatenate decoded chunks in displayed order. This specifies framing, not grammar or word meanings. It reduces crowding and compression pressure, but whole-sentence exact match compounds chunk errors.

**Verification:** beam candidates are rewritten by the frozen writer and ranked by distance to the received raster, with decoder score breaking ties. For noisy inputs, minimize over small alignment offsets; calibrate a rejection threshold on dev. It cannot recover a sentence missing from the beam or distinguish true channel collisions.

| Option | Mechanism; pretrained/new | Total parameters, approximate | H100 seconds/step, guess | Tatoeba exact / word %, guess | Main risk |
|---|---|---:|---:|---:|---|
| A | MiniLM + T5 decoder; new v3 writer/polar adapter | 72M | .10–.35 | 10–35 / 55–80 | Soft-channel leakage, forgetting |
| B | MiniLM slots + T5 decoder; new heads/polar adapter | 68M; 87M with T5 encoder | .10–.40 | 10–40 / 55–82 | Slot distribution shift |
| T5 joint | T5 encoder/decoder; new slots and polar memory | 68M | .12–.40 | 15–45 / 60–85 | Simultaneous code/vision learning |
| **T5 staged** | Same pretrained T5; new symbolic adapter and slot recognizer | 68M | .10–.35 symbolic; .12–.34 visual | **20–50 / 65–88** | Symbolically good, visually bad code |
| TrOCR staged | Frozen T5 writer + pretrained OCR; new slot writer | 100M | .25–.90 | 10–45 / 55–85 | Visual domain shift, fewer updates |
| FSQ/VQ staged | T5; new quantizer, ink assignment and polar reader | 68M | .10–.35; .12–.34 | 15–50 / 60–88 | Geometry/code mismatch |
| Two-ring staged | Shared T5/eyes, two shorter chunks | 68M | .20–.60 per sentence batch | 25–60 / 70–92 | Two opportunities for errors |
| Staged + verify | Same weights; beam-32 rewrite/rank | 68M | No training change | 25–60 / 68–92 | Beam coverage, collisions |

These overlapping ranges are subjective planning estimates for equal spending, not confidence intervals; collapse to near-zero exact remains possible. Word accuracy means v3’s positional metric. Timings assume batch 64 sentences, targets up to 64 tokens, H100 BF16, excluding evaluation; validate them. Existing COGS logs show approximately .05–.06 seconds/step for the smaller v3 system.

## One budgeted experiment

**Components:** `google-t5/t5-small`; two 256-wide slot-attention blocks producing 36 ring/mark pairs; 256-wide code embeddings plus angular positions; two 256-wide memory blocks projected to 512; existing PolarEyes CNN plus a 36-query slot classifier. Render at 288 pixels, mark scale .25; unwrap to **48×576**, yielding 72 angular CNN tokens. Increasing canvas resolution alone previously left sampling unchanged. Reader input always comes from the raster, never the renderer’s latent polar buffer.

Replace the 40%-soft schedule with a short warm-up, substantial hard-code learning, then writer freezing. At this size, selective freezing is simpler than LoRA; adapters save optimizer memory but do not eliminate backbone computation.

**Data:** fixed 400,000-sentence train-only sample, stratified by length with rare-word oversampling. Each batch: 80% natural sentences, 20% fresh 3–10-word combinations sampled using `random_vocab.json`; exclude held-out strings. Preserve case in training. Audit T5 round trips, unknown characters and lengths before training; dynamically pad longer examples rather than silently truncate. Report unsupported strings as failures. Do not claim all possible English is covered.

**Budget:** [Modal lists H100 at $0.001097/second](https://modal.com/pricing). Cap total GPU allocation at **11,520 seconds (3.2h), $12.64**. Two physical CPU cores plus 8GiB host memory add approximately $0.51 at listed rates: approximately $13.14 total before rounding components, leaving over $2 from $16. Include startup, checkpointing and evaluation; stop at $14 if actual ancillary charges differ. Prepare downloads/tokenization locally.

1. **600 seconds:** benchmark 100 warmed steps; overfit 256 examples; train the slot recognizer on random hard renders. Verify nonzero writer gradients and raster sensitivity. Failures here stop the run.
2. **4,200 seconds, at most 12,000 updates:** symbolic text reconstruction. First 500 updates soft, next 1,500 hard ST-Gumbel, remainder deterministic argmax-forward/softmax-backward. Anneal temperature 1→.5. For probabilities `p`, use `z=onehot(argmax(p))+p−stopgrad(p)`. Gradients pass decoder → memory → `z` → writer; no raster in this phase. Corrupt 5% of symbols using pilot visual confusions, with identity STE. Use AdamW, new modules 3e−4, T5 2e−5, clipping 1; freeze pretrained weights for the first 500 updates. Decoder-prefix dropout .15 discourages language-only guessing.
3. **5,100 seconds, at most 15,000 updates:** freeze the writer, including shared embeddings; cache hard codes. Train eyes with ring/mark CE, then add text CE through soft predicted symbol embeddings and the frozen symbolic reader. Gradients reach eyes through that frozen reader; no writer gradient is needed. After 1,000 updates, permit decoder adaptation at 1e−5 while keeping shared embeddings frozen. Keep 20% clean-symbol replay to limit forgetting.
4. **1,200 seconds evaluation; 420 seconds reserve.** Time caps override update counts. Move unfinished evaluation to the laptop/CPU instead of spending the margin.

At phase 2’s midpoint, require hard-code dev exact ≥20% and random word accuracy ≥40%; otherwise spend remaining phase time on coding, not vision. If still below thresholds, report failure. If symbolic exact exceeds visual exact by >20 points and slot accuracy stalls, use remaining visual time on two rings with the same frozen 36 codes divided 18/18. Concatenate the recognized slots before the unchanged symbolic reader; this fallback changes geometry, unlike the independently trained chunking alternative above. Report the changed format explicitly. Do not restart multiple architectures.

Evaluate full 3,000 dev/test and 1,000 random dev/test sets; test once after dev selection. Report normalized exact, case-sensitive exact, v3 word accuracy, edit-aligned WER, length/rarity strata, symbolic decoding and shuffled-image controls. For beam 32 report oracle candidate recall, selected exact, collision counts and latency; rank against received pixels, never gold codes. Check exported PNG round trips. Guess: random exact 0–20%, substantially below natural text.

**Pre-mortem:** (1) Language priors conceal missing information: random combinations, prefix dropout and shuffled-image controls expose it. (2) Symbolic codes exploit indistinguishable ink: confusion corruption, slot supervision and the two-ring fallback hedge it. (3) Fine-tuning erases useful language knowledge or overruns compute: delayed low-rate adaptation, replay, staged diagnosis and hard wall-clock limits contain the damage.
