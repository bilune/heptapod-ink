# 1. Diagnosis

**Recommendation: buy information about the bottleneck first, then spend on deterministic-code training. Do not spend the budget extending the original stochastic hard phase.** Freezing has already produced the largest measured improvement. The next decisive training experiment is a reader trained on the actual frozen discrete choices, bypassing rendering *for diagnosis only*. First screen whether beam candidates already contain the correct answer. Those two measurements determine whether to repair the code, the visual reader, or decoding.

**90% is a stretch target, not an evidence-based forecast.** My planning guess is 75–85% for the selected image-to-text system, with 90% plausible if the discrete-code probe reaches at least 95%, or beam candidate coverage reaches 90% and re-rendering resolves the candidates. Confidence in that forecast is low: we have neither measurement. These conditions are decision thresholds, not established ceilings.

## Evidence and scope

I read [the supplied context](context.md) first, then [v3/heptapod.py](../../v3/heptapod.py) and [v2/heptapod_ink.py](../../v2/heptapod_ink.py). Additional implementation evidence comes from [channel_probe.py](../../v3/channel_probe.py), [modal_app.py](../../v3/modal_app.py), and [cogs_graphs.py](../../v3/cogs_graphs.py). Throughout this document:

- **Measured** means the supplied context, local logs, or the read-only tokenizer audit below.
- **Code finding** means a directly inspected implementation property.
- **Guess** means a proposed mechanism, hyperparameter, runtime, or gain not yet measured here. All intervention gains and decision thresholds are guesses unless explicitly identified otherwise. Gains are percentage points of sentence exact match, not relative percentages, and must not be added together.

The launch logs were still growing during inspection. At the common **20,000-step snapshot**:

| Run | Dev exact, first 512 | Train minibatch CE | Dev right / shuffled CE, 128 examples | Training ink mean |
|---|---:|---:|---:|---:|
| freeze25 | 66.99% | 0.0230 | 0.2542 / 12.8300 | 6.60% |
| freeze40 | 65.43% | 0.0440 | 0.2622 / 12.0661 | 6.89% |

Sources: [freeze25 log](../../v3/runs/modal_freeze25.launch.log), [freeze40 log](../../v3/runs/modal_freeze40.launch.log). These update the context's approximately 65% plateau and approximately 4 shuffled CE. They are not final full-dev/test results. The local `v3/out/model.pt` is a **40-step pilot**, according to its saved configuration; it must not be used as the frozen incumbent. Fetch the actual Modal run checkpoints before executing this plan.

### (a) Frozen code quality and ambiguity

**Supported:** freezing at 12k improves freeze25 from 33.40% to 63.09% after just 2k reader updates; freeze40 goes from 15.23% to 59.38%. Thus a substantial amount of usable information was already present in the deterministic code. This is strong evidence against interpreting low switch-time exact match as a hard information ceiling. It also confounds several changes: code stationarity, deterministic choices, writer dropout disabled, a new reader optimizer, and reduced LR all happen together.

**Still unresolved:** the code could omit distinctions or encode them in a way that generalizes poorly. At 20k the frozen runs have training minibatch CE around 0.02–0.04 despite augmentation and word dropout, versus approximately 0.25 on clean dev with gold prefixes. These are unmatched samples/objectives, but the gap makes simple global undertraining less convincing. High shuffled CE establishes dependence on the drawing; it does **not** establish that every word, role, or surface-form distinction is transmitted. A meaning-preserving code can still fail exact sentence reconstruction.

**A crucial code finding:** lowering `tau` does not make hard Gumbel samples deterministic. For positive temperature, `argmax((logits + gumbel) / tau)` is independent of `tau`. Annealing changes the soft surrogate and its gradients; it does not turn sampled hard choices into the evaluation-time `argmax(logits)`. At the .4 switch fraction, both the 30k and 80k schedules have temperature approximately 1.15. Longer training with the same schedule is not an annealing-away of this mismatch. Measure per-slot sampled-versus-argmax disagreement and logit margins, alongside reader accuracy on each channel.

**Cheap measurements:** cache deterministic codes for train and dev; count distinct ordered 48-component code vectors, per-slot frequencies, conditional entropy, top-two logit margins, occupancy, and train-derived minimal-pair differences. Do not quotient rotations: the deployed channel is canonically oriented. Group dev examples by identical code and separately by identical rendered image. The empirical deterministic collision bound is

\[
C=\frac{1}{N}\sum_z\max_y n(z,y).
\]

This is an upper bound on accuracy on that finite sample, not a population estimate. No collisions does not imply learnability, and approximate image distances are not collision proofs. Inspect train/dev cross-split conflicts separately.

Train a small **symbolic-code reader** on train codes only, then evaluate on dev: learned ring-ID and mark-ID embeddings plus slot position, a transformer encoder, and the same autoregressive text decoder. It receives hard IDs, never logits, MiniLM features, text length, or soft probabilities. If it reaches 95% while pixels remain near 70%, the deployed bottleneck is largely visual. If it fits train but remains near 70% on dev, changing the visual reader alone is unlikely to solve the problem under this budget. A failed probe is not an information-theoretic bound; check optimization and a continuous-feature control before condemning the code. Budget: 6–12 H100 minutes, including that control if needed.

### (b) Reader capacity and training

**Supported:** reader-only training is demonstrably effective initially, but gains slow: freeze25 improves only 3.90 points from 14k to 20k. The decoder has three 256-wide layers, random token embeddings, and a 30,522-way output. `target_loss` keeps 30% input-word dropout even after the writer is frozen. Both CE and BOW independently call `reader.memory`, running the visual tower twice and updating its BatchNorm statistics twice per training batch. These are concrete opportunities for objective alignment and compute savings.

**Contradictions/limits:** a larger reader is not automatically indicated by a low train CE. The frozen run's actual train exact match and length-specific errors are missing. More capacity might memorize an arbitrary code better without helping new sentences. BOW supervision contains presence but not order or multiplicity; it cannot by itself teach role binding. At 20k, logged BOW contributions are about 0.010 and 0.013, versus CE 0.023 and 0.044: BOW is not numerically negligible, although scalar loss sizes do not determine gradient sizes.

**Cheap measurements:** clean greedy exact and unmasked teacher-forced CE on the same fixed 2,000 train examples and full dev; exact by token length and frequency; first wrong token; insertion/deletion, repetition, role swap, article and EOS error counts. Fit 256 training sentences to near-perfect reconstruction as an optimization control. Compare two 6k-update reader continuations from the identical checkpoint: existing objective versus lower dropout/BOW and weaker augmentation. If train exact is already above 98% and dev remains flat, stop treating elapsed updates as the main missing ingredient. Budget: approximately 12–18 minutes for the pair and evaluations.

### (c) Physical channel capacity and legibility

**Supported:** [the supplied random-code probe](context.md) finds only 71–79% recognition of present marks and 15–42% ring recognition after 1,200 steps. `RingMarkRenderer.polar` unions overlapping ink with `1 - product(1 - ink)`, so marks can conceal other marks and ring details. PolarEyes compresses the radial dimension to one cell and the angular dimension to 48 tokens. The channel adds perturbations in training but none in normal evaluation. These mechanisms can make a sufficiently large discrete alphabet visually hard to learn.

**Important correction:** approximately 2–3% ink describes the joint hard runs, not the current frozen runs. The latter retain approximately **6–7% ink** in the logs. A minimum-ink penalty cannot improve a frozen writer and is not an explanation of its plateau. Nor is the existing ceiling penalty actively rewarding blankness: `relu(ink - .12)` has zero gradient below 12%. The precise cause of joint collapse remains unmeasured.

The renderer also supplies a specific gradient concern: for an ink union, the derivative with respect to one overlapping contribution contains the product of the other contributions' remaining transparency. Near-opaque overlap can therefore suppress writer gradients. This follows from the implemented union, but its contribution to the observed collapse is a **hypothesis**. During writer repair, record head-gradient norms, blank logits and hard occupancy; do not diagnose this from ink density alone.

The nominal capacity is `24 * log2(33 * 84) = 274.5` bits. This is a combinatorial upper bound, not usable capacity. The context's “60–100 bits per sentence” is a guess; vocabulary size, occupancy and symbol recognition accuracy do not establish sentence entropy or a channel limit. Recognition errors from a short mismatched probe also do not constitute a Bayes-error bound. The 25% and 40% runs learned different codes; their current exact scores do not isolate mark width.

**Cheap measurements:** on the *actual used codes*, train a slot-ID prediction head; report ring/mark confusion, nonblank recall, per-slot accuracy and **whole-code exact**, clean and perturbed. Compare a symbolic reader, a reader on `script.polar` before Cartesian resampling, and the deployed raster reader where necessary. The first two are diagnostic bypasses, not valid final scores. Compare renderings of the same codes at scales .25/.4, initially by confusion/distance and then by a short reader recalibration; blindly changing scale under an existing reader is not a fair test. Budget: 2–5 minutes for a targeted perception probe, included in the visual-repair allocation.

### (d) Decoding

**Code finding:** `Reader.generate` uses only greedy next-token decisions. Teacher-forced CE does not measure free-running sequence success. A wrong early word can change all later prefixes. As an illustration, nine independent token decisions would need approximately `0.9**(1/9) = 98.84%` correctness for 90% sentence exact; the independence assumption is false in general. The logged word accuracy is positional whitespace agreement, not token accuracy or edit-distance accuracy, so it cannot be substituted into this calculation.

**Cheap measurement:** beam widths 4 and 8 on dev, then 32 only if candidate coverage is promising; score EOS, use no gold length or gold prefix, and test length-normalization exponent 0 and .6. Record top-1 exact, whether the gold sentence occurs anywhere in the candidate list, and likelihood ranking of gold versus the greedy output. Gold-in-beam is an analysis oracle, never the reported decoder. Wider search can choose a higher-probability wrong sentence; its gain is not guaranteed.

**A particularly relevant inference option:** re-encode each generated candidate with the frozen writer, render it, and compare it with the received image. On the clean deterministic channel, a correct candidate must reproduce the image, up to numerical tolerance. If exactly one candidate matches, this can resolve a likelihood-ranking error without observing the original text or original slot IDs. If gold is absent, or two sentences render identically, this cannot repair the ambiguity. This is a proposed analysis-by-synthesis decoder, not a measured result or a new communication channel. Report it separately from a standalone fresh reader. Budget: 2–6 minutes for cached/vectorized dev decoding and reranking, subject to an initial timing check.

### (e) Evaluation artifacts

I ran a **read-only local audit** with the cached MiniLM tokenizer, `truncation=False`, `decode(skip_special_tokens=True)`, and the exact `normalize` function from v3:

| Split | N | Mean WordPiece length including CLS/SEP | Maximum | Length > 40 | Normalize/decode round-trip mismatches |
|---|---:|---:|---:|---:|---:|
| Train | 23,657 | 9.876 | 24 | 0 | 0 |
| Dev | 2,862 | 10.026 | 25 | 0 | 0 |
| Test | 2,858 | 9.987 | 21 | 0 | 0 |
| Gen | 20,974 | 12.435 | 64 | 537 | 0 |

There are zero normalized sentence overlaps between train and each held-out split. Train contains 804 token IDs including CLS/SEP; full tokenizer vocabulary is 30,522. `gardner` (ID 11764) is absent from train and occurs in one dev and one test sentence. That is about 0.035% of either IID split, not a 35-point explanation. A train-vocabulary mask would make these examples impossible; therefore do not use held-out vocabulary to build an output mask. Gen additionally contains unseen `monastery` and 537 overlength sentences: truncation affects that split but does not explain IID failure or approximately zero across all gen categories.

Thus **tokenization and length truncation are ruled out as major IID plateau explanations** for the current JSON files. This is exact normalized sentence reconstruction on the project's meaning-deduplicated splits, not standard COGS semantic parsing. Preserve the current target text; do not silently make scoring semantic or remove punctuation to increase the number.

Other concrete issues:

1. `evaluate(..., limit=512)` always takes the first 512. Approximate binomial 95% half-width at 65% is **4.13 points** for 512 versus **1.75 points** for 2,862. Repeated runs on the same slice have no new sampling noise when deterministic, but checkpoint selection overfits that slice and it may be unrepresentative. Its mean whitespace length is 7.809 versus 7.662 on all dev. Compare checkpoints with paired predictions, not independent error bars.
2. **The current fresh-reader routine is invalid as a clean acquisition test.** The new reader is not a submodule of `model`; `model.eval()` never sets it to eval, so its dropout and BatchNorm stay active during evaluation. Then `evaluate` ends with `model.train()`. After the first 500-step evaluation, the supposedly frozen writer resumes training-mode Gumbel sampling and dropout when generating training examples. `no_grad()` does not prevent that. It also evaluates test every 500 updates. Fix all three issues before using a fresh-reader result.
3. `information_check` and `evaluate` unconditionally restore training mode. The main frozen training loop repairs writer mode before the next batch, so this does **not** invalidate the frozen-run dev numbers above. Replace these helpers with exact mode preservation anyway.
4. `save` contains weights/configuration but no optimizer, scheduler, RNG state, or step. `load` does not restore arbitrary slot/resolution changes from config. There is no resume flag. Re-running `train` starts over. Record these distinctions in the proposed continuation runner.
5. `paint()` changes resolution and contrast relative to training; `report()` evaluates the internal 192-pixel render. Exported-PNG accuracy is a separate endpoint, not a cause of the internal plateau. Also, automatic `report()` evaluates the entire 20,974-example gen split after every training run: avoid this expense and repeated test exposure during selection.

# 2. Candidate interventions

The table gives **guessed incremental gains over the best comparable incumbent**, conditional on the stated bottleneck, and marginal H100 minutes including ordinary checks. “Low confidence” means the mechanism is plausible but unmeasured here. Costs assume cached frozen inputs and actual runtime enforcement; they are not promises. These are alternatives, not a request to execute their summed cost.

| Intervention | Mechanism and evidence basis | Guessed exact gain; confidence | H100 min | Risk and interaction |
|---|---|---|---:|---|
| **1. Full-dev audit and mode fixes** | Correct fresh-reader modes; establish comparable deterministic evaluation. Direct code findings above. | Existing main reader: 0–1 point expected; high confidence that it does not explain 25 points. Fresh-reader correction: unknown. | 2–5 | May reveal optimistic selection rather than improve performance. Required for every other row. |
| **2. Longer frozen-reader training** | More supervised examples of the same fixed code; directly extends the successful freeze intervention. Start a fresh LR schedule, not the dying tail of the old schedule. | 0–5 points; medium-low, given 14k–20k slowing. | 8–18 for 10–20k updates | Overfitting; terminate on full-dev stagnation. Cache codes/images and reuse memory for CE/BOW. |
| **3. Lower decoder dropout/BOW and channel noise** | Once writer is fixed, preventing writer neglect is no longer the same optimization problem. Move closer to clean autoregressive evaluation. Try word dropout .1→0, BOW weight .1→0, predominantly clean images. | −2 to +6 points; medium-low. | 6–10 per 6k arm | Removing all regularization can hurt generalization. Compare to an identical-LR continuation retaining the original objective. |
| **4. Bigger reader** | Increase to d=384, 3 image-encoder layers, 4 decoder layers, 6 attention heads; keep angular resolution initially. Test whether the oracle-to-image gap reflects capacity. | 0–8 points; low; larger only if perception/fit diagnostics support it. | 20–28 for up to 12k updates | More memorization and slower throughput. Combine with slot supervision or teacher distillation; do not attribute a combined gain solely to size. |
| **5. Beam search** | Preserve alternate prefixes; measure gold-in-beam to separate ranking and search from missing information. | −2 to +5 points; low. | 1–4 | Length/EOS bias; larger beams need not help. Select width and normalization on dev. |
| **6. Candidate re-rendering** | Gold in the beam must reproduce the clean received raster under deterministic writer/rendering. Use the writer only on candidate strings. | 0–15 points; low; bounded by candidate coverage and collisions. | 2–6 | Cannot recover candidates never proposed. Distinct sentences can share images; do not query the writer with gold text at inference. Synergizes with beams and complementary readers. |
| **7. Distill a symbolic-code teacher into the visual reader** | The teacher learns code→text without visual errors. Student trains on images with gold CE plus teacher-logit KL; final reader never receives IDs. This is a proposed mechanism, justified only if the diagnostic teacher is better on dev. | 0–8 points; low; never claim teacher accuracy as the visual result. | 8–20 incremental | Copies teacher mistakes and possible memorization. Teacher trained on train only; retain gold CE. Can accompany row 4. |
| **8. Slot-supervised perception** | Frozen writer supplies exact train ring/mark labels for image→slot auxiliary heads. Unlike BOW, this teaches which visual distinctions carry the code. | 0–10 points; low, conditional on a large symbolic/visual gap. | 3–15 | May waste effort on redundant/invisible ring variants. Monitor whole-code exact and sentence exact; a two-stage hard parser compounds errors. Prefer soft auxiliary heads with the sentence decoder retained. |
| **9. Ensemble two readers of one code** | Average conditional token probabilities or union beams; helps only where their errors differ. | 0–4 points; low. | 10–25 to train, 2–4 to assess | Correlated errors; cannot repair lost information. Readers must consume the **same writer's images**. The freeze25/freeze40 pair is not a free valid ensemble of one transmitted logogram. |
| **10. Frozen-code self-training / additional text** | More independently sourced training sentences can be encoded by the writer with known input text as the exact target. That is supervised augmentation, stronger than pseudo-labeling its own outputs. | Existing train corpus alone: 0–2 points; new valid IID-like text: 0–10; low. | 10–25, excluding text sourcing | No new information from repeating confident pseudo-labels. Do not use dev/test/gen sentences, merge original excluded COGS paraphrases, or synthesize from held-out examples. No verified extra corpus is currently supplied, so defer. |
| **11. Test-time augmentation** | Average probabilities across identity and tiny image perturbations; may reduce visual instability. | −2 to +2 points; low. | 1–4 | Exact deployment images are already canonical; interpolation destroys details. Try identity/±1° only, not a random rotation of the code. Adds little if beam rerendering already works. |
| **12. Small-LR writer unfreezing with deterministic hard ST** | Hard argmax in every forward pass, soft surrogate only backward; use writer LR 1e−5 versus reader 1e−4, followed by re-freezing. Directly addresses code stationarity/deployment mismatch shown by freezing. | −10 to +15 points; low. | 12–18 for 6k joint + 6k reader steps | Can destroy the incumbent and remains biased. Branch checkpoints, constrain drift, run writer with dropout off, and rollback on dev decline/ink collapse. Add code→text auxiliary CE if the symbolic oracle is weak. |
| **13. Improve code before freezing / longer soft phase** | Evaluate deterministic-hard reconstruction during development and briefly adapt a reader at candidate freeze points. A soft loss can exploit mixture weights absent from the final code. | −5 to +8 points; low. | 18–35 | The 80k run with a 32k soft phase collapsed; it is not evidence that longer soft training is sufficient. Better freeze-point selection and a hard-forward curriculum matter more than duration alone. |
| **14. Entropy and usage regularizers** | Penalize per-example entropy while encouraging per-slot marginal diversity: `mean H(p) − H(mean_batch p)`, with separate blank/occupancy control. This discourages uncertain choices and constant codes rather than indiscriminately rewarding entropy. | −5 to +8 points; low. | 1–3 overhead inside a 12–25 minute writer arm | Can encode irrelevant distinctions, force visually indistinguishable marks, or destabilize margins. Uniform usage is not the target. Helps row 12 only if diagnostics show collapse/low usage; no effect on a frozen writer. |
| **15. Minimum-ink/occupancy floor** | Per-example hinge discourages disappearing messages during writer updates; log hard occupancy, not just differentiable expected ink. | Frozen code: exactly 0; writer retraining: 0–5 points as a collapse hedge; low. | <1 overhead | Ink is not information; all examples can emit the same dense glyph. Use a safety floor, not a demand for film-like density, and pair with reconstruction/diversity. |
| **16. Codebook/slot/geometry changes** | Prune confusable marks using train-only recognition confusions; compare same-code scale .25/.4. A 16-mark-plus-blank, 4-ring-plus-blank inventory at 24 slots still has 153.8 nominal bits. More slots or larger raster are other options. | −5 to +15 points; low. | 3–5 probe; 20–35 new training | Inventory reduction may discard meaning; additional slots increase overlap at fixed circumference. Do not spend on 48 slots at 192px without a channel probe. Geometry change requires reader recalibration and saved renderer metadata. |
| **17. VQ / FSQ alternative** | Learned nearest-neighbor quantization or bounded scalar quantization changes how discrete decisions are optimized. Recent sources below motivate alternatives to categorical ST, not an ink-specific performance forecast. | −20 to +10 points within this short budget; very low. | 30–60 plus implementation | There is currently no learned VQ codebook to reseed. Mapping a quantized vector through a new hard mark-ID lookup breaks the proposed differentiable path unless its surrogate is specified. Replacing the quantizer does not fix overlapping ink. Defer. |
| **18. REINFORCE with an action-independent baseline / ZGR** | REINFORCE can optimize real hard samples without differentiating the renderer; use per-sentence CE cost and detached greedy-policy cost as baseline. ZGR changes the categorical surrogate's bias/variance tradeoff. Sources below. | −10 to +10 points; very low. | 15–35 pilot, more to establish convergence | Sequence exact reward is initially too sparse; use dense CE first. Forty-eight categorical choices make credit assignment difficult. ZGR still samples hard actions and does not itself remove the train/deployment mismatch. Lower priority than deterministic ST. |
| **19. Preserve lexical identity at the writer** | Add a learned residual token embedding to MiniLM input, initialized with zero contribution, and/or a code→text auxiliary decoder. Tests whether frozen contextual features and training pressure discard surface detail. | 0–10 points; low, conditional on the code diagnosis. | 12–25 | More overfitting; auxiliary reconstruction must cross the **hard discrete** choices to force information through the actual bottleneck. No grammar labels or word→mark rules. Use only after the code probe. |

**What the literature actually supports.** [Shekhovtsov, ICML 2023](https://proceedings.mlr.press/v202/shekhovtsov23a.html) derives a computationally inexpensive zero-temperature Gumbel–Rao estimator and studies bias/variance; it does not establish higher exact reconstruction for this renderer. [Mentzer et al., FSQ, 2023 / ICLR 2024](https://arxiv.org/html/2309.15505v2) replace a learned VQ codebook with a bounded, rounded low-dimensional grid and obtain competitive results on their image tasks. Their favorable codebook-usage behavior does not establish a lower bound on information about sentences or on visually distinct ink choices. [Fifty et al., 2024 / ICLR 2025](https://arxiv.org/html/2410.06424v2) propagate gradients using a rotation/rescaling between an encoder vector and its nearest code; this presupposes a vector quantizer and is not a one-line replacement for this categorical renderer. For a VQ arm, start from the actual [VQ-VAE formulation](https://arxiv.org/abs/1711.00937), define commitment/codebook updates and the renderer-gradient path explicitly. A greedy-policy reward baseline is motivated by [Rennie et al., self-critical sequence training](https://arxiv.org/abs/1612.00563); here its proposed application is to writer-code sampling with sentence reconstruction cost. None of these papers supplies a credible numerical gain forecast on this task.

For clarity, a viable categorical policy-gradient pilot would minimize

\[
\mathrm{stopgrad}(\ell(z)-\ell(z_{\rm greedy}))\sum_{s,h}\log p_{s,h}(z_{s,h}\mid x),
\]

plus ordinary reader CE, with the reader fixed during each writer-gradient calculation and the baseline detached. The sign is for minimizing a **cost**. This specifies a mechanism; merely replacing ST with “RL” does not. For a VQ/FSQ proposal, a learned continuous decoder that can emit arbitrary pixels would violate the fixed-ink bottleneck. The final output must still select and render whole library pieces, with the additional hard mapping's backward estimator explicitly defined. That complication is why neither alternative is on the default budget path.

# 3. Pre-mortem

Assume the selected system finishes below 75%. The three likeliest explanations, in descending order of my current concern, are:

1. **The frozen protocol generalizes poorly, even when its symbols are visible.** Low train CE and much higher dev CE already point in this direction. Codes can be distinct yet arbitrarily organized; uniqueness will not rescue an unseen sentence. **Hedge:** run the discrete-code reader early, with a continuous MiniLM-token reconstruction control if it fails. Reserve 45 minutes for writer repair, including code→text CE through hard choices and an optional lexical residual. Never commit all $15 to the frozen reader. Preserve the incumbent so a failed repair cannot erase the best result.
2. **The reader cannot resolve visual distinctions reliably enough, or the renderer destroys them.** A 79% per-mark probe is a warning, though not a bound. More decoder layers cannot recover an erased mark. **Hedge:** require symbolic-vs-pixel and actual-code perception diagnostics; add slot supervision and, if indicated, a same-code rendering comparison. Reserve one compact visually selected inventory/layout attempt only when that diagnosis is strong. If the geometry demonstrably merges distinct required messages, increasing language-model capacity is the wrong response.
3. **Apparent progress is specific to teacher forcing, one reader, or the 512-example slice.** A larger reader may lower CE without improving free-running exact; beam may confidently prefer the wrong sentence; the existing fresh-reader code can contaminate the comparison. **Hedge:** evaluate full-dev greedy exact, beam coverage and selected beam exact separately; preserve inference modes; choose checkpoints on dev only; reserve 35 minutes for an independently initialized reader on the final frozen writer. A reranker-assisted 90% result and a standalone fresh-reader 70% result must both be reported, not combined into a claim that acquisition reached 90%.

These hedges change the allocation materially: only 18 minutes are committed to blind continuation, while diagnostics, writer rescue, and acquisition remain protected. No extra soft-only 80k run, VQ rewrite, or REINFORCE search is purchased by default.

# 4. Final plan

## Budget and implementation contract

Using the **supplied** H100 price of $3.95/hour, $15 buys **227.85 billed GPU minutes**. At .06 seconds/update that is 227,848 bare training updates, or approximately 15,190 updates/dollar; evaluation, loading, rendering caches and larger models consume part of this allowance. The frozen logs give approximately .054 seconds/update including their existing evaluation cadence; the base log takes 2,259 seconds for 30k, approximately .075 seconds/update. These are better budgeting anchors than assuming every future update takes .06 seconds.

Use **one H100 allocation at a time**, a 220-minute total ceiling ($14.48), and count all time billed after this plan starts, including already-running incumbents. The remaining $0.52 is outside the planned allocation. The table's caps include evaluation within each stage; the separate final-evaluation allocation is for final models only. Costs below are rounded.

| Stage | Maximum H100 minutes | Maximum cost | Planned work |
|---|---:|---:|---|
| R0: audit/cache/decode screen | 12 | $0.79 | Both actual frozen checkpoints; full-dev greedy, beam and collisions |
| R1: matched reader continuations | 18 | $1.19 | Two branches, 6k updates each |
| R2: discrete-code diagnostic | 12 | $0.79 | Up to 10k code-reader updates; optional 2k continuous control |
| R3: visual-reader repair | 28 | $1.84 | Short perception probe + up to 12k reader updates |
| R4: writer repair, conditional | 45 | $2.96 | Two bounded attempts, stopping failed arms early |
| R5: consolidate selected system | 30 | $1.98 | Up to 20k reader updates or one complementary reader |
| R6: fresh-reader acquisition | 35 | $2.30 | Up to 20k independently initialized reader updates |
| R7: locked final evaluation | 15 | $0.99 | Full test once, selected original/fresh systems and decoding variants |
| Running-job/startup/slowdown reserve | 25 | $1.65 | Charge any incumbent tail here; no hidden extra runs |
| **Total ceiling** | **220** | **$14.48** | Unused stage time is not a requirement to spend |

**Before any paid run, implement the following local changes; this document does not modify implementation files.** The specifications below are proposed code changes, not flags falsely claimed to exist today. The current `train --steps ...` cannot perform these continuations.

1. Add a budget runner with explicit input checkpoint, output directory, stage, update cap and wall-time cap. Add it to `modal_app.py`'s uploaded files if it is a separate file. Load the real `/runs/freeze25/model.pt` and `/runs/freeze40/model.pt`; inspect and record their configurations and steps. Never overwrite an incumbent. Stop its existing job after a safe checkpoint if it is no longer providing useful results; otherwise subtract its remaining billed time from the 25-minute reserve.
2. Cache train/dev **deterministic hard** codes and token IDs, preserving exact existing text. Cache float32 192×192 images if useful: train images require approximately 3.49 GB, so do not silently quantize them to uint8 and change the task. Writer-repair stages may cache frozen MiniLM token vectors instead. Build caches with writer/encoder in eval mode. Generate test inputs only at the locked final evaluation. The tokenizer audit above was a format check, not model selection on test.
3. Use `model.eval()` and explicit `reader.eval()` in every evaluator; restore each module's prior training flag, including child exceptions, afterward. Frozen training must explicitly keep writer/encoder eval on **every** iteration and set only the trained reader to train. `no_grad` alone is insufficient. Select the fresh reader on dev, not test.
4. Factor `Reader.forward` into `memory(img)` and `decode_from_memory(mem, ids)`; compute memory once for CE/BOW/auxiliary heads. Keep the full pretrained tokenizer vocabulary in the default plan. Suppress only invalid output control tokens (PAD, CLS, MASK and unused control IDs), while retaining SEP/EOS and ordinary lexical tokens; measure this decoding change separately. No held-out-derived vocabulary whitelist or gold length.
5. Save writer, reader, renderer inventory/configuration, optimizer, scheduler, RNG states, update count and data hashes. `load` must reconstruct any changed dimensions, slots and renderer before loading weights. Save latest as well as best. No automatic full `report()` after exploratory stages; it unnecessarily touches test/gen.
6. Use AdamW, weight decay .01 and gradient clipping 1.0 unless specified below; batch 64. Use a fixed shuffled 512-example **dev** subset for inexpensive checks every 2k updates, and full dev at stage endpoints and branch decisions. Keep paired predictions. Full-dev greedy exact chooses training checkpoints; decoding settings are then selected on full dev with the tiny specified grids. Do not select on training loss or the information-check gap.
7. Time 200 representative updates including auxiliary teacher work, then estimate whether the stage fits. Reduce updates if necessary; the wall cap wins. Skip optional experiments before consuming R6/R7 time. If actual reserve spending exceeds 25 minutes, shorten R4 then R5 by the excess. Never infer that a nominal step count authorizes exceeding $15.

## R0 — Establish the real incumbent and the cheapest possible decoding gain

Evaluate both actual frozen checkpoints on full dev and the fixed 2,000-example train sample. Export per-example normalized prediction, sentence likelihood, first-error position, hard choices, ink and length. Compute collision summaries, without interpreting noncollisions as success. Choose the stronger **full-dev** writer/reader as incumbent; the 1.56-point difference on the current 512 slice is not enough to preselect freeze25 confidently.

Run beam 4/8 with length exponent 0/.6, then beam 32 only if beam-8 gold coverage exceeds greedy by at least 5 points. Cache candidate texts and probabilities. Batch their writer evaluations; `writer.eval()` must remain true. Define

`d(x, y) = mean(abs(x - render(writer(y)))) / max(mean(abs(x)), 1e-6)`.

First prefer a **unique** candidate whose rendered raster has maximum absolute error below `1e-6`; verify that tolerance by re-rendering repeated train examples across batch sizes. If several match, retain the highest beam probability. Otherwise compare pure beam selection with `log P(y|x) / length(y)**alpha - lambda*d(x,y)`, using only `lambda ∈ {1, 5, 20}` and `alpha ∈ {0, .6}`. The score and tolerance are guesses to validate, not an established metric. The input to this routine is the image plus model-generated candidates; **never the original code IDs, gold sentence, or gold length**.

**Decisions:** if selected full-dev exact already reaches 90%, skip R1–R5 and do R6/R7. If beam coverage is below 80%, stop widening beams and move to representation training. If coverage exceeds 90% but selected exact does not, prioritize reranking/perception diagnosis; do not infer that another 30k CE updates is necessary. Keep raw greedy and reranked results separate.

## R1 — Test objective alignment before buying capacity

Clone the same incumbent weights into two arms. Freeze writer and renderer, reset reader optimizer, LR `1e-4`, 300-update linear warmup, cosine decay to `1e-5` over **6,000** updates. New optimizer is intentional because the old optimizer state is unavailable.

- **R1A control:** original word dropout .30, BOW coefficient 1.0 in `CE + coefficient * 100 * BCE`, original `channel()` on every training image.
- **R1B intervention:** word dropout .10 for updates 1–3,000, then zero; BOW coefficient .10 for updates 1–3,000, then zero. Half the minibatches use clean images; half use at most ±1° rotation, ±1% scale, ±.5% shift and Gaussian sigma .03. Preserve the exact clean deployment renderer.

Use identical batch-order seed 17 for the comparison, with separately controlled augmentation RNG. Compare full-dev exact and paired changed examples at 3k and 6k, plus teacher-forced CE. Keep the best checkpoint, including the starting incumbent. **If neither arm improves full-dev exact by at least 1 point and train exact is above 98%, do not allocate R5 to simply extending the same configuration.** A smaller improvement may still be retained as the incumbent; the threshold controls spending, not whether a result is real. If one arm gains at least 2 points and continues improving at 6k, it earns consolidation time later.

## R2 — Determine whether the code itself is worth preserving

For the selected writer, train a new reader whose image-memory input is replaced by 24 vectors `E_ring[id_r] + E_mark[id_m] + E_slot[s]`, all width 256. Use two 256-wide encoder layers and three decoder layers, four attention heads, the same tokenizer/output space, no BOW, word dropout .1 for the first 2k then zero. AdamW peak LR `3e-4`, 500-step warmup, cosine to `3e-5`; seed 23; **10k updates maximum**, checking at 2k/6k/10k. Train on train only. Do not initialize this probe from MiniLM text memories or supply them as targets at inference.

**Decisions:**

- **Symbolic dev ≥95%, visual ≤85%:** preserve this writer; prioritize R3. There is an empirically demonstrated recoverable code and at least a 10-point interface/reader gap.
- **Symbolic ≥90% with a ≥5-point visual gap:** still prioritize R3, but 90% visual is not guaranteed.
- **Symbolic ≤80% with train ≥98%:** prioritize R4; allocate at most the short perception probe portion of R3 unless it reveals an independent easy fix.
- **Symbolic train <95%:** spend up to 2k updates on a diagnostic decoder reading the unquantized MiniLM token vectors, with proper masking and learned projection. Its job is to expose decoder/input/optimization trouble. Failure here invalidates a strong code-ceiling diagnosis. Also inspect the 256-example overfit control. Do not deploy this bypass or count it toward the reconstruction goal.
- **Intermediate cases:** run bounded R3 and choose by measured improvement per minute. These thresholds are allocation heuristics, not hypothesis-test cutoffs.

## R3 — Repair the visual interface if the code supports it

First spend at most 2k updates/4 minutes on actual-code slot classification. Reuse PolarEyes; pool each adjacent angular-token pair to a 24-slot representation and add separate ring/mark heads. Train CE averaged over 24 slots, reporting present-mark recall and whole-code exact. If performance is poor, compare actual-code clean versus perturbed images; a weak random-code result alone does not justify an inventory rewrite.

Default repair: continue the best d=256 reader with

`L = text_CE + .25 * mean(slot_ring_CE + slot_mark_CE)`

for up to **12k** updates; LR `1e-4→1e-5`, 300-step warmup, the winning R1 channel/dropout schedule, no BOW. Remove slot-auxiliary weight for the final 2k updates. These are learned visual labels from the writer, not grammar supervision. The final decoder continues to use pixels, not oracle slots.

If the tiny-fit/perception results indicate insufficient capacity, use a fresh **d=384 / image-encoder depth 3 / decoder depth 4 / heads 6** reader instead, LR `3e-4→3e-5`, 500 warmup, seed 29, within the same 12k/28-minute cap. Adjust CNN output, angular positions, token embeddings and decoder dimensions together. Do not claim its dimensions are available CLI flags in the current code.

If the symbolic teacher is at least 5 points better on dev, optionally add `.25 * T² * KL(teacher || student)`, `T=2`, on nonpad text positions under the same gold prefix, while retaining full gold CE. Teacher sees train codes; student sees only train images. Count teacher inference in the stage timer; cache teacher outputs or omit distillation if it would exceed the cap. Do not add this optional loss if the teacher is not demonstrably stronger.

**Decision:** retain R3 only if full-dev exact improves; otherwise restore R1. If symbolic exceeds pixels by at least 10 points after the stage, use the geometry variant of R4B. If full-dev reaches 90%, stop development and reserve the rest for fresh acquisition and final evaluation.

## R4 — Improve what is written, with rollback

Run this stage only if the chosen code/reader still needs it. It has two attempts inside a **combined 45-minute ceiling**. Do not resume the collapsed `long` writer.

**R4A: 6k deterministic-hard joint updates, then 6k frozen-reader updates.** Start from the best surviving incumbent. Keep the writer in eval mode to disable its dropout, but enable gradients and call `writer.logits` explicitly. For each head use:

```python
p = logits.softmax(-1)                 # backward temperature 1.0
q = one_hot(logits.argmax(-1), K).float()
choice = q + p - p.detach()            # hard deterministic forward
```

Do not call the existing training-mode `Writer.forward`, which adds Gumbel noise. Use writer LR **1e−5**, reader LR **1e−4**, 300-step warmup, cosine to one tenth of each; freeze encoder and renderer. This is a biased optimization proposal, not an unbiased estimator claim. Keep the winning reader/channel schedule, but use clean images for at least half of batches.

For a code-limited diagnosis, add the R2 symbolic decoder as an auxiliary branch taking the same hard choices with straight-through gradients: `L = image_text_CE + .25 * code_text_CE`. Its token embeddings are multiplied by the ST choices; no soft logits are forwarded as information. Initially freeze this auxiliary decoder for 1k updates, then train it at `1e−4`. This specifically rewards retaining reconstructive information in the deployed discrete message. Without that diagnosis, omit this loss.

Use a small reference-policy anchor, `.01 * mean KL(p_initial || p_current)`, averaged across heads and slots on train inputs, to discourage gratuitous protocol drift. For collapse protection add `.05 * mean(relu((.02 - per_image_ink)/.02)**2)`. The .02 floor is a guess and below the incumbent's measured 6–7% mean; it is not a request to increase frozen ink. Log hard usage and per-example ink percentiles.

Only if usage actually shrinks, add `.01 * mean_{slot,head}[mean_batch H(p) - H(mean_batch p)]`, with entropies normalized by `log K`, and a separate occupancy report. Do not maximize conditional entropy alone. All of these weights are provisional; run **one specified setting**, not a grid hidden in the budget.

Evaluate every 1k updates. Roll back/stop an arm if full-dev exact drops by more than 3 points on two checks, mean hard ink falls below half its initial value, or more than 10% of the 48 hard decisions change on a fixed train audit set between successive 500-step checks without a dev gain. Those are conservative spending guards. After joint updates, freeze the best writer, regenerate its train cache, reset reader LR to `1e−4`, and train 6k updates with cosine decay. Judge the repaired protocol **after** this reader adaptation, not solely during co-adaptation.

**R4B: one targeted fallback, only if R4A fails.**

- If **code-limited**, start again from the preserved incumbent, add a trainable WordPiece embedding residual into the writer input, with its output projection initialized to zero, and increase hard-code auxiliary CE weight to **1.0**. Repeat the 6k joint + 6k frozen schedule. This preserves the initial function and supplies a learnable lexical path; no token is assigned to a mark by hand. Keep all hard-forward and rollback rules.
- If **visually limited**, retain the frozen IDs and try the alternative mark scale (.25↔.4), keeping every actual mark from the library; retrain/adapt the visual reader for up to **8k** updates. Only if measured train-only confusions show irreducible inventory ambiguity, replace this scale trial with a compact inventory: 16 nonblank marks and 4 nonblank ring segments selected by visual distinguishability, plus blanks, still 24 slots. This requires new output heads and a new learned protocol: allocate at most **12k joint updates + 8k frozen updates**, all hard-forward, with LR writer `1e−4` and reader `3e−4`, and the same auxiliary code reconstruction. Stop after 4k if train reconstruction is not improving and usage has collapsed. Do not pretend old reader weights or old code meanings transfer unchanged. The combined R4 timer can shorten this fallback substantially; preserve R6/R7.

Do **not** simultaneously add slots, change resolution, change inventory and swap gradient estimators. The narrower fallback is already a high-risk use of the remaining budget. A longer soft-only phase is deliberately excluded because it has no evidence of solving the deployment mismatch here.

## R5 — Consolidate only a measured improvement

Freeze the best writer selected on full dev. If the best reader is still gaining at least 1 point per 4k updates, continue up to **20k** updates, LR `5e−5→5e−6`, no BOW, no word dropout during the final 4k, retaining the winning channel setting. Stop after two 4k intervals each below .5-point improvement and no favorable paired error trend.

If reader loss is converged but different earlier readers of this **same code** have complementary errors, use the allocation for one second initialization instead, up to 15k updates at the selected architecture/schedule. Keep a two-reader probability ensemble only if dev exact gains at least 1 point. Measure the oracle union of their correct predictions before paying for elaborate fusion. Do not combine a freeze25 reader and a freeze40 reader on their separate transmitted images: that sends two messages.

Repeat the small beam/rerender selection on the chosen final code. Try identity/±1° TTA only if dev perception errors show perturbation sensitivity and at least 2 minutes remain in this stage. Keep it only for a measured dev gain. No pseudo-label self-training or unverified additional-text generation in the default plan.

## R6 — Fresh-reader acquisition is a protected deliverable

Freeze the final writer and all rendering choices. Initialize a new reader at **seed 101**, independent of any selected reader; same architecture as the successful visual model, random weights, no copied teacher or reader parameters. Train on train sentence/image pairs only, batch 64, AdamW LR `3e−4`, 500-step warmup, cosine to `3e−5`, clipping 1.0, up to **20k updates or 35 minutes**. Use ordinary text CE, the selected channel schedule, and word dropout .1 for 2k then zero. No symbolic teacher or slot-label loss in this acquisition test: this measures learning from paired logograms and text. If a supervised-symbol acquisition variant is later desired, label it separately.

Evaluate on dev at 2k/5k/10k/15k/20k; choose the best by full-dev greedy exact. Keep learning curves and train exact. If it is still rising at the wall cap, report incomplete convergence; do not call it unlearnable. Use any genuinely unspent earlier-stage budget to extend this run only while preserving the total 220-minute ceiling and R7. A single fresh seed is a budget-constrained acquisition demonstration, not a seed-robust estimate.

## R7 — Lock and report the answer

Lock the writer, renderer, reader checkpoints, beam parameters and reranking rule using dev; then evaluate **full IID test once** for the selected original reader and independent fresh reader. Report greedy, selected beam, and the full selected inference system; explicitly mark any use of a frozen-writer reranker or reader ensemble. Also report full-dev and sampled-train exact, word/token errors, test N, binomial/Wilson intervals and billed minutes/dollars. At 90% on 2,858 examples the approximate 95% half-width is 1.1 points: distinguish an observed ≥90% score from evidence that population accuracy exceeds 90%.

If the selected system achieves 90% but the fresh reader does not, state that the reconstruction target was met but acquisition did not reach that level within the allotted training. If both remain below 75%, report the oracle/perception/beam diagnostics that explain the failure; do not spend the reserve on an unplanned estimator sweep. Gen evaluation is optional only after the IID and acquisition deliverables fit the budget, and must disclose the 40-token truncation issue. There is no evidence here that improving IID reconstruction will resolve COGS systematic generalization.
