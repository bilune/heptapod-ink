**Choose a per-slot batch-entropy regularizer**, with a sample-sharpness term to discourage uniformly ambiguous assignments. This is my best short-budget bet, not a demonstrated cure. Existing reseeding has not prevented collapse; periodic k-means changes learned symbol meanings without addressing the training incentive. More time, higher writer LR, or lower commitment offers no direct utilization pressure. Residual codes require additional transmitted symbols; vision cannot restore discarded information.

Compute auxiliary assignments from **non-detached `raw`**, before corruption, using FP32 distances to detached books:

`d[b,s,k] = mean_D((raw[b,s] - books[s,k])²)`

`p = softmax(-d / tau_s); pbar[s] = mean_batch(p[:,s])`

`L = CE + .25*commit_MSE + lambda*mean_slots(mean_batch(H(p)) - H(pbar))`

Use natural-log entropy; average over slots, never pool their histograms. Ramp `lambda` from 0 to **0.1 over 200 updates**, then hold. Calibrate each fixed `tau_s` once on training activations by bisection to median assignment entropy **ln(4)**; this avoids assuming a latent distance scale. Keep hard nearest-neighbor forward/STE and EMA updates unchanged. The entropy objective has precedent in [MAGVIT-v2](https://arxiv.org/html/2310.05737v3); transferring it here is an engineering hypothesis.

Retain batch 128, 50% random combinations, prefix dropout .15, commitment .25, EMA .99, reseeding 200, writer/adapter LR 3e−4, pretrained LR 2e−5, clipping 1, and **alpha=1**. Changing corruption simultaneously would confound recovery.

**Resume correctly:** `--resume stage1.pt` currently skips to stage 2. Explicitly re-enter stage 1 in VQ mode, bypass warm-up **and `vq_init`**, preserve books/EMA/counts/mark mapping, bypass the initial pretrained freeze, and disable automatic stage 2. `warm_steps=0` alone still reinitializes books. Optimizer state is absent, so rebuild AdamW. Re-evaluate the loaded checkpoint: `stage1.pt` saves best natural exact, not necessarily step 11000.

**15-minute H100 cap:** reserve minutes 0–2 for loading, calibration, and baseline; 2–12 for training; 12–15 for final evaluation. Use fixed 512-item natural/random dev sets, greedy hard symbolic decoding, and identical seeded confusion-channel evaluation. Track per-slot hard entropy/perplexity and usage on 256 fixed sentences.

**Pass:** clean random word accuracy ≥40% and ≥10 percentage points above baseline; natural words ≥70%; measured-channel random accuracy does not regress; minimum hard usage ≥8 marks. Otherwise stop this rescue and automatic vision spending. Soft entropy alone never counts as success. Passing licenses continued stage-1 work, not automatic vision readiness. These are proposed thresholds, not measured results.
