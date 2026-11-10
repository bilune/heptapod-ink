# Heptapod v4 VQ run: plateau and slot collapse (live run, decide fast)

Your round-2 recipe is running on H100 (v4/heptapod_t5.py, --quant vq; read stage1, encode, vq_update, vq_init). Batch 128, 50% random combos, prefix dropout .15, commitment .25, EMA .99, reseed entries unused 200 updates, alpha (measured confusion channel) ramped 0->1 at VQ steps 3000-3500.

Continuous warm-up (4000 steps): natural dev 99.2% exact, random word acc 99.3%.
After k-means init and VQ (symbolic, 512 dev items each; min_marks = fewest distinct marks used by any slot over 256 dev sentences):
step 5000 nat 23% exact/65% words, rand 17% words, min_marks 9
step 7000 nat 35%/74%, rand 25%, min_marks 7
step 8000 (alpha=1) nat 37%/75%, rand 28%, min 7
step 9000 nat 36%/73%, rand 29%, min 6
step 10000 nat 32%/71%, rand 26%, min 1
step 11000 nat 34%/73%, rand 26%, min 2
Plateau; some slots collapse to 1-2 marks. Stage-1 midpoint gate (rand >=40% words) will fail at this rate. Stage 1 ends at 60 min, then stage 2 (eyes) starts automatically. Budget left ~USD 10.

Question: the single most effective change to raise usable code rate (rand word acc) for a continuation run that resumes from the current stage1 checkpoint (writer, adapter, T5, books saved). Options: batch-entropy regularizer on soft assignments per slot; periodic k-means re-init of collapsed slots; lower commitment; higher LR for writer; residual/second-stage cheap trick; more VQ time; or accept and move to vision. Give hyperparameters and a 15-minute H100 test with a pass/fail rule. <400 words. Write to research/debate2/round3_codex.md.
