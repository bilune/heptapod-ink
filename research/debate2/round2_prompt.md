# Heptapod v4: discrete bottleneck fails to learn; need a recipe

Context: v4/heptapod_t5.py (read it). Pipeline: T5-small encoder -> SlotWriter (36 learned queries, 2-layer transformer decoder over encoder states) -> per-slot code -> Adapter (code -> 256-d embedding + slot pos, 2-layer encoder, Linear->512, LayerNorm) -> pretrained T5-small decoder. Later stage: frozen code is rendered as 36 film-ink marks on a ring; eyes (polar CNN) recognise marks at ~91% per slot with 81-82 usable marks. Data: Tatoeba English <=12 words, 20% random word combos (vocab 32k). Goal: >=30% exact / >=60% word acc on natural dev, >=30% word acc on random combos, through the rendered ring. Budget left: ~USD 13 of H100 (~3h).

Measured (local M4, batch 32, random-combo-only data, word accuracy on 128 held-out random combos, 1000 steps):
- direct (T5 encoder states -> decoder): 100% by step 200.
- slots256 (36 slots x 256 continuous, no quantization): 0% until step 800, then 30.5% at step 1000 (phase transition, still rising).
- slots4 (36 x 4 continuous tanh, i.e. the FSQ pre-rounding width): ~0% at 1000 steps; same without the final LayerNorm.
Earlier H100 run with categorical 82-way argmax-STE (soft 500 / gumbel 1500 / argmax after), 6000 steps batch 128 on mixed data: natural dev word acc ~22%, random ~0.5%; decoder loss with the writer's soft probabilities was much lower than with argmax (dev 0.84 vs 1.87), i.e. information lived in soft mixtures. Local 7-min A/B of FSQ [3,3,3,3] vs categorical: both ~0% on random combos.

Question: what is the most reliable recipe, within budget, to get a 36-slot x ~81-symbol discrete code that reconstructs open English (including random combos) well? Candidates I see:
(a) VQ with a learned 256-d codebook of 81 entries per slot (or shared), STE + commitment, warm-started from a slots256 continuous autoencoder with k-means codebook init;
(b) categorical with wide embeddings but long soft/gumbel phase plus entropy penalty to close the soft/hard gap;
(c) progressive squeeze / distillation from the continuous model;
(d) anything better (e.g. product of 2 codebooks of 9 per slot, residual VQ, more slots, larger symbol alphabets by combining mark+ring, multi-token per slot).
Also: why might slots4 be so much slower than slots256 (optimization vs capacity), and how to verify fast.

Answer concretely: chosen recipe, hyperparameters, a fast local check (<10 min on M4) that would falsify it, and the H100 plan with stop rules. Be brief (<700 words). Write to research/debate2/round2_codex.md.
