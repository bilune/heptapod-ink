> Scope note (2026-11-08): the goal is a language between models. Human readability is out of scope; steps 5 (human decipherment kit) and 6 (printed photos) are dropped. Outsider learnability is tested with fresh and cross-seed model listeners only.

Goal: validate (or falsify) the core claim of Heptapod Ink: that an order-free, circular ink script invented by two networks can be a real compositional language, not a private code, and that being circular buys something over a linear script.

Start from raw material, not my toy vocabulary:
- Meanings: the COGS dataset (github.com/najoungkim/COGS), English sentences paired with logical forms, using its generalization split.
- Ink: the 38 Arrival logograms already in reference (Wolfram repo, CC BY-NC, local use only) and the existing whole-mark library. Read Wolfram's featureDetection-01.nb for how the production analysed them.
- Method references: Chaabouni et al. 2020 "Compositionality and Generalization in Emergent Languages", Kirby, Cornish & Smith 2008 on iterated learning, and Jessica Coon's paper on the linguistics of Arrival. Read the methods sections, not the abstracts.

Build it in v3:
1. Reconstruction game: the speaker encodes a COGS logical form as one ring logogram (the number of slots can grow with sentence complexity). The listener must decode the full logical form. Report exact match on the train, dev and generalization splits.
2. Linear control: the same speaker, listener budget, mark library and channel noise, but marks laid on a line and read in order. Match capacity in bits.
3. Iterated learning: periodically replace the listener with a fresh one so the language must stay learnable.
4. Metrics: topographic similarity, positional disentanglement, bag-of-symbols disentanglement, plus a partial-reveal curve (accuracy vs fraction of the ring shown).
5. Blind decipherment: give a vision model 50 logograms with their translations and ask it to read 20 unseen ones. Prepare the same kit as a printable sheet so a human can try.
6. Physical channel: render logograms, let me print them and take phone photos, and measure listener accuracy on the photos.

External checks and pass criteria:
- COGS generalization split: circular >= linear at equal capacity. Fail if circular is worse, or if accuracy collapses on unseen structures.
- Blind decipherment: >50% of meaning pieces correct on unseen logograms. Fail if no outside reader beats chance by a wide margin.
- Phone photos: >90% read correctly. Fail below that, since it would suggest pixel-level steganography.

Report each check as pass or fail with numbers, and state plainly which parts of the claim survived.
