# Measured diagnosis of freeze25 (final checkpoint), local run of v3/diagnose.py

Final metrics: train exact 99.6% (word acc 99.9%), dev 70.4%, test 71.7% (word acc 91.8%), COGS gen ~0% (word acc 50%).
freeze40: train 99.4%, test 69.2%.

- Code collisions among the 2,858 test sentences: **0**. Test codes identical to a different train sentence's code: **0**.
- Word error rate on test by the word's frequency in train: 1-9 occurrences **93.2%**, 10-99 **27.9%**, 100-999 **7.6%**, 1000+ **4.2%**.
- Failed test sentences by number of wrong words: 1 wrong: 543, 2: 99, 3+: 167 (of 809 failures).
- Most frequent swaps: a<->the (110), beside/on/in confusions (~70), liked/said/hoped/noticed (mental verbs), chalk->pencil, abigail->emma, and rare nouns/names replaced by frequent ones (kid->journalist, moose->tiger, gravel->plate).

Interpretation (Claude): the frozen code is injective and the reader reads training sentences almost perfectly; the generalization gap is lexical — rare lemmas are not read correctly from codes the reader has seen too few times. Proposed lever: lexical substitution data augmentation (swap lemmas within the same type/frame, from the COGS graphs) so every lemma appears hundreds of times; plus definiteness and preposition errors.
