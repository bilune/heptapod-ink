# Round 1: Claude's independent plan

## Diagnosis of the ~65% plateau (frozen runs)

The plateau arrived within ~4k reader steps of freezing and stayed flat, with word accuracy ~90%. Candidate limits:

1. **Frozen code ambiguity (my main suspect).** If two distinct sentences map to identical or near-identical argmax codes, no reader can separate them. The code was frozen at a point where the soft-phase writer read only 15-33% of sentences exactly, so its argmax code was never optimised as a hard code. Cheap test: encode all dev and train sentences, count exact code collisions and the Hamming-distance distribution between codes of sentences that differ by one word. Also run the reader on train: if train exact is also ~65%, the ceiling is in the code, not in reader generalisation.
2. **Lexical bottleneck.** 90% word accuracy at 65% exact means roughly one wrong word per failed sentence. With 645 lemmas (many nouns and names seen rarely), the code must give each word a distinct, legible pattern. Cheap test: per-word error rates against train frequency, and a confusion list (which words get swapped). If errors concentrate on rare nouns or names, the fix is on the writer's lexical encoding.
3. **Greedy decoding.** Beam search (width 4-8) usually buys a few points of exact match when per-token accuracy is high. Zero training cost.
4. **Reader.** The fast plateau suggests the reader is not the binding constraint, but the cosine schedule after freezing decays the lr; a longer reader phase with a restart could add a little.
5. **Eval noise.** Dev numbers use 512 sentences (±2 points); the full evaluation at the end settles it.

## Plan

1. **Measure first, on the existing freeze25 checkpoint** (minutes, nearly free): train vs dev exact, code collisions, per-word errors, beam search. This decides between "the code is the ceiling" and "the reader is".
2. **Better code before freezing** (the main lever if collisions/lexical errors dominate):
   - longer soft phase (long reached 32% at the switch with a 32k soft phase vs 15% for freeze40 at 12k);
   - an ink/marks floor so the writer cannot drift towards empty rings (the collapse seen in `long`);
   - a *hard-code warm-up* before freezing: ramp the probability of hard samples from 0 to 1 over ~5k steps instead of a switch, so the argmax code itself gets optimised while both agents still adapt;
   - then freeze and train the reader to convergence.
3. **Unfreeze-and-polish:** after the reader converges on the frozen code, unfreeze the writer with lr ~1e-5 (reader keeps training) for a few thousand steps, to repair specific collisions without destabilising the reader. Stop if dev drops.
4. **Beam search plus a fresh reader** at the end: a reader trained from scratch on the final frozen code, which doubles as the "it's a language, not a private code" test.

Budget: one measurement pass (~USD 0.5), one main run of ~60k steps with a 30k soft phase and 5k ramp (~USD 4.5), polish 10k steps (~USD 0.7), fresh reader 20k steps (~USD 1.5). Total ~USD 7, leaving room for one retry.
