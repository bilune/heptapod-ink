# Final results, locked exam (2026-11-09, evaluated once)

System: v4 code continued 40 min on the mixed natural pool (FineWeb 55% / Wikipedia 15% / CC-News 10% / Tatoeba 20%, no random combos), eyes adapted 15 min; sender-side search (K=16 nearest, R<=20, keyed views, visual guard). Read from 16-bit PNG through the eyes. Fingerprints verified (v4/LOCKED_EXAM.json). Wilson 95% intervals.

| set | plain exact | search exact | search, unseen perturbation | shuffled image | search words / WER |
| --- | --- | --- | --- | --- | --- |
| locked natural, 2,000 (UD GUM/PUD/LinES/ParTUT) | 33.5% (31.4-35.5) | **81.0% (79.2-82.7)**, cased 80.5% | 75.5% (73.6-77.4) | 1.2% | 92.9% / 5.9% |
| locked random 3-10 words, 2,000 | 4.1% | 36.8% (34.7-38.9) | 29.8% | 0.6% | 61.3% / 34.5% |
| locked uniform 10 words, 1,000 | 0.0% | 0.0% (0-0.4) | 0.0% | 0.0% | 17.3% words |

Search exact by corpus: LinES 86.3%, ParTUT 80.5%, GUM 78.6%, PUD 70.4%. By length: 3-5 words 94.4%, 6-8 87.8%, 9-12 67.9%.
Remaining natural errors (380/2000 cased): other words 129, names/capitalised words 155, numbers 37, punctuation/quotes only 59, case only 10.

## Fresh reader (dev panels, plain codes, image read by frozen eyes)
New mark embeddings + adapter + base T5-small, nothing inherited. Exact (words): 1k 0% (10%); 5k 0% (22%); 20k 7.8% (45%) web / 24.6% (68%) Tatoeba; 200k (12k steps only) 25.4% (65%) web / 58.6% (89%) Tatoeba. Original reader: 41.8% (76%) / 70.7% (93%). Searched rings do not transfer (fresh reader 23.4% on them).

## Structure probes (v4/probe_structure.py, plain writer)
- Word swap (10 pairs x 50 Tatoeba sentences): one word changes 4.8 of 36 slots on average; the changed (slot, mark) items for the same swap overlap across sentences (Jaccard 0.13; Tom->Mary 0.30) vs 0.004 between different swaps; top-5 slots carry 33-55% of a swap's changes. Swapped word read back 72-96%.
- Poke (one slot replaced, 120 dev sentences x 36 slots): 83% no change, 11% one word, 2% half or more. Within-slot spread of affected word position 0.21 vs 0.29 uniform; some slots act on sentence start (e.g. 9, 35, 6) and several on the end (14, 28, 33).
Reading: a distributed but not holistic code; word identity has a partly consistent, localized signature; slots specialise loosely by sentence position.

## Grammar probe (v4/grammar_probe.py; plain writer; 60 fresh template pairs per operation; consensus edit learned on 30, applied to the other 30 base rings)
| operation | edit size (slots) | exact transformed (symbolic / through image) | marker present | random edit same size | other op's edit |
| --- | --- | --- | --- | --- | --- |
| plural subject ("The teacher is" -> "The teachers are") | 1 (slot 26) | 96.7% / 100% | 100% | 0% | 0% |
| past tense | 2 (0, 26) | 60% / 60% | 77% / 73% | 3% | 0% |
| I -> she (with agreement) | 8 | 50% / 47% | 100% | 0% | 0% |
| yes/no question | 10 | 30% / 30% | 100% ('?') | 0% | 0% |
| negation | 15 | 0% / 0% | 100% ("don't") | 3% | 0% |
Consistency: within-operation Jaccard of edits 0.18-0.62 vs 0.004-0.013 between operations; slot 26 takes part in every operation.
Caveats: narrow templates (consensus edits learned and tested on the same sentence shapes); exact failures for negation/question mostly change the verb/object or subject while the grammatical marker is right.
