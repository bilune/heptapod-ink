# History: v0 to v4

Each folder is one attempt, kept as it was so the path is visible. Every step was replaced because
it answered its question and showed the next limitation.

## v0: relations in a 2D field

[v0/](../v0/README.md). The starting point was a broader question: can a model reason with 2D
representations instead of text? v0 is a tiny NumPy prototype (generated with ChatGPT's help)
that composes two relations stored as N x N grids ("i trusts j", "j helps k" -> "i trusts someone
who helps k"). It reaches 100% and generalises from 4 to 12 entities, but that is by construction:
the architecture already does relational composition, and the network only learns a 4-cell AND gate.
The circular picture is a fixed rendering, not something the model reads.

**Why replaced:** nothing was invented. The interesting version needs two models that must
communicate through an image.

## v1: an emergent ring script

[v1/](../v1/README.md). A speaker paints one ring per sentence with a differentiable brush; a listener
sees only the image, rotated at a random angle, blurred, noisy and partly covered, and recovers the
sentence. The meaning space is synthetic: 6 subjects x 6 verbs x 6 objects x 3 tenses x 2
polarities, 1,296 sentences, with 36 subject-verb-object triples held out. Held-out triples were
read 99% exactly, with topographic similarity 0.40 (shuffled 0). A "holistic" speaker without
per-word embeddings also worked.

**Why replaced:** the brush is made up. The goal was the film's ink.

## v2: the film's own ink

[v2/](../v2/README.md). Same game, but the brush is made of real ink: the 38 logograms from Wolfram
Research are unwrapped and cut into 83 marks and 341 ring segments. Each of 12 slots picks a ring
segment and a mark. Discrete choices were much harder to learn: the first runs reached 0-1% until a
soft start (mixtures for the first 40% of training) was added. The final run read 94% of held-out
triples, with topographic similarity 0.42.

A review of the measurements later found problems (metrics computed on sampled instead of deployed
messages, a "clean" channel that still had noise, configurations not saved with checkpoints), listed
in [research/SYNTHESIS.md](../research/SYNTHESIS.md).

**Why replaced:** a few hundred synthetic sentences is not a language. The next step was real text.

## v3: real (but simple) English with COGS

[v3/](../v3/README.md). Text goes through a frozen MiniLM encoder; a writer fills 24 slots on a fixed
ring (no rotation, as in the film); a reader with "polar eyes" (unwrap the ring into a strip, then
wrap-around convolutions) decodes text. Data: COGS, simple sentences over 645 lemmas. After many runs
(freezing the writer when the soft phase ends was the key fix, then reader-side augmentation), it
reached 77% exact on the COGS test, 92% with read-and-verify (the reader proposes 32 candidates, the
writer redraws them and the one matching the pixels wins). A fresh reader trained from scratch on the
frozen code did as well as the original one, so the code was learnable.

**Why replaced:** COGS vocabulary only. Arbitrary English did not work, and COGS's systematic
generalisation categories stayed near 0%.

## v4: open English with T5

[v4/](../v4/README.md). The encoder and decoder become T5-small, the ring gets 36 slots with marks
overlapping their neighbours, and training is split into stages (eyes first, then the code "on paper"
through the eyes' measured confusion, then the eyes learn the frozen code). Several ways of turning a
slot into a mark failed before per-slot VQ codebooks worked:

| Attempt | Outcome |
| --- | --- |
| Categorical choice with a straight-through estimator | information hid in soft probabilities |
| FSQ (4 numbers rounded to -1/0/1) | too narrow to learn through |
| Continuous 256-d slots, no quantization | 99% exact, but not marks |
| Per-slot VQ + entropy term | about 60% exact on paper |
| Stage 2, read from the image | 64% exact natural sentences |

The first full v4 (trained on Tatoeba) read 62.7% of Tatoeba test sentences exactly, but only about
21% of web sentences. Three changes followed, each argued out in the
[debate4](../research/debate4/) notes:

1. **Sender-side rereading.** The writer tries mark swaps and keeps the ring a frozen reader
   understands best. Web dev went from 21% to 67% exact. Trying to distill this search back into the
   writer failed in four variants, so it stays a search.
2. **New target and data.** The goal moved from "any word list" to meaningful but varied English
   (names, numbers, punctuation, web and news). Stage 1 continued on a 500k mix of FineWeb, Wikipedia,
   CC-News and Tatoeba, and the eyes were adapted: web dev 42% plain, 80% reread. Failing on nonsense
   lists was accepted.
3. **A locked exam**, built and fingerprinted before the final run, scored once: 81.0% exact with
   rereading on 2,000 unseen sentences ([results.md](results.md)).

After the exam, a set of probes looked inside the code: zones on the ring, transplantable grammar
marks, analogies, and the relation to T5's own geometry. The browser version in [web/](../web/README.md)
came last.
