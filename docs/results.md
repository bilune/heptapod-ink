# Results

All numbers are for the current checkpoint, `v4/runs/s2/model.pt`, unless a section says
otherwise. Raw files: [v4/runs/exam/](../v4/runs/exam/) (locked exam), [v4/runs/s2/](../v4/runs/s2/)
(probes), and the lab notes in [research/debate4/](../research/debate4/), starting with
[final_results.md](../research/debate4/final_results.md).

## Exam protocol

- **Built once, evaluated once.** [v4/locked_exam.py](../v4/locked_exam.py) built three sets and
  stored their SHA-256 fingerprints in [v4/LOCKED_EXAM.json](../v4/LOCKED_EXAM.json). They were
  not looked at during development and were scored a single time, after every design decision
  was made. Marker files in `v4/runs/exam/` block a second run.
- **Natural set:** 2,000 sentences of 3-12 words from the Universal Dependencies English treebanks
  GUM, PUD, LinES and ParTUT (dev and test portions), deduplicated against Tatoeba, EWT and each
  other, and kept only if T5's tokenizer round-trips them cleanly.
- **Random sets:** 2,000 random word combinations of 3-10 words, drawn like the training sampler
  (new seed), and 1,000 uniform 10-word lists.
- **Through the image.** Every ring is saved as a 16-bit PNG, loaded back and read by the eyes.
- **Conditions.** *Plain:* the writer's first ring. *Reread:* after sender-side search
  ([how-it-works.md](how-it-works.md#rereading-sender-side-search)). *Perturbed:* reread rings
  read under perturbations the search did not see. *Shuffled:* each reader gets a different
  sentence's ring, which measures how much the reader can guess without the picture.
- **Metrics.** Exact match (case-insensitive unless marked "cased"), word accuracy, word error
  rate. 95% Wilson intervals.

## Locked exam

| Set | Plain exact | Reread exact | Reread, perturbed | Shuffled image | Reread words / WER |
| --- | --- | --- | --- | --- | --- |
| Natural, 2,000 | 33.5% (31.4-35.5) | **81.0% (79.2-82.7)**, cased 80.5% | 75.5% (73.6-77.4) | 1.2% | 92.9% / 5.9% |
| Random 3-10 words, 2,000 | 4.1% | 36.8% (34.7-38.9) | 29.8% | 0.6% | 61.3% / 34.5% |
| Uniform 10 words, 1,000 | 0.0% | 0.0% (0-0.4) | 0.0% | 0.0% | 17.3% words |

Natural set, reread exact:

- By source: LinES 86.3%, ParTUT 80.5%, GUM 78.6%, PUD 70.4%.
- By length: 3-5 words 94.4% (498 sentences), 6-8 words 87.8% (654), 9-12 words 67.9% (848).
- Rereading turned 951 wrong sentences into exact ones and broke none (the final
  "not worse" check makes zero losses partly by design).
- The 390 remaining cased errors: 155 involve names or capitalised words, 129 other words,
  59 only punctuation or quotes, 37 numbers, 10 only capitalisation.

Some typical failures:

```
Coron is in the province of Palawan, Philippines on Busuanga Island.
-> Coron is in the province of Palaghan, districts on Babuanga Island.
Arash Javanbakht is Assistant Professor of Psychiatry, Wayne State University.
-> Arash Javan Bakhn is Assistant Professor of Psychiatry, Wayne State University.
```

### For comparison: a classical codec

Before the exam, a hand-built reference was measured on dev panels: compress the sentence with a
language model (ideal arithmetic coding with GPT-2 or Qwen2.5-1.5B), protect it with a
Reed-Solomon code over the 81 marks, and send it through the same eyes. That would decode about
99% of Tatoeba sentences and about 98% of web sentences, and even 99% of uniform 10-word lists
(they need about 150 bits, the channel has room for 177 after error correction). The learned
code is well below what the channel can carry. Details in
[step1_results.md](../research/debate4/step1_results.md). The point of the project is what the
networks invent, not beating this baseline.

## The subword cliff

Recounting the exam predictions by T5 tokens (including the end token): 95.8% exact for up to 10
tokens (716 sentences), 57.2% for 16-20 (376), 8.9% for 21 or more (90). Among sentences of
exactly 12 words, those with at most 1.5 pieces per word read 64.3% exact (101/157), the rest
9.1% (5/55). Rare words, names and source are competing explanations. This is an exploratory
analysis of an already-scored exam
([round3_codex.md](../research/debate4/round3_codex.md)).

## Errors are close in meaning

Looking at all 1,279 isolated wrong-word substitutions of the plain writer on the exam, the wrong
word is related to the right one (MiniLM cosine) more than chance and more than T5's own guess
from context:

| Replacement | Mean relatedness to the correct word |
| --- | --- |
| Actual reader error | **0.484** |
| Random vocabulary word | 0.218 |
| Frequency-matched random word | 0.268 |
| T5 filling the blank from context only | 0.364 |

Examples: boat -> ship, glass -> mirror, understood -> knew. But relatedness is not synonymy:
yes -> no and purple -> green also score high, and they change the facts
([round4_codex.md](../research/debate4/round4_codex.md)).

A controlled follow-up ([v4/semantic_probe.py](../v4/semantic_probe.py), 92 sentence frames)
swapped the ring of a word for the ring of a synonym, a contrasting word or an unrelated word. A
synonym's ring helps the reader produce the target more than a contrasting word's ring (+4.7
nats, 95% CI 3.3-6.1), the same through the images. Rings of synonyms are also closer (5.5 of 36
slots differ) than contrasts (6.4) or unrelated words (8.8).

Where does this come from? Mostly from T5. The base T5 encoder shows the same ordering, and the
ring's distances are correlated with base T5's (r = 0.59) for these word swaps. At the scale of
whole sentences the agreement almost vanishes (Spearman 0.085 over 384 sentences), less than a
random projection code of the same size would keep. Sentences with the same words in a different
role ("The man bit the dog") get closer rings than paraphrases, and base T5 has that same bias.
Summary: the ring keeps T5's local word relations and reshapes its global geometry
([round5_codex.md](../research/debate4/round5_codex.md),
[figure](../research/debate4/round5_figure.png)).

## What the slots do

**Anatomy** ([v4/anatomy.py](../v4/anatomy.py), 8,000 sentences): for each property of a sentence,
how much of it can be read from a single slot.

| Zone | Slots | Carries |
| --- | --- | --- |
| Who | 7, 9, 29, 35 | subject, first word, frequent names; written four times |
| What it does | 2, 8, 13, 17, 25, 26 | verb, tense, negation, agreement |
| On what | 14, 15, 10, 30 (+0, 4) | object noun |
| How it ends | 22, 32, 33 | final punctuation, question or statement |
| When / where | 15, 19, 28 | numbers, time and place words |

A single slot (33) alone resolves about 89% of the question-versus-statement information. The figure with the zones on
an actual ring is [anatomy_zones.png](../v4/runs/s2/anatomy_zones.png), the full heatmap
[anatomy.png](../v4/runs/s2/anatomy.png).

**Causal check** ([v4/zones_causal.py](../v4/zones_causal.py)): copy one zone's marks from a donor
sentence into another sentence's ring.

| Transplant | Target effect | Same number of random slots |
| --- | --- | --- |
| Who (different subject) | 99.5% subject changes | 0% |
| How it ends (question donor) | 87.5% becomes a question | 0% |
| What it does (negative donor) | 50% negation appears | 2% |

Inside the "who" slots, single marks act like a small dictionary: in slot 7 one mark means "Tom"
in 361 of 372 uses. The zones are the same in the earlier v4 checkpoint trained only on Tatoeba.

**Nouns** ([v4/content_probe.py](../v4/content_probe.py)): an object noun is written as a
combination of 2-4 marks; with the best 2 noun slots it can be identified 93% of the time, with 4
slots 98%. Transplanting the 4 noun slots swaps the noun and keeps the rest of the sentence 44% of
the time (random slots 1%).

**Structure probes** ([v4/probe_structure.py](../v4/probe_structure.py)): changing one word changes
4.8 of 36 slots on average, and the same word swap tends to touch the same (slot, mark) pairs across
sentences (overlap 0.13, Tom -> Mary 0.30, versus 0.004 between different swaps). Replacing one slot
at random leaves the reading unchanged 83% of the time, changes one word 11%, and wrecks half the
sentence 2%. It is a distributed code, but not a holistic one where any change repaints everything.

## Grammar marks

[v4/grammar_probe.py](../v4/grammar_probe.py): for each operation, learn a consensus edit on 30
template pairs and apply it to 30 other sentences' rings.

| Operation | Slots edited | Exact result (symbols / image) | Marker present | Random edit, same size |
| --- | --- | --- | --- | --- |
| Plural subject ("The teacher is" -> "The teachers are") | 1 (slot 26) | 96.7% / 100% | 100% | 0% |
| Past tense | 2 (0, 26) | 60% / 60% | 77% / 73% | 3% |
| I -> she, with agreement | 8 | 50% / 47% | 100% | 0% |
| Yes/no question | 10 | 30% / 30% | 100% | 0% |
| Negation | 15 | 0% / 0% | 100% | 3% |

Applying another operation's edit never produces the intended change. Caveat: narrow templates,
learned and tested on the same sentence shapes. For questions and negation the grammatical marker
is right but the verb, object or subject often changes too.

## Analogies

[v4/analogy_probe.py](../v4/analogy_probe.py) does word arithmetic inside the carrier sentence
"The answer was {w}." on the writer's continuous slot vectors (before choosing marks), then reads
the resulting ring. [v4/visual_analogy.py](../v4/visual_analogy.py) tries the same on the images.

| Relation (90 each) | Writer vectors | Offset from another relation | Pixel arithmetic | "Changed marks" rule | Random edit |
| --- | --- | --- | --- | --- | --- |
| Singular -> plural | **62%** | 2% | 11% | 0% | 0% |
| Country -> nationality | **54%** | 23% | 42% (40% with no change at all) | 30% | 30% |
| Adjective -> comparative | **39%** | 1% | 11% | 0% | 0% |
| Present -> past | **26%** | 2% | 7% | 0% | 0% |
| Male -> female | **20%** | 1% | 7% | 0% | 0% |

So king - man + woman style arithmetic works in the writer's space, but not on the drawings: adding
and subtracting images, or reusing the marks that changed for other words, does no better than
random edits. ([analogy_queen.png](../v4/runs/s2/analogy_queen.png) shows one example.)

## Can a stranger learn it?

[v4/fresh_reader.py](../v4/fresh_reader.py): freeze the sender (writer, codebooks, eyes) and train a
brand-new reader (new mark embeddings, new adapter, base T5-small decoder, nothing inherited) on N
pairs of (plain ring, sentence). Read through the frozen eyes, greedy decoding, dev panels.

| Training pairs | Web sentences, exact (words) | Tatoeba, exact (words) |
| --- | --- | --- |
| 1k | 0% (10%) | n/a |
| 5k | 0% (22%) | n/a |
| 20k | 7.8% (45%) | 24.6% (68%) |
| 200k (12k steps only) | 25.4% (65%) | 58.6% (89%) |
| Original reader | 41.8% (76%) | 70.7% (93%) |

A newcomer learns a good part of it from examples, so it is not a purely private code. Reread rings
do not transfer: the search tuned them for the original reader, and the fresh reader gets 23.4% on
them. The fresh reader's architecture is also simpler than the original one, so this is a lower
bound, not a fair contest.

## Earlier numbers, for context

| Checkpoint | Panel | Plain exact | Reread exact |
| --- | --- | --- | --- |
| v4 trained on Tatoeba only | Tatoeba test (3,000, run once) | 62.7% | (read-and-verify 88.2%) |
| v4 trained on Tatoeba only | EWT web dev (256) | 20.7% | 66.8% |
| s2 (mixed data + eyes adapted) | EWT web dev (256) | 41.8% | 80.5% |
| s2 | locked exam (2,000) | 33.5% | 81.0% |

"Read-and-verify" was an older receiver-side trick: the reader proposes 32 candidates, the writer
redraws each, and the one that matches the pixels wins. Rereading on the sender side replaced it.
See [v4/README.md](../v4/README.md).

## Limits

- Nonsense fails. Random 10-word lists get 0% exact. The system leans on the decoder's knowledge of
  English and only sends what it cannot guess; that was an accepted design choice.
- Rare names, technical words and numbers cause most remaining errors.
- Non-ASCII is lost before the model sees it: T5's tokenizer has no "í", so "Bahía" comes back as
  "Baha".
- Up to 12 words. Accuracy drops with subword count, as above.
- The reader was trained on clean renders with small perturbations. Photographed, stylised or
  hand-drawn rings are untested.
- The probes use narrow templates and dev material; treat them as exploratory.
