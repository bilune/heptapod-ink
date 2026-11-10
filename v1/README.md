# Heptapod Ink v1: an emergent circular script

Two neural networks invent a circular ink script to talk to each other, under constraints taken from *Arrival*'s Heptapod B.

- **Speaker:** receives a whole sentence (`subject verb object`, tense, polarity) and paints **one** ring-shaped logogram through a differentiable brush renderer.
- **Listener:** sees only the image, **rotated at a random angle**, blurred, noisy and sometimes partly covered, and must recover the full sentence.

Nobody designs the alphabet. Which marks mean what, and where they go on the ring, emerges from the pressure to communicate.

## Heptapod constraints

| Arrival | Here |
| --- | --- |
| One logogram expresses a complete thought | One image encodes all 5 slots of the sentence at once. There are no tokens and no reading order |
| Circular, with no beginning or end | The listener sees the ring at a random rotation, so meaning has to live in relative structure |
| Ink: blots, tendrils, varying weight | Brush renderer with 32 overlapping sectors: ink, width, radial wobble, splash and splash length |
| Deliberate marks | Ink has a cost, plus a penalty for "half-on" marks. Heavy pixel noise makes grey levels useless |

This is an original design brief inspired by the film. It is **not** canonical Heptapod B, and it does not reuse the film's logograms.

## Vocabulary

- Entities: human, heptapod, ship, weapon, time, language
- Verbs: sees, gives, fears, offers, understands, becomes
- Tense: past, present, future
- Polarity: affirmative, negative

That gives 1,296 sentences. **36 of the 216 subject-verb-object triples are held out** in every tense and polarity: 216 sentences the speaker never writes during training, although every individual word is seen.

## Run

```bash
python v1/heptapod.py train --steps 3000                  # ~20 min on an M4 Pro (MPS)
python v1/heptapod.py say heptapod offers weapon --tense future
python v1/heptapod.py say human understands time --negative
```

Outputs in `v1/out/` (`out_holistic/` is the holistic-speaker run):

- `metrics.json`: accuracy per slot and per whole sentence, on training meanings and on held-out triples, plus topographic similarity
- `minimal_pairs.png`: one sentence with one word changed at a time, and the diff of the ink
- `lexicon_grid.png`: subject × verb grid
- `held_out.png`: logograms for sentences never seen in training
- `say_*.png`: any sentence you ask for, together with what the listener read

## What to look at

1. **Generalization:** accuracy on held-out triples. If it is close to training accuracy, the script composes rather than memorizes.
2. **Topographic similarity:** the Spearman correlation between how different two sentences are and how different their logograms look. A shuffled baseline sits near 0. Higher means compositional.
3. **Minimal pairs:** whether each word owns a region or a kind of mark, or whether changing one word repaints the whole ring (holistic, more like the film).

## Honest limits

- The speaker embeds each slot separately before mixing them. That is an inductive bias toward compositionality, so held-out success is partly built in. Removing it (one embedding per whole sentence) is the harder test.
- The meaning space is small and synthetic. It is not natural language.
- The channel is continuous and differentiable. Real emergent-language work often uses discrete channels; here the noise does that job.
- The ring is the only allowed shape: the renderer imposes the circle. The marks on it are free.

## Results (`out/metrics.json`)

- Held-out triples, full rotation: 99.0% whole sentences exact (training meanings 99.6%).
- Topographic similarity 0.40 (shuffled baseline about 0).
- The holistic speaker (one embedding per whole sentence) also reaches 99.9% on its meanings, with topographic similarity 0.38.

## Next ideas (v2 took the "real ink" route instead)

- **Holistic speaker:** remove per-slot embeddings and see whether the script stays compositional.
- **Recursive sentences:** "heptapod offers [human understands time]", nested inside the ring.
- **Population:** several speakers and listeners, so the script cannot be a private code between two networks.
- **Writing in time:** reveal the ring stroke by stroke and measure how early the listener knows the whole sentence. That is the film's idea that the writer knows the end before starting.
