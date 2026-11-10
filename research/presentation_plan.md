# How to present Heptapod Ink (synthesis of prior-work review, 2026-11-09)

Prior work: logogram-looking image generators (meghabyte/arrival-logograms, Logograms-GAN, Şeker's Logogram Language Generator) never decode anything; Google's 2017 "Heptapod B" Word Lens was an April Fools' joke; Wolfram analysed the real ink ("if this were real, what would we compute?"); academic visual emergent communication (Learning to Draw, Emergent Graphical Conventions, Interactive Sketch QA, Gonçalo et al.) picks 1 of K images or regresses a word vector, never reconstructs open sentences.

What is new here: open-vocabulary English sentences -> one ring of real Arrival ink -> exact reconstruction from pixels, measured on a locked exam (81% exact, 94% for 3-5 words), with controls.

## Hook
"Two neural nets invented a writing system from Arrival's ink. One writes any sentence as a single ring; the other reads it back from the image alone." Optional second line: "Google faked this on April Fools' 2017."

## Figures (in order)
1. Hero strip: sentence -> ring -> decoded sentence (✓). A GIF of the loop.
2. Pipeline: writer (T5) -> 36 slots x 81 real ink marks -> eyes -> reader (T5).
3. Grid of 9-16 rings sampled at random from the locked exam, including failures.
4. Operating curves: exact vs sentence length / subword count (the cliff) and vs image noise.
5. Error gallery: boat->ship, glass->mirror next to the context-only baseline (0.48 vs 0.36 relatedness).
6. "It is a code, not a style": change one word -> which marks change; slot-position heatmap.
7. Fresh-reader learning curve (a stranger can partially learn it).

## Controls (one claim each)
wrong image ~1% (meaning is in the image); noise 75% (robust); random equal-capacity code mirrors T5 better (the ring is shaped for the reader, not a map of T5); errors closer in meaning than context alone (inherits T5 semantics); fresh reader 89% words on common sentences (learnable).

## Honesty
Disclose the sender-side reread search next to every number (plain 33.5% vs reread 81%). Show failures as a mapped boundary (rare names, technical words, nonsense lists, accents). Call it an invented script / learned code, not "Heptapod B" or "a language". Not human-readable. Random, not cherry-picked, examples.

## Demo (optional)
Type a sentence -> get a ring -> watch it read back; noise slider; download PNG; "decode a shared ring".

## Licensing (required if published)
Ink derived from the Arrival logograms (Martine Bertrand; Arrival 2016, dir. Denis Villeneuve) released by Wolfram Research under CC BY-NC 4.0; state the marks were cut and modified; generated rings under CC BY-NC; no commercial use (no paid API, prints, tattoo sales); "not affiliated with Paramount or the filmmakers".
