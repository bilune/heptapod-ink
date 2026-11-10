# Debate 4: from "works" to "an expert would be surprised"

Read: v4/README.md, research/debate3/context.md, round1_codex.md, round1_claude.md, round2_codex.md, round3_codex.md, v4/heptapod_t5.py, v4/search.py, v4/distill.py.

## Where we are (2026-11-09 evening)
System: Tatoeba English <=12 words -> T5-small encoder -> 36-slot writer -> per-slot VQ (81 entries, each = one real Arrival ink mark) -> one ring logogram (fixed ring, marks 30 deg wide at 10 deg pitch, overlapping) -> polar eyes (99% marks read) -> adapter -> T5-small decoder.

Plain writer, read from the image (v4 test, once): natural 62.7% exact / 88.2% words; random word combos 9.2% / 44.1%. Read-and-verify (beam 32, writer redraw): natural 88.2% exact; random 29%.

New today: sender-side search ("writer rereads"): coordinate descent over the 16 nearest symbols per slot, minimizing teacher-forced NLL of the source under the frozen symbolic reader, ~9 accepted moves, guarded by a visual multi-view check (clean + 3 perturbations) and a no-worse greedy check. Measured through 16-bit PNG export/reload and the eyes:
- Tatoeba dev (300, random sample): 64% -> 95.3% exact, 98% words.
- Tatoeba dev_random (300): 7% -> 49% exact, 42% -> 75% words.
- Fresh exam (never used before): UD English EWT web sentences 3-12 words (576 of 1,000 evaluated): 21% -> 68% exact; with unseen perturbations 19% -> 57%. Fresh random combos (576): 12% -> 54% exact; perturbed 11% -> 45%. Shuffled-image control ~0-1%.
- Cost ~1 s/sentence on H100.
Distilling the search into the writer failed in 4 variants (CE on distances exploded; MSE pulled vectors to the wrong cells; MSE+commit collapsed; decision-region hinge passed a 4-sentence overfit gate but degraded on 8.7k targets). Your round-3 diagnosis: shared-parameter coupling.
Capacity facts: sum of per-slot marginal entropies 155 bits (natural) / 201 bits (random) vs 228 nominal; random 10-word combos need ~139 bits; 22% of natural dev sentences have another beam candidate with the identical ring; continuous (unquantized) 256-d slots reached 99% exact on both natural and random.

## Constraints
Trained models only (search over model outputs is fine); human readability out of scope; one ring per sentence is the film's convention (geometry changes allowed if argued). Gonzalo's own Modal H100 (~USD 3.95/h); a new budget must be proposed explicitly. Local M4 Pro 24GB: tiny smoke tests only (a heavy local job crashed the Mac today). Every change smoke-tested before the H100. A stall watchdog is mandatory on Modal runs.
