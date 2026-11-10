# Round 1, Claude: measurements first

I ran two free local measurements on the final v4 checkpoint (v4/runs/final/model.pt) while you researched.

## 1. Per-slot code entropy (4,000 fresh sentences each, plug-in estimate)
- Natural: sum of per-slot marginal entropies 155 bits (min 2.1, median 4.5, max 5.8 bits/slot).
- Random combos: 201 bits (min 3.7, median 5.9, max 6.3 bits/slot), vs 228 nominal and ~139 needed (your 13.85 bits/word estimate).
So the alphabet is mostly used on random combos. The upper bound on carried information is not the binding limit; the writer does not pack the words efficiently into what it uses. This points at an amortization gap (Cremer et al. 2018, "Inference Suboptimality in VAEs") more than at channel capacity.

## 2. Writer-side code search (your option 3), measured
Script: scratchpad diag/write_search.py. Frozen model. For each sentence: start from the writer's nearest-centroid code; candidates per slot = the K=8 nearest codebook entries to the writer vector; coordinate ascent: evaluate all 36x8 single-slot substitutions by teacher-forced NLL of the source text under the symbolic reader, take the best, repeat up to R rounds, stop when no improvement. Then render the searched code, read it with the EYES (visual path), greedy decode. 100 dev items each:

| set | writer as is: exact / words | writer rereads (R=20): exact / words |
| --- | --- | --- |
| dev_random | 6% / 43% | **42% / 69%** |
| dev natural | 66% / 89% | **95% / 96%** |
(R=6 on dev_random: 7%/47% -> 46%/72%.)

This is far above your +3-12 forecast and costs zero training. It is analysis-by-synthesis on the sender side (rational-speaker style: the speaker models the listener), using only trained models. It must be reported separately from "writer alone", like read-and-verify.

## My position
1. Make sender-side search the centrepiece, and improve it before spending GPU: (a) robust objective: expected NLL under the measured eyes confusion, or accept moves only if the rendered ring decodes correctly through the eyes, to avoid fragile adversarial codes; (b) wider K (16-32) and a few random restarts or a small beam over codes; (c) gradient shortlist to cut cost; (d) combine with receiver-side read-and-verify (note: verification must redraw with the same searched writer, as you said; a deterministic search makes that possible but expensive).
2. Then amortize the search into the writer (expert-iteration / self-distillation): generate searched codes for many train sentences on the H100, train the writer with CE toward the searched code indices (plus the existing text loss), iterate. That converts test-time compute into a better writer, so plain `say` improves and read-and-verify stays cheap. This is the training-time use of the budget I would prioritise over rotation trick / SimVQ, because it attacks the measured gap directly.
3. Rotation trick is cheap and orthogonal; fine as a second branch. Alphabet expansion (161 symbols) only if amortized search plateaus well below the continuous 99%.
4. Holdout hygiene: the Tatoeba test has been used once. For any new claim we need a fresh holdout: freshly generated random combos (new seed, exclusion-checked) and natural sentences from a newer Tatoeba export (sentences added after our download), deduplicated against everything we used.

Questions for you: is there a flaw in the search measurement (e.g. it exploits the reader in ways that will not survive the eyes or PNG? I did read through the eyes); what is the best search algorithm here; is amortizing it into the writer sound, and how would you set it up within USD 5-10; and does anything in the literature beat this?
