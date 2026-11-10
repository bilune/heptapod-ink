# Research synthesis: what to do next

Distilled from four deep reads (2026-11-08). Full notes, with citations, are in `notes/`:
`codex_methodology.md` (EGG plus 5 emergent-communication papers), `arrival_sources.md`, `drawing_games.md` and `meanings_and_survey.md`.

## Purpose (fixed)

Two models invent a language as close as possible to Heptapod B: circular, written at once, in the film's ink, able to express progressively more. We want to know whether it is a **real language** (systematic, and learnable by models that did not invent it) and whether it is **Heptapod-like**. Human readability is out of scope.

## What the sources settle

1. **Novelty.** Of 181 emergent-language papers surveyed (Peters et al. 2025), 5 use images as the signal and none is circular or order-free (`meanings_and_survey.md` §5).
2. **Communication is not language.** Success with the original partner says little. The tests that matter are held-out generalization and acquisition by a fresh listener on a frozen speaker (Chaabouni 2020; Rita 2022: the original listener can hide what the messages make learnable). Compositionality metrics are neither necessary nor sufficient for generalization (Chaabouni 2020).
3. **The film's script is measurably compositional.** Compound logograms reuse the base word's ink at the same angle: OfferWeapon holds 100% of Weapon's ink, ThereIsNoLinearTime 98% of Time's, against a 0.42 median for arbitrary pairs. Ring size is constant, and complexity shows up as more ink (`arrival_sources.md`). This gives a concrete, canon-grounded "Heptapod-like" test.
4. **In the film, orientation is fixed.** Words sit at the same absolute angle. Our random-rotation channel goes beyond the canon, so both conditions should be reported.
5. **Prior Arrival code is not reusable** as a base. Wolfram's notebooks are worth borrowing as diagnostics (12-wedge de-rotated grid, nearest-neighbour retrieval, aligned difference images); our RANSAC unwrap is more accurate than his centred unwrap.
6. **EGG is archived.** Borrow ideas and metric definitions, but don't import the framework (`codex_methodology.md` §4).

## Problems in our current evidence (must fix first)

From `codex_methodology.md` §0.2, verified against the code:

- The topographic similarity is computed after `evaluate()` puts the agents back in training mode, so v2 measures **sampled** Gumbel messages, not the deployed code. It also uses raw pixel distance and tie-broken ranks instead of a tie-aware Spearman over symbol distances.
- `rotate=0` still applies blur, noise and occlusion, so it is not a clean channel. Python's RNG (which drives occlusion) is unseeded in v2.
- Checkpoints don't store their configuration. `out_run3` (40 ring segments, 94% held-out) and `out_run4` (32 segments, 3 marks, 57% held-out) are different configurations and must not be merged.
- A zero-shot "entity never seen as subject" split is invalid with role-specific embeddings: it tests an untrained embedding, not the language. Use a split where every value appears in every role.

## Next experiment (one, focused)

Question: **is the v2 film-ink script a real, Heptapod-like language?**

1. **Fix the measurement:** eval mode for every metric; tie-aware Spearman over cyclic symbol distances; a truly clean channel condition; seeded Python RNG; config saved in checkpoints.
2. **Balanced split** (Codex §5.2): test = triples with object `(s+1) mod 6`, dev = `(s+2) mod 6`, train = the rest (864/216/216). Every entity appears as subject and object in training. Select the checkpoint on dev, never on test.
3. **Two orientation conditions:** fixed (canon) and random rotation.
4. **Controls** (listener-only, cheap): an engineered compositional code and a random injective code rendered with the same ink. They show whether the task is learnable at all and what "unstructured" looks like.
5. **Fresh listeners:** freeze each trained speaker and train 3 new listeners on train meanings only. Compare their held-out accuracy and learning curves with the controls.
6. **Heptapod test (from the film):** for minimal pairs (one word changed), measure the share of ink kept in place, and compare with the film's numbers (≈0.95–1.0 for base inside compound, 0.42 for arbitrary pairs). Also apply training-derived single-mark substitutions to held-out sentences and check that only the intended word changes.
7. **Seeds:** 3 per condition first. Codex recommends 10 for proper statistics; expand only if the 3-seed picture is promising.

Deferred until this passes: ring vs line (Codex §3.3 has the capacity-matched recipe, which needs a canonical cyclic codebook), larger meaning spaces (an in-house COGS-style graph generator first, then deduplicated COGS; AMR later for breadth), populations and iterated learning.
