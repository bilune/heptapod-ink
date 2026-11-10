# Round 2: lexical coverage before protocol surgery

**Recommendation:** preserve freeze25, test constrained lexical augmentation for its reader, and measure writer-assisted decoding separately. The 90% target remains a stretch. All forecasts below are judgment ranges, not measured improvements; gains are not additive.

## 1. Revised diagnosis and abandoned spending

The [final diagnosis](diagnosis_freeze25.md) establishes 99.6% train, 70.4% dev and 71.7% test exact. This confirms my generalization concern and Claude's lexical-coverage hypothesis. The frequency gradient—93.2%, 27.9%, 7.6%, 4.2% word error across increasing frequency bins—is stronger evidence than either earlier plateau argument. Frequency also correlates with restricted contexts and lexical class: it does not isolate a reader-only cause.

Zero observed code collisions kills **exact discrete collisions as the explanation for these measured failures**. It does not prove global injectivity, raster injectivity, robust separation, or compositional coding. A writer can assign unique but hard-to-learn codes. Claude's inference that lexical errors require writer repair is therefore unsupported; my expensive code/visual diagnostic ladder is now disproportionate.

Near-perfect training reconstruction weakens global capacity shortage, unavoidable rendering loss, and simple undertraining. It does not eliminate visual generalization failures. The 543 single-wrong-word failures support local ambiguity and candidate repair, but do not establish gold-in-beam coverage. Freeze25 beating freeze40 does not isolate mark width: the writers differ.

**Wasted money now:** Claude's longer soft phase, collision-repair polish and ink floor; my default symbolic/perception suite, larger reader, inventory redesign, estimator changes, and blind original-data continuation. An ink floor cannot affect a frozen writer. A ramp of stochastic hard Gumbel samples does not specifically optimize deterministic argmax choices. Neither plan should buy those interventions without new discriminating evidence. Tokenization and IID truncation remain negligible explanations. No IID result here establishes COGS systematic generalization.

## 2. Attack and concrete augmentation variant

Reader-only augmentation is coherent: for each synthetic sentence, run the **unchanged deterministic writer** and train on that sentence's actual raster. Never attach a substituted target to the original image. This changes the reader's experience, not the writer's mapping. It can expose already-encoded lexical distinctions across contexts; it cannot repair distinctions the writer discards. Writer retraining is needed only if those distinctions remain unusable after reader learning, not automatically because the lemmas were rare.

The main risks are substantial:

- **Leakage:** `cogs_graphs.build()` constructs `lemmas.json` from all splits. Do not use it. Derive dictionaries, frequencies, templates, inflections and frames exclusively from deduplicated train. No held-out vocabulary, confusion-driven substitution lists, or test-derived sampling weights.
- **Accidental training exposure:** train-derived substitution can regenerate held-out sentences or canonical meanings. Freeze the generator first; a separate exclusion-only audit must remove normalized-text and canonical-graph matches against dev/test/gen, exposing only exclusion counts to training. Never use held-out examples as templates. Disclose this held-out-aware exclusion; without it, “never seen in training” is unverified. Audit the final corpus again.
- **Distribution shift:** type/frame compatibility is insufficient for COGS IID. Uniform lexical balancing changes frequencies; noun substitutions can violate countability/selectional preferences; identical role sets can conceal different complement syntax or alternations. Grammatical synthetic examples still need not follow the test generator's distribution.
- **Goal qualification:** graph-guided augmentation introduces linguistic supervision into data construction. It preserves a learned ink code but relaxes the original “nothing about grammar or meaning hand-coded” training claim. Report that explicitly; original-data controls retain the stricter claim.

**Variant:** generate at most 150,000 unique variants, one entity/event substitution per train sentence. Sample replacement lemmas proportional to their deficit below 200 distinct training contexts, measured after exclusions; report uncovered lemmas rather than promising universal coverage. Preserve graph topology, identity/coreference, definiteness, voice, surface order and punctuation. Align substitutions to nodes, not global string replacement; skip uncertain alignments. Proper/common entities stay separate; common nouns require compatible train-attested determiner/countability signatures. Verbs require the same role frame **and** attested surface realization, complement class and inflection. Unsupported rare verbs remain uncovered. Reject truncation and malformed realizations. Do not synthesize article/preposition changes initially.

Use batches with 50% original examples and 50% variants. Cache hard codes/targets and render faithfully; inspect train-derived minimal pairs for code/raster collisions and distance distributions. Never suppress difficult pairs merely to improve reported training accuracy. Expect roughly **+5–12 points** from successful augmentation, with zero or negative gain plausible if the code lacks transferable lexical structure. Hundreds of repetitions alone are not hundreds of useful contexts.

## 3. Adversarial review of analysis-by-synthesis

A correct candidate passed through the deterministic writer must reproduce the received clean raster, subject to numerical reproducibility. Distinct codes make this attractive, but do not prove distinct rasters. Beam search can omit rare words because the reader assigns them very low probability; one-word errors do not guarantee easy search. Pixel distance is not a semantic metric.

Start with beams 4/8, length exponent 0 or .6. Include greedy in every candidate set. Prefer a unique raster match using a tolerance calibrated on repeated train renders across batch sizes; otherwise retain the chosen likelihood baseline. Avoid an unvalidated nearest-raster rule. Measure candidate oracle coverage, selected exact, false matches and latency on dev. Only if beam coverage beats greedy by five points, try width 32. A bounded extension replaces one word in greedy outputs using train-derived compatible alternatives, capped at 256 candidates per input; compare likelihood-ranked and frequency-balanced proposals on dev, without gold error positions.

The image and proposed strings are the only inference inputs: no sender IDs, original sentence, gold length, or held-out candidate database. This is legitimate **writer-assisted decoding** of one Heptapod image, but not proof that an independent reader acquired the language. Report standalone fresh-reader scores separately. Exported/noisy images require separate validation.

Forecast: **+2–8 points** initially, conditional on coverage; zero remains possible. Repairing every reported single-word failure would raise 71.7% to about **90.7%**. That optimistic accounting requires fixing almost all such failures to reach 90%, and is not an achievable-search estimate.

## 4. One lean sequence, at most four paid runs

Before payment, implement explicit checkpoint continuation, deterministic caching, beam decoding and augmentation audits locally. Preserve freeze25. Fix fresh-reader evaluation: explicitly set the external reader to eval; preserve/restore every module's prior mode; keep writer/encoder eval and gradients disabled every training iteration; train only the reader. Fix `evaluate`/`information_check`'s unconditional `model.train()`. Select on full dev, never periodic test. Save fresh-reader weights, optimizer/scheduler/RNG/step, configuration and corpus hashes. Disable automatic `report()` during development. No code changes are executed by this document.

Use batch 64, AdamW, weight decay .01, clipping 1.0; checkpoints/full-dev checks every 4k updates. All continuations use CE, zero word dropout/BOW, 50% clean images and 50% existing channel augmentation. These shared changes need a matched control.

| Ordered run | Exact intervention and decision | Planning exact-match range |
|---|---|---|
| 1: 15 min | Reproduce full-dev baseline; cache and screen decoding above. Keep only dev improvements. | Greedy 70–72%; assisted 73–80% |
| 2: 50 min | Two branches from freeze25, each up to 12k updates: original-only control versus 50:50 augmentation. Same original-example stream; LR 1e-4, 300-step warmup, cosine to 1e-5. Includes synthetic cache creation. | Augmented greedy 76–84%; assisted 79–88% |
| 3: optional 45 min | Only if augmentation beats control by ≥2 dev points and its last 4k gains ≥0.5: continue best augmented reader up to 20k, LR 5e-5→5e-6. Stop after two checks each gaining <0.5. Otherwise skip; no writer rescue. | Greedy 80–87%; assisted 83–91% |
| 4: 55 min | Fresh random reader, seed 101, same selected corpus/channel, up to 24k updates; LR 3e-4, 500-step warmup, cosine to 3e-5. Reserve final 10 minutes for locked full-test evaluation of selected original/fresh readers, greedy and assisted. | Fresh greedy 75–85%; selected assisted 80–91%, conditional on earlier success |

Ranges after failed gates revert to the measured incumbent. Keep every starting checkpoint eligible. Acquisition uses only paired text/images, no copied reader weights or symbolic teacher. Lock decoding on dev; report the already-exposed test as such, and measure augmented-corpus overlap counts alongside results.

At the **supplied** $3.95/H100-hour rate, 165 minutes cost $10.86; ten additional minutes for overhead give **$11.52 maximum planned GPU spend**. Enforce a total dollar stop below $12 including other billed charges; wall caps override update targets. Reserve run 4 before optional spending. No extra paid retry.

I converge with Claude on lexical augmentation, but disagree that uniqueness proves a good code, that lexical failures demand writer retraining, or that augmentation guarantees 90%. I retain rerendering as a separately labeled inference lever, not an acquisition substitute.
