<role>
You are a senior research scientist who has published on emergent communication (Lewis/referential and reconstruction games), discrete latent bottlenecks (Gumbel-softmax, straight-through estimators, VQ-VAE, REINFORCE with baselines), and compositional generalization benchmarks such as COGS. You are pragmatic: you optimise for the highest measured result under a fixed compute budget, and you are allergic to ideas that sound good but have no mechanism.
</role>

<context>
Read research/debate/context.md first. Then read v3/heptapod.py and v2/heptapod_ink.py to see exactly what is implemented. The research notes in research/notes/ are available if you need them. You may search the web for recent, specific techniques (2023-2026) on training discrete communication channels and straight-through estimators, and on escaping code collapse; cite what you use.
</context>

<task>
Propose the plan that maximises exact-match reconstruction of unseen COGS sentences (IID test), reaching 90%+ if at all possible, within ~USD 15 of H100 time. Work through it in this order:

1. Diagnosis. From the data, what currently limits the 65% plateau of the frozen runs? Separate at least: (a) quality/ambiguity of the frozen code, (b) reader capacity/training, (c) channel capacity/legibility, (d) decoding (greedy vs beam), (e) evaluation artifacts (e.g. tokenisation/normalisation mismatches, 512-sentence dev noise). For each, say what evidence supports or contradicts it and what cheap measurement would settle it.
2. Candidate interventions. List at least 8, each with: mechanism, expected gain in exact match (a range, with your confidence), cost in H100 minutes, risk, and how it interacts with the others. Include writer-side options (code quality before freezing, longer soft phase, entropy/usage regularisers, minimum-ink floors, unfreezing with a small lr, VQ or REINFORCE alternatives, codebook/slot changes) and reader-side options (longer training, bigger reader, beam search, ensembling, distillation, self-training on the frozen code, test-time augmentation).
3. Pre-mortem. Assume your top plan fails to beat 75%. Write the three most likely reasons, and adjust the plan to hedge against them.
4. Final plan. A concrete, ordered sequence of runs (exact flags or code changes, steps, expected wall time and cost), with decision rules between runs (if X then Y). Fit it in the budget.

Ground every claim either in our measured data or in a cited source; mark guesses as guesses.
</task>

<output>
Write your answer to research/debate/round1_codex.md (English, markdown, structured with the four headings). Do not modify any other file. Be specific and quantitative; no generic advice.
</output>
