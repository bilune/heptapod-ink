Act as a world-class research scientist who has published at NeurIPS/ICLR on emergent communication, learned discrete representations and neural compression, and who reviews for those venues. You are advising a small personal project with a modest budget. Be demanding: the goal is a result and a write-up that would genuinely surprise an expert reviewer, not an incremental tweak.

Read research/debate4/context.md and everything it lists. Use live web search: read recent papers and strong baselines (2023-2026) relevant to: high-rate discrete codes and VQ training (rotation trick, SimVQ, FSQ/LFQ/BSQ, codebook utilization), neural compression of text with LM priors (bits/char of short sentences, arithmetic coding with LLMs), learned channel codes / deep JSCC, amortized vs iterative inference, emergent communication with images/sketches (e.g. learning to draw, Lewis games with visual channels), OCR-free document models, and anything else an expert would bring. Cite with links.

Write research/debate4/expert_codex.md (<=1,800 words):
1. Expert verdict: what is genuinely interesting here, what a reviewer would dismiss, and what result would actually surprise them (state concrete target numbers and why they would be surprising, e.g. bits per ring vs. theory, random-combo exact, robustness, learnability by a fresh reader, compositionality probes).
2. The ceiling: an honest capacity analysis (bits per ring needed vs achievable with this alphabet/geometry and 99% mark reading), and what limits us now.
3. A research plan to get there: 3-5 experiments in priority order, each with mechanism, why it should work (evidence), H100 minutes, local smoke test, pass/fail rule. Include at least one bold idea you would bet on.
4. Evaluation protocol an expert would accept: held-out sets, baselines (e.g. LM-prior entropy coding of the same text into the same symbol budget as an upper reference; a non-learned code), ablations, controls, confidence intervals, what to report.
5. Budget proposal in tiers (USD 10 / 25 / 50) with stop rules, and what you would NOT do.
6. What would make you change your mind.
Be concrete and opinionated; no filler.
