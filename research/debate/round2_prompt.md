<role>
Same role as round 1: senior researcher in emergent communication, discrete bottlenecks and compositional generalization, optimising measured results under a hard budget. This round you are also the adversarial reviewer of two plans, including your own.
</role>

<new_evidence>
Read research/debate/diagnosis_freeze25.md. Key facts measured on the final freeze25 checkpoint (full splits):
- train exact 99.6%, dev 70.4%, test 71.7%; freeze40 test 69.2%.
- 0 code collisions among test sentences, 0 test codes equal to another train sentence's code.
- Test word error by train frequency of the word: 1-9 occurrences 93%, 10-99 28%, 100-999 8%, 1000+ 4%.
- 543 of 809 failed test sentences have exactly one wrong word; errors are rare nouns/names replaced by frequent ones, a<->the, and preposition confusions.
</new_evidence>

<inputs>
- Your round-1 plan: research/debate/round1_codex.md
- Claude's round-1 plan: research/debate/round1_claude.md
- Context: research/debate/context.md, code in v3/heptapod.py and v3/cogs_graphs.py (COGS meaning graphs give each lemma's type: event vs entity, proper vs common; and verb role frames).
</inputs>

<task>
1. Update your diagnosis with the new evidence. Which round-1 hypotheses does it confirm, weaken or kill? Be explicit about anything in either plan that is now wasted money.
2. Evaluate Claude's proposed main lever: train-only lexical substitution augmentation (swap lemmas within the same type, e.g. common noun for common noun, proper name for proper name, verbs only among verbs with the same role frame, keeping definiteness), so that every lemma appears hundreds of times. Attack it: leakage risks, whether it can change the frozen writer's codes (it can't retrain a frozen writer; does it need a writer retrain or only reader retraining?), whether augmented sentences are IID with COGS test, and what gain to expect. Propose the best concrete variant.
3. Evaluate your own analysis-by-synthesis reranking (beam candidates re-rendered by the frozen writer) given 0 collisions and one-wrong-word failures. Is it legitimate as part of "the model speaks Heptapod"? Expected gain?
4. Converge: a single lean plan of at most 4 paid runs, under USD 12 total, ordered, with exact changes, decision rules and expected exact-match after each step. Include the fresh-reader bug fixes. Say where you and Claude still disagree, if anywhere.
</task>

<output>
Write research/debate/round2_codex.md, at most 1,500 words, markdown. Do not modify other files.
</output>
