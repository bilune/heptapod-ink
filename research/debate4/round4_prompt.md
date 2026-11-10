Act as the same expert reviewer. Gonzalo noticed something: when the plain writer's ring is misread, the wrong word is often a near-synonym or semantically close word, e.g. "across the glass" -> "around the mirror", "boat" -> "ship", "understood" -> "knew", "week" -> "day", "coffee" -> "work" (less close), "wrote" -> "read". He finds this interesting: does the ring carry meaning rather than spelling?

Data you can use (read-only, CPU-only, small analyses with .venv/bin/python and HEPTAPOD_DEVICE=cpu, <=3 minutes each, no training; the Mac crashed earlier under a heavy job):
- v4/runs/exam/preds_exam_locked_natural_2000.json: texts + "plain" (writer as is) + "search" predictions + perturbed + shuffled, locked exam (UD GUM/PUD/LinES/ParTUT).
- v4/runs/s1/ and v4/runs/s2/ search_eval_*.json (examples), v4/runs/s2/my_sentences.json (20 sentences Claude wrote; plain vs reread).
- research/debate4/final_results.md, round3_codex.md; v4/heptapod_t5.py for the system.
- sentence-transformers/all-MiniLM-L6-v2 and GPT-2 are in the local HF cache; T5-small too.

Use web search for related literature (semantic errors in learned codes / emergent communication, "semantic paraphasias" in human aphasia and speech errors, lossy semantic compression, rate-distortion with semantic distortion, decoder language-model priors producing plausible substitutions).

Write research/debate4/round4_codex.md (<=1,100 words):
1. Measure it: from the locked-exam plain predictions, extract aligned single-word substitutions (gold word -> predicted word, same position after alignment). Quantify semantic closeness (e.g. MiniLM cosine of the two words, WordNet-style relation if available) versus proper baselines: random word pairs; frequency-matched substitutes; and, crucially, what a language model alone would substitute in that slot given the context (e.g. T5 or GPT-2 top alternatives with the gold word masked). The key confound: the decoder's English prior may produce plausible substitutes regardless of the code. Report numbers.
2. Decide what can be claimed: "the ring carries meaning", "the decoder prior fills gaps plausibly", or both; how to separate them (e.g. compare substitutes under the true ring vs a ring with that word's information disrupted; distance in code space; paraphrase pairs giving nearby rings).
3. Propose the 2-3 cleanest follow-up experiments (cheap, local or <=10 H100 minutes) and what each outcome would mean. Say whether this is worth leading a post with.
Be concrete; numbers over adjectives.
