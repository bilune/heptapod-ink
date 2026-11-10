Gonzalo’s observation survives measurement: **the reader’s wrong words are more related to the originals than random, frequency-matched, or context-only T5 alternatives.** This supports a pattern in the system’s errors. It does not yet locate that pattern in the ring’s geometry rather than the pretrained encoder, adapted decoder, or their interaction.

I analyzed all 2,000 locked natural examples in `preds_exam_locked_natural_2000.json`, using `plain`. All inference used `.venv/bin/python`, `HEPTAPOD_DEVICE=cpu`, offline cached models, two CPU threads, and timeouts of at most 165 seconds; each job finished below three minutes. No training or checkpoint changes occurred. These are exploratory analyses of an already-inspected exam.

I lowercased, normalized apostrophes, tokenized words and numbers, and performed unit-cost Levenshtein alignment, preferring diagonal matches/substitutions on ties. I excluded numeric substitutions and ignored punctuation/case. There were **2,808 alphabetic word substitutions** in 1,240 sentences. The primary analysis uses **1,279 isolated substitutions** in 944 sentences: adjacent alignment operations must be exact matches or sentence boundaries. This reduces arbitrary pairings inside multiword rewrites. Among these, 465 were the sentence’s only word-level edit.

Each word was embedded separately with `all-MiniLM-L6-v2`, using attention-masked mean pooling and normalized cosine. All 2,808 substitutions averaged **0.432**; isolated substitutions averaged **0.476**.

For the crucial control, base `google-t5/t5-small` received the original sentence with the target word replaced by `<extra_id_0>`. It saw both surrounding contexts, with no ring. I generated ten beams, stopping at `<extra_id_1>`, with an eight-token cap and length penalty zero; retained distinct, completed, single-word alternatives; and excluded the gold word. At least one alternative survived for **1,200/1,279** slots. “Top alternative” below means highest-ranked surviving beam, not an exhaustive vocabulary optimum.

Frequency controls use the 32,210-word Tatoeba training vocabulary: stored sampling weights are square roots of counts, so squaring recovers frequencies. For each error, replacement words came uniformly from the predicted word’s `floor(log2(count))` bin. Counts existed for 1,097 isolated substitutes. This is a frequency proxy from Tatoeba, not the complete mixed training corpus.

The table compares the **same 1,035 substitutions across 800 sentences**, where both controls were available. Random baselines exclude the gold word and use exact pool expectations rather than Monte Carlo draws.

| Replacement source | Mean gold–replacement cosine |
| --- | ---: |
| Actual plain-reader error | **0.484** |
| Uniform random training-vocabulary word | 0.218 |
| Random substitute from this cohort’s empirical prediction distribution | 0.256 |
| Frequency-matched random word | 0.268 |
| Context-only T5: top alternative | 0.364 |
| Context-only T5: mean of up to five alternatives | 0.357 |

The paired excess over T5 is **0.120**, with a **95% sentence-cluster bootstrap interval of 0.108–0.132** (2,000 resamples, seed 404). T5’s top alternative equals the reader’s error in **88/1,035 cases (8.5%)**; its first five contain it in **186/1,035 (18.0%)**. Even selecting the most semantically similar of those five using the gold word gives only **0.442**, although that is a gold-assisted diagnostic, not a deployable baseline.

Two sensitivity checks retain the effect. For 372 cases with exactly one word-level edit, reader/T5 cosines are **0.493/0.367**. Excluding pairs with normalized character similarity ≥0.5 leaves 865 cases at **0.457/0.358**. Thus, inflectional or spelling similarity alone does not explain the aggregate difference.

These are **relatedness scores, not synonym percentages**. MiniLM gives boat→ship **0.595**, purple→green **0.658**, and yes→no **0.734**. The latter two occur in the locked exam and change the facts. A high cosine does not establish preserved meaning. No WordNet relation labels were computed.

The evidence supports **message-specific information plus plausible decoder completion**. The separate searched-ring result, 81.0% exact versus 1.2% with shuffled searched images, establishes dependence on the transmitted image. It is not a target-word ablation for plain rings. Meanwhile, the context-only baseline itself produces related alternatives, but does not reproduce the full observed similarity. The residual is suggestive, not a causal decomposition: base T5 differs from the adapted reader, receives ideal surrounding text, and the comparisons condition on actual reader failures. Part of speech, names, pretrained geometry, and selection into the common cohort remain confounds. “Meaning rather than spelling” is therefore too strong; a learned lexical code and semantic priors can coexist.

There are useful precedents. [Dell et al. (1997)](https://pubmed.ncbi.nlm.nih.gov/9337631/) model semantic paraphasias through lexical retrieval; the analogy concerns error structure, not identical mechanisms. [Ben Zion et al. (2024)](https://papers.nips.cc/paper_files/paper/2024/hash/c6f83c27a2223d817f9f1ade48d281a2-Abstract-Conference.html) distinguish semantic consistency and spatial meaningfulness in emergent communication. [Wang et al.](https://arxiv.org/abs/2407.14112) explicitly use language-model sequence priors in communication decoding. [Gilbert et al. (2023)](https://arxiv.org/abs/2304.12512) distinguish exact from semantic reconstruction, while [Zhao et al. (2025)](https://arxiv.org/abs/2509.10061) formulate semantic distortion constraints. Here, training minimizes token reconstruction loss; no explicit semantic distortion objective establishes a semantic rate–distortion optimum.

I would run two frozen-model experiments on fresh development material, with hard caps of four H100 minutes each, including setup; neither requires training:

1. **Hold context fixed and intervene on word evidence.** Build 120 sentence frames with target, synonym, antonym, and unrelated alternatives, balanced for frequency and T5 pieces. Encode every version. For each target–unrelated pair, replace exactly the differing slots: this produces an actual writer-generated counterfactual ring. Compare equal-size, activity-matched control patches. Under an identical teacher-forced prefix, measure exact-target probability and probability mass on a predefined semantic family. Repeat symbolically and through rendered images. Family probabilities following the counterfactual word, beyond control patches, demonstrate causal semantic information conveyed by the code. Stable related guesses despite changed word evidence favor contextual priors; effects confined to exact identity support lexical detail without establishing semantic geometry.

2. **Test geometry independently of decoding.** Encode 120 matched synonym/antonym/unrelated/spelling-neighbor sets and 60 paraphrase-versus-role-reversal triplets. Measure discrete Hamming distance and within-slot codebook-vector distance, controlling token count, spelling overlap, frequency, and position. Compare with the pretrained encoder’s geometry. Consistently closer synonyms and low-overlap paraphrases support semantic organization; similarity explained by spelling or encoder geometry narrows the novelty. A null result rejects this distance-based claim, not every possible semantic representation.

**Worth leading a post with as a measured puzzle:** “Its wrong words are more semantically related than context-only T5’s guesses.” Put the 0.484-versus-0.364 comparison and the yes→no counterexample beside that sentence. Reserve “the ring writes meanings” for the intervention results.
