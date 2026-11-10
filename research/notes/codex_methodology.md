# Methodology audit for Heptapod Ink

Date: 2026-11-08. This is a reading and design document, not a report of new training runs.

## 0. Scope, provenance, and what the existing evidence supports

I read the full local texts, including appendices, of the five assigned papers; the requested project documents and `v2/heptapod_ink.py`; the supporting `v1/heptapod.py`; all Python modules in `egg/core`; and the relevant zoo training, data, architecture, evaluation, intervention, parameter-grid, and analysis code. The zoo audit includes `compo_vs_generalization`, `compo_vs_generalization_ood` and its `learning_alone` control, `language_bottleneck`, `channel`, `basic_games`, `pop` including its sanity checks, `compositional_efficiency`, `emcom_as_ssl`, `objects_game`, `summation`, `simple_autoenc`, `mnist_autoenc`, `mnist_vae`, and `dsprites_bvae`. I read the code/markdown cells of the bottleneck style-transfer and OOD analysis notebooks. This does not mean that every stored run log, notebook image, or unrelated EGG game was independently reproduced.

Code citations beginning `egg/` are relative to `research/repos/EGG/`; other project paths are relative to the workspace root. Citations refer to the local checkout at commit **`54eb11c6b4ba3d3171811378f7ad7a117fa93a97`**, dated 2025-05-06. Papers are the corresponding `.txt` files in `research/papers/`, with PDFs alongside. Paper findings and my interpretations/proposals are explicitly distinguished below. The repository's archived status is supplied by the assignment; the local README does not establish an archive date.

The five sources are:

- **C20:** Chaabouni et al., *Compositionality and Generalization in Emergent Languages* (ACL 2020).
- **C19:** Chaabouni et al., *Anti-efficient encoding in emergent communication* (2019).
- **EGG19:** Kharitonov et al., *EGG: a toolkit for research on Emergence of lanGuage in Games* (2019).
- **R22:** Rita et al., *Emergent Communication: Generalization and Overfitting in Lewis Games*, local arXiv v2, October 2022.
- **S20:** Lazaridou and Baroni, *Emergent Multi-Agent Communication in the Deep Learning Era* (2020 survey).

Local full-text provenance (SHA-256 prefixes):

| Text file under `research/papers/` | SHA-256 prefix |
|---|---|
| `chaabouni2020_compositionality_generalization.txt` | `6237537c02d72700` |
| `chaabouni2019_anti_efficient_encoding.txt` | `5785b77da1f2b227` |
| `kharitonov2019_egg.txt` | `bedffb333b627bad` |
| `rita2022_generalization_overfitting_lewis.txt` | `f42ef2ea4680ba85` |
| `lazaridou2020_emergent_communication_survey.txt` | `788887d51629ddc4` |

### 0.1 Four claims, not one score

1. **Communication:** the original pair reconstructs meanings through an image.
2. **Systematic generalization:** it reconstructs prespecified unseen combinations or structures.
3. **Compositional organization and acquisition:** reusable message components have stable semantic effects, and new listeners can acquire the protocol from limited examples.
4. **A circular-layout advantage:** at a matched channel and learning budget, the circular rendering improves a prespecified outcome over a linear rendering.

C20 demonstrates that claim 2 can succeed with low values of its compositionality metrics. R22 demonstrates that the original listener can obscure what a speaker's messages make learnable. Neither paper compares circular and linear visual scripts. None establishes a threshold at which a protocol becomes a “real language.” A private convention can be compositional. Rotation invariance is also not arbitrary permutation invariance: a ring retains orientation, adjacency, and relative order.

### 0.2 Current project facts and corrections to the earlier review

The current meaning space is `(subject, verb, object, tense, polarity)` with sizes `(6,6,6,3,2)`, hence **1,296 meanings**, or **10.33985 bits** under the uniform prior. The existing split holds out 36 of 216 subject–verb–object triples, with all six tense/polarity combinations: **1,080 train / 216 test**, and no separate development set (`v1/heptapod.py:42`, `:74`, `:79`). This tests missing triples, not new syntax or recursion.

The v2 speaker chooses a ring segment and a whole mark at each of 12 slots. The code selects 32 nonblank ring segments and prepends blank, and prepends blank to the mark library (`v2/heptapod_ink.py:43`, `:50`, `:65`, `:81`). With the present 83 nonblank marks this gives **33 × 84 = 2,772 joint slot symbols**, not an 83-symbol channel. The two tracks can both carry information. Rendering uses wrapped angular placement and overlapping soft unions (`v2/heptapod_ink.py:94`). Blank is meaningful absence, not an EOS token.

Important limitations of the existing evidence:

- `evaluate()` enters evaluation mode but unconditionally returns the agents to training mode; the subsequent topographic function does not enter evaluation mode (`v1/heptapod.py:262`, `:277`, `:282`). Since v2 samples in training and takes argmax in evaluation (`v2/heptapod_ink.py:174`), the existing topographic result can measure sampled messages instead of the deployed greedy code.
- The existing topographic statistic correlates raw pixel Euclidean distances with semantic distances, uses `argsort().argsort()` and jitter instead of average ranks, and shuffles a pair-distance vector for its null (`v1/heptapod.py:282`). It is not a trustworthy substitute for the paper's tie-aware Spearman statistic or a cyclic symbol-distance analysis.
- `rotate=0` in evaluation does **not** mean a clean channel: blur, noise and occlusion still apply (`v1/heptapod.py:235`). Python's RNG drives occlusion, while v2's training routine seeds only Torch (`v1/heptapod.py:253`; `v2/heptapod_ink.py:196`).
- v2 uses AdamW, a one-cycle schedule, a soft-to-hard curriculum, changing noise, dynamically weighted attribute losses, and an ink-target penalty (`v2/heptapod_ink.py:196`). These are substantive differences from a fixed categorical reconstruction game. The squared ink-target penalty penalizes too little as well as too much ink; it is not a hard message-capacity constraint.
- Checkpoints save model states without the complete configuration, RNG state, optimizer/scheduler state or mark-library hash (`v2/heptapod_ink.py:237`). High-resolution `paint()` also changes contrast, whereas `say()` decodes a different low-resolution image (`v2/heptapod_ink.py:280`, `:349`). Exported-image accuracy has not thereby been measured.
- The holistic baseline trains its lookup table on **all** meanings and has different optimization (`v1/heptapod.py:186`, `:314`); it is not a valid unseen-composition baseline.
- The earlier review's proposed zero-example subject-role holdout needs correction. Subject and object have separate input embeddings and output heads (`v1/heptapod.py:160`, `:210`; `v2/heptapod_ink.py:161`). Withholding an entity from the subject role leaves its subject embedding untrained and its subject-output class without positive examples. A failure then mixes protocol generalization with an untrained interface. Use full role/value coverage first; test stronger role transfer only with an explicit shared representation or a controlled few-shot exposure design.

**Correction to `v3/CODEX_REVIEW.md`:** the current `v2/out/metrics.json` is now an **8,000-step** run, with train exact **0.8231**, held-out exact **0.5683**, reported topographic **0.3689**, and mean marks **2.99** (`:3`, `:14`, `:22`, `:30`, `:37`). The review's “30-step / 0%” statement is stale. `v2/out_run3/metrics.json` separately records **0.9880** train exact, **0.9387** held-out exact, reported topographic **0.4226**, and **10.31** mean marks, with **40** ring segments (`:5`, `:12`, `:20`, `:28`, `:35`). These results must not be merged into a single current-configuration claim. Their topographic values inherit the issues above.

The scope note in `v3/PLAN.md:1` drops human decipherment and printed-photo tests. This document follows that scope: new and cross-seed **model** listeners, not human readability, are the relevant outsiders.

## 1. Paper-by-paper evidence

### 1.1 C20: compositionality is not necessary for the tested generalization

#### Setup actually reported

**Game and data:** a reconstruction game. A meaning is a tuple of categorical attributes, encoded as concatenated one-hot vectors. The sender emits a fixed-length discrete string; the receiver predicts every attribute. The sweep varies numbers of attributes and values, message length and vocabulary, with channel capacity at least the number of meanings. The body describes 2–4 attributes, 4–100 values, lengths 3, 4, 6, 8 and vocabularies 5, 10, 50, 100; Appendix Table 3 contains additional small configurations, and Appendix Table 7 extends lengths to 10 and 12. Use the tables, not an inferred full Cartesian product of the prose ranges. There are ten random initializations per setting (C20 §§2–3, Appendix Tables 3 and 7).

**Agents and optimization:** single-layer GRUs with 500 hidden units for both agents; a linear map initializes the sender from the input. The final receiver state feeds attribute classifiers. The receiver minimizes mean attribute cross-entropy by backpropagation; the sender uses REINFORCE with a running-mean baseline and an entropy bonus. Adam learning rate is **0.001**. Training messages are sampled; test messages use greedy decoding (C20 §2).

**Evaluation:** random **90% training / 10% held-out combinations**. The main analysis conditions on successful training, **>99.9% training accuracy**. Compositionality is measured on training meanings, deliberately avoiding a definition that rewards test success directly (C20 §3.2). This conditional analysis is not an unconditional success rate over all attempts.

**Not stated in the paper as a universal setting:** a single batch size or entropy coefficient for every sweep. A current local grid uses batch 5,120, data scaling 60, 3,000 epochs, entropy coefficient 0.5, sender/receiver embeddings 5/30, hidden size 500 and seeds 0–9 (`egg/zoo/compo_vs_generalization/hyperparams/gridS27.json:6`). These are checkout configuration facts, not permission to fill every paper-methods gap with today's defaults.

#### Numerical findings and their locations

| Finding | Reported result | Location |
|---|---|---|
| Metrics measure different properties | Spearman correlations: topographic/positional **0.08**, topographic/bag **0.38**, positional/bag **0.31**, all p<0.01 | §3.1 |
| Larger meaning spaces favor generalization in this sweep | Input-space size versus generalization **ρ=0.86**, p approximately zero | §4, Figure 1 |
| Little evidence for a general composition–generalization correlation | Only **4 of 141** configurations have a significant correlation for any of the metrics, at p<0.01 | §5 |
| Strong generalization can coexist with low metrics | In the 2×100, length-3, vocabulary-100 setting, games above **98%** generalization span positional **0.02–0.72**, bag **0.03–0.40**, topographic **0.11–0.64** | §5, Figure 2 |
| High scores sometimes accompany good generalization, but are not a theorem | Among games with score >0.5, proportions with generalization >0.8 are **0.90 positional**, **0.50 bag**, **0.11 topographic** | §5 |
| Input density matters | Holding 10,000 observed examples while increasing value counts 100→140→200, densities **1 / 0.51 / 0.25**, gives generalization **92.7% / 66.7% / 22.8%** | Appendix §8.4 |
| Channel size alone is a weak explanation | Generalization versus capacity **ρ=0.14** | Appendix §8.5 |
| Larger channels tend to lower disentanglement | Capacity correlations: positional **−0.094** (p=.0058), topographic **−0.208**, bag **−0.574** (latter two p<10⁻⁶) | Appendix §8.6 |

For sufficiently large meaning spaces (at least 625 meanings in the discussed sweep), the smallest successful capacity/input-space ratios average **5.9**, SD **4.4** (C20 §4). This is an observed optimization result, not a required number of bits or a law that every new channel needs sixfold redundancy. Reducing the GRU to 100 units does not rescue generalization in the small 2×10 condition (Appendix §8.3, Figure 3).

**Causal inspection:** Table 1 studies a highly generalizing language with positional score about **0.70**. Preserving only position 1, 2 or 3 while shuffling the other two positions across examples gives `(attribute 1, attribute 2, both)` accuracies of **(1,3,0)**, **(1,68,0)**, **(89,1,1)** percent. Shuffling positions 1, 2 or 3 gives **(89,69,61)**, **(100,3,3)**, **(1,100,1)** percent. Thus the language is mostly disentangled but has “leakage”; a score is not a complete semantic analysis. Appendix Table 5 reports positional MI pairs **(1.10,2.01)**, **(0.19,4.16)**, **(4.44,0.13)** for this example. A low-positional-score language, approximately **0.05**, also generalizes near **99%**. Table 1 averages ten random shuffles (reported SD approximately zero) over training and test messages that the original pair correctly decoded. These interventions are a useful precedent for our component perturbation tests, but they are not held-out productive semantic-substitution tests; shuffling is also not deletion.

**New-listener acquisition:** §6 selects 30 games with generalization at least 0.8, freezes senders and trains fresh receivers: GRU-500, GRU-50 and a two-layer 500-unit feedforward network, with three receiver initializations. Learning speed is the area under the training-accuracy learning curve, not an intrinsic property independent of training budget. Table 2, for the main length-3/vocabulary-100 setting, reports:

| Metric | Speed correlation, GRU500 / GRU50 / FFN | New-listener generalization correlation, same order |
|---|---|---|
| Positional | **.87 / .71 / .35** | **.80 / .55 / .50** |
| Bag | **.85 / .68 / .33** | **.81 / .55 / .51** |
| Topographic | **.87 / .71 / .35** | **.79 / .54 / .48** |

All are significant at p<0.01. Across broader channel settings, Appendix Table 7 gives mean speed correlations positional **.75/.62/.61**, bag **.72/.63/.59**, topographic **.79/.67/.62**, and mean generalization correlations **.71/.42/.54**, **.72/.49/.51**, **.70/.51/.63**, respectively; some long-message conditions are weaker. These are correlations among selected successful languages, not randomized proof that increasing a metric will cause better transfer.

Appendix §8.2/Table 4's constructed languages give `(topographic, positional, bag)` scores **(.82,1,.42)**, **(.75,.79,.13)** and **(.75,.43,1)**. This is a direct warning against treating the three metrics as interchangeable.

#### Limits and implications

**Paper's scope/limitations:** finite tuples and juxtaposition, not recursive syntax, function application or unrestricted semantics; conditional analyses on trainable languages; acquisition studied with frozen speakers, not full generational cultural transmission. The paper leaves richer generalization and propagation questions open.

**Interpretation for our ring:** success on held-out tuples is necessary evidence about our chosen split, but cannot by itself establish compositionality. Low topographic similarity cannot by itself falsify systematic communication. Fresh-listener learning curves are especially well motivated. Our meaning density, channel size and number of seen combinations must be reported explicitly. Whole-mark perturbations should inspect whether role/value effects transfer across contexts, rather than hoping that a single metric certifies “language.”

### 1.2 C19: brevity and efficient coding do not emerge automatically

#### Setup actually reported

**Game:** reconstruct one of **1,000 atomic identities**, presented as a one-hot vector. Inputs follow a truncated Zipf distribution, approximately `p(rank r)=1/(r H_1000)`, so the most frequent identity has probability about 0.13. These identities have no compositional internal structure and there is no unseen-combination test (C19 §2).

**Channel:** alphabet sizes **3, 5, 10, 40, 1,000**, including EOS, and maximum lengths **2, 6, 11, 30**, with configurations restricted to adequate capacity. Messages terminate at EOS; EOS is forced at the limit. If `a` includes EOS and total length includes its terminating position, the number of possible messages is

`C(a,Lmax) = Σ_{l=1}^{Lmax} (a−1)^(l−1)`.

This is not `a^Lmax`. The optimal code assigns shorter admissible strings to more frequent ranks, with `length(r)=min{n: Σ_{k=1}^n(a−1)^(k−1) ≥ r}` (C19 §2 and optimal-code comparison).

**Agents/optimization:** single-layer LSTMs. Appendix §A1.1 lists sender/receiver hidden-size pairs **(100,100), (250,100), (250,250), (500,250)**, entropy coefficients **1, 1.5, 2**, and four random seeds, giving **48 runs per configuration**. Adam learning rate **0.001**; REINFORCE sender with running-mean baseline and differentiable cross-entropy receiver. Sample during training, greedily decode for evaluation. Training uses **2,500 episodes × 100 batches × 5,120 examples**. Final evaluation covers all 1,000 inputs uniformly. The analysis retains runs above **99%** accuracy and configurations with more than three successful runs; average successful runs are approximately **25/48** (C19 §2, Appendix §A1.1).

**Comparisons:** optimal prefix-like rank allocation over the admissible EOS strings; monkey-typing strings with EOS probability `1/a`, enforcing unique messages; and the 1,000 most frequent words in English, Arabic, Russian and Spanish. These natural-language comparisons concern frequency and length, not an equivalence between an atomic game and human language.

#### Results with numbers

The central qualitative finding is **anti-efficient encoding**: frequent meanings tend to receive long messages, especially at maximum length 30, contrary to Zipf's law of abbreviation (C19 §3.1, Figure 1). Appendix §A1.6/Table A1 reports frequency-weighted lengths:

| Alphabet a | Optimal | Monkey typing | Emergent |
|---|---:|---:|---:|
| 5 | 3.55 | 7.56 | **26.98** |
| 10 | 2.82 | 11.27 | **26.73** |
| 40 | 2.29 | 21.30 | **29.40** |
| 1,000 | 1.86 | 29.67 | **29.98** |

For reference, that table gives English **3.68** and Arabic **3.14**; the regularized `a=40` condition has length **7.22**. Its permutation tests use **100,000 permutations**: the anti-efficient direction has p<10⁻⁵ for alphabets 5, 10 and 40, but **p=.928** for 1,000, where lengths already saturate. Monkey typing at 40 and 1,000 is also not significantly efficient in that table (p=.186 and .25). Do not generalize the significant result to every alphabet or rely on the appendix's inconsistent prose about the null instead of its table.

**Mechanism probes:** 90 untrained speakers, 30 at each hidden size 100/250/500, behave more like monkey typing (§3.2.1). Fifty untrained listeners with hidden size 100 produce more separated hidden representations for the learned long messages (§3.2.2, Figure 3). This suggests an architectural/optimization advantage for long strings; it is not an isolated causal identification of every source of the effect.

**Length cost:** adding `α × message length` to the loss (§3.2.3, Eq. 4) shortens messages. For `a=40, Lmax=30, α=.5`, Figure 4 still shows an ascending frequency–length pattern among the most frequent 40 items. Larger penalties slow learning and reduce convergence. Appendix §A1.8 reports that α=1 can bring the `a=1,000` case near optimal while stronger pressure can prevent successful communication. A brevity penalty is a tradeoff, not a guaranteed route to human-like coding.

**Redundancy:** among the top ten symbols, `P(ss)>P(s)^2` holds in more than **97.5%** of the length-30 runs; comparison is with 25 randomized controls matched for length and unigram statistics (§3.3). Appendix entropy analyses use natural logarithms; do not silently compare them numerically to EGG's bit-valued entropy. Appendix §A1.5 also finds a significant difference under uniform-input training (t-test p<10⁻⁹), so the entire effect is not simply a restatement of the Zipf input prior.

#### Limits and implications

**Paper's limits:** atomic meanings; a particular LSTM reconstruction setup; bounded length, with 30 limited by practical resources; natural-word comparisons are restricted statistics; explicit length regularization can impair optimization. This paper does not measure compositionality or establish that short emergent strings generalize.

**Interpretation for us:** “looks sparse” or “uses few marks” is not evidence of an efficient compositional language. Charge and measure ring-segment choices, mark choices, gaps, ink area, redundancy and errors separately. If we test abbreviation, introduce a prespecified nonuniform meaning prior; the current uniform meanings cannot demonstrate a frequency–brevity law. Our ink-target penalty is different from this paper's linear message-length cost. A whole complex mark can carry more identity information than a thin stroke even at the same occupied-slot count.

### 1.3 EGG19: useful abstractions, not a validation result

**Purpose and game abstraction:** EGG19 presents a toolkit for predominantly unidirectional, single-round sender/receiver games, with wrappers handling the communication estimator and a game loss handling the task (EGG19 §§2–3). It is not a paper with one canonical vocabulary, one optimizer, or one empirical compositionality benchmark.

**Training estimators (§3):** Gumbel–Softmax samples

`y_i = exp((log p_i + g_i)/τ) / Σ_j exp((log p_j + g_j)/τ)`.

Straight-through variants use a hard forward symbol with a relaxed backward gradient. REINFORCE uses the score-function gradient, typically with a baseline; the receiver can still be trained by ordinary backpropagation. These optimize different estimators of a discrete protocol, and soft evaluation can silently become a continuous channel.

**Worked experiment (§4):** MNIST image reconstruction using a pretrained LeNet-1 encoder, LSTM hidden size **20**, vocabulary **10**, maximum message length **2**, REINFORCE, **15 epochs**, batch **32**, and pixel cross-entropy. The receiver maps a message embedding/final recurrent state to image-sized pixel probabilities in [0,1]. The paper does not provide a sufficiently complete decoder-layer or optimizer/lr specification to infer those details here. Figure 5 probes all **100 two-symbol strings**. EOS is symbol 0, so a first-position EOS causes the second symbol to be ignored: 100 plotted probes do not imply 100 distinct effective messages. The figure illustrates first-symbol digit information and second-symbol style effects, including digit-dependent behavior; it does not report a numerical systematic-generalization score, compositionality score or circular-layout comparison.

**Limits:** a toolkit demonstration, a qualitative latent-code illustration and the then-supported game family. Multi-turn/multi-direction and richer analysis were future work. The current local `egg/zoo/mnist_autoenc/train.py` is a one-symbol Gumbel–Softmax MLP example, not an exact implementation of the paper's LSTM/REINFORCE demo. Today's defaults must not be retroactively attributed to the paper.

**Interpretation for us:** reuse the separation of sender, channel, receiver, task loss and interaction logs. Do not inherit EOS semantics, a sequential receiver, or a nominal message-length statistic when the actual channel is a fixed-size rendered ring. The paper provides no MPS guarantee and no evidence that visual beauty, discreteness or a toolkit's metric callback validates language.

### 1.4 R22: distinguish information loss from listener overfitting

#### Exact theoretical distinction

For input prior `p(x)`, speaker `πθ(m|x)` and listener `qφ(x|m)`, the reconstruction negative log likelihood is

`L(θ,φ) = E_{x,m}[−log qφ(x|m)]`

`= Hθ(X|M) + E_{m∼pθ(m)} KL(pθ(X|m) || qφ(X|m))`.

The posterior is `pθ(x|m)=p(x)πθ(m|x)/pθ(m)`. R22 §2, Eq. 5 calls these **information loss** and **co-adaptation loss**. The speaker affects both terms; changing the listener alone changes only the second. A low training task loss can coexist with high test loss because the speaker discarded relevant information, because the listener failed to acquire an available decoding rule, or both. Train–test gaps are signed differences; do not assume every empirical estimate is nonnegative.

The equality uses the full Bayesian posterior. With a finite-capacity probe, a factorized multi-head receiver, optimization error and finite samples, probe NLL is only an approximation to the information term, generally including residual decoder limitations. Subtracting an imperfect probe from the original listener can even yield a negative estimated adaptation component. For our multi-head output, mean attribute CE is also a scaled factorized loss, not automatically the exact joint-posterior decomposition.

#### Setups and evaluation

**Synthetic game (R22 §4.1):** six categorical attributes, ten values each: **10⁶ meanings**. Disjoint uniform samples: **4,000 train, 1,000 validation, 1,000 test**, less than 1% observed coverage. Six random seeds; training accuracy above 99% except noted failures. Messages have **10 tokens** and vocabulary **10**, with a hardcoded final EOS. The visual-game appendix (§D.1.3) describes the same nominal length/vocabulary capacity as `10^10`; because EOS budgeting can mean different things, preserve that reported convention and audit the implementation before claiming an exact bit-for-bit reproduction. The cited R22 implementation is a separate Population repository, not the current EGG `pop` game.

Sender: one-hot input of size 60, linear projection to 128, layer-normalized LSTM with hidden size 128. Receiver: embedding and layer-normalized LSTM, size 128, with six ten-way output heads. Adam **5×10⁻⁴**, β=(.9,.999), batch **1,024**; REINFORCE sender, batch-mean reward baseline and entropy coefficient **.01**. Validation controls stopping/model selection (R22 §4.1).

**Crucial measurement convention:** the synthetic “Gen.” results evaluate a **fresh probing listener** trained on training data and stopped using validation, and report **mean attribute accuracy**, not exact six-attribute reconstruction and not necessarily the original listener's accuracy (R22 §4.2). Thus .95 in Table 1 cannot be compared directly with our .9387 exact-match result.

To estimate information loss, §3.1 trains probing listeners on the training distribution for the training term, and on the **full input population distribution** for the test information term. This is a diagnostic with access to the population, not a permissible held-out generalization training procedure. The ordinary transfer/generalization probe remains trained only on training examples. The paper's topographic measurement samples 1,000 full-population meanings and stochastic messages, repeats 100 times and averages; this differs from C20's training-set emphasis and greedy test convention (§4.2).

**Visual games (§5.3, Appendix §D.1):** referential identification from frozen pretrained BYOL ResNet-50 image features, dimension **2,048**. Images are center-cropped; this is not an image-valued communication channel. Sender/listener use 256-dimensional projections, embeddings and layer-normalized LSTMs. The receiver scores candidates by dot product. Each target competes with **999 uniformly sampled distractors**, without replacement and excluding the target. Message setting is again ten tokens / vocabulary ten. Adam learning rates **5×10⁻⁴ sender**, **10⁻³ listener**, batch **2,048**, entropy **.01**. Base image partitions are 80/10/10, with approximately 1.4 million ImageNet and 200,000 CelebA images. Table 3 gives the actual reduced-set counts: CelebA **8,492 / 2,123**, ImageNet **50,732 / 12,683** for the columns labeled “1/20” and “1/100.” Use the listed counts; the shorthand labels should not be converted into invented exact counts.

#### Interventions and effect sizes

R22 does **not** claim that co-adaptation is inherently bad. It can help establish a shared convention. The problem is co-adaptation that overfits the training examples. In §5.1/Figure 2, both training and test information-loss estimates quickly approach zero, while the test co-adaptation loss grows: in this setting much of the failure comes from the listener's decoding generalization rather than loss of input information in the message. This is not a universal statement that every sender always preserves all information. An injective arbitrary code can also have zero conditional entropy while being very hard to acquire from sparse examples.

1. **Balancing the objectives:** §3.2 optimizes `(1−α)L_info + αL_adapt`, with α from 0 to .5; .5 corresponds to the ordinary objective up to scale. Its speaker reward can be written `(1−2α) log q_probe + α log q_original` (Appendix B). Figure 2/§5.1 reports approximately **15 percentage points** better generalization when increasing α from 0 to .5. The co-adaptation term helps; removing it is not the remedy. α>.5 is unstable in the reported formulation.
2. **Partial listener training:** reset the listener **after every speaker update**, then train it for N steps (§3.3). This is much stronger and more expensive than occasionally restarting our CNN. Figure 3 finds underfitting for **N<50**, useful regimes roughly **50<N<250**, and overfitting for **N>250** in this setup. These counts are not transferable constants.
3. **Early-stopped listener training:** also resets the listener, but uses validation to determine when to stop training it (§3.3). This adapts the amount of co-adaptation instead of fixing N.
4. **Ordinary regularization:** dropout .2, removal of LSTM layer normalization, and weight decay .01, individually and combined. Appendix C.1 chooses values by average generalization subject to at least 75% convergence. “No LN” does not mean removing all normalization from an unrelated image CNN.

R22 Table 1, synthetic game:

| Training variant | Fresh-probe attribute generalization | Topographic | Estimated test co-adaptation loss |
|---|---:|---:|---:|
| Continuous training | **.58 ± .05** | **.22 ± .02** | **4.64 ± 1.22** |
| Dropout | .64 ± .03 | .24 ± .01 | 4.86 ± .52 |
| No layer normalization | .70 ± .03 | .24 ± .02 | 4.68 ± .38 |
| Weight decay | .72 ± .03 | .25 ± .03 | 4.29 ± .56 |
| No LN + weight decay | .87 ± .07 | .30 ± .03 | 2.12 ± .67 |
| Early-stopped listener | **.95 ± .04** | **.39 ± .04** | **1.10 ± .69** |
| Best partial-training setting | **.95 ± .03** | **.42 ± .02** | **.97 ± .55** |

The early-stopped improvement is **37 percentage points**, with topographic similarity increasing by **.17**. The table's ± values are reproduced as reported; they must not be relabeled as standard errors without an explicit definition. Selecting the best partial-training setting is different from a single prespecified hyperparameter.

Speaker regularization is less reliable. Appendix C.2/Table 2 adds speaker weight decay .005: the continuous baseline moves from .58 to **.62 ± .02** generalization, but the combined no-LN/weight-decay listener result drops from .87 to **.82**. Do not assume symmetric speaker/listener regularization is optimal.

Visual retrieval, R22 Table 1:

| Dataset | Continuous: full / 1⁄20 / 1⁄100 | Early-stopped listener: same order |
|---|---|---|
| CelebA | **.94±.01 / .67±.02 / .39±.07** | **.97±.01 / .80±.03 / .69±.04** |
| ImageNet | **.96±.01 / .77±.01 / .51±.03** | **.98±.01 / .81±.01 / .64±.01** |

The largest listed visual gain is CelebA 1/100, **+30 percentage points**; ImageNet 1/100 improves **+13 points**. The improvement is therefore especially important when observed data are sparse, but does not imply a comparable gain for our ring script.

There is **no corresponding consistent visual topographic increase**. Appendix Table 4, reduced sets: CelebA continuous **.28/.32**, early **.27/.30**; ImageNet **.17/.17** versus **.18/.20**. Using CelebA attribute distances instead of visual-feature cosine distances still gives essentially **.13/.15** for both methods (Table 5). Table 6 shows that regularization helps some reduced-data conditions, but early stopping remains strongest overall: weight decay yields CelebA **.77/.60**, ImageNet **.79/.62**; no-LN plus weight decay fails to converge for CelebA 1/20. Generalization and this geometry metric can move independently.

#### Limits and application to our project

**Paper's limits:** controlled Lewis games and particular architectures; expensive reset/probe procedures; approximate loss decomposition through learned probes; dependence on validation and training budget; topographic similarity has limited interpretability for natural-image games. Its visual experiments send discrete strings about images, not rendered ink images. Populations and generational transmission are discussed in the wider literature but are **not the tested remedies in this paper**.

**Interpretation/application:** record both original-listener and fresh-listener accuracy. If a fresh listener learns a frozen speaker's training code and generalizes better, the original listener is part of the failure. If both fail despite a successful engineered code, the learned protocol or its acquisition is suspect. First test ordinary listener regularization and validation-based checkpoint selection. If needed, implement the paper's actual reset-and-train-inner-loop procedure as a separately budgeted experiment. Do not call a periodic reset an exact R22 replication, and do not call it iterated cultural learning. A population with independently initialized listeners may reduce pair-specific conventions, but that is a new intervention whose benefit must be measured.

### 1.5 S20: what the survey does and does not establish

This is a survey, not a single experiment. **There is no one exact vocabulary, length, optimizer, agent count, or numerical treatment effect to report.** Those fields are heterogeneous or unspecified across the reviewed works. Attributing one setup to the whole survey would fabricate precision.

Relevant setups and claims:

- §2.1 contrasts discrete, reinforcement-trained communication with differentiable/continuous training such as RIAL and DIAL. A continuous training channel and a discretized test channel can have materially different capacities and optimization problems.
- §2.2/Figure 3 describes the classic visual referential setup: roughly **500 image categories**, target plus **one distractor**, and a single symbol from a finite alphabet, commonly with pretrained visual features and neural agents. The survey does not supply a universal alphabet size or optimizer for this family. The illustrated numeric food/baked-food/pizza symbols are examples of conventions, not measured generalization results.
- §3/Figure 4 discusses communication that survives replacement of natural images with noise. The lesson is that a successful image game can rely on low-level shared distinctions rather than object categories. A two-choice game may need only a binary distinction for a particular context, even when its image pool contains hundreds of categories.
- §3.1 distinguishes **positive signaling** (messages depend on observations) from **positive listening** (the receiver's behavior causally depends on the message). MI alone tests association; message interventions are needed to test use. Other observed actions or turn-taking can be communication channels too.
- §3.2 discusses topographic similarity, systematic generalization, disentanglement and tree-reconstruction approaches. Generalization and compositionality are related questions, not interchangeable measurements. Tree reconstruction error requires a chosen compositional structure and composition operator; it is not assumption-free.
- The survey reports work where a third agent encourages a common convention rather than pairwise idiolects, but it gives no single controlled population effect size that can be transplanted to our setting. Population and transmission effects depend on the game, learning bottleneck and agents.
- §§4.1–4.3 discuss optimization differences between discrete/continuous channels, the limitations of cooperative cheap-talk setups, nonstationarity and partner co-adaptation. §5 emphasizes the limited treatment of richer syntax, pragmatics and human-compatible language.

**Stated scope:** a synthesis of deep-learning emergent communication, with unresolved evaluation problems and limited linguistic complexity. It does not validate a particular ring metric or assert that naturalistic-looking marks induce human language.

**Interpretation for us:** retain reconstruction instead of making a weak two-candidate retrieval task our principal evidence. Test role reversals and negation, intervene on messages, check that no target metadata or renderer parameters reach the listener, and include fixed-code controls. A strong retrieval score or nonzero MI can coexist with a shortcut. Population training is worth a later controlled experiment; the survey is motivation, not a numerical guarantee.

## 2. Metrics: definitions, implementation, and circular adaptations

### 2.1 A common evaluation contract

Let `X=(X₁,…,X_A)` be a categorical meaning, `M=(M₁,…,M_L)` a hard discrete message, and `Y` the received image. For v2 a slot is the **joint pair** `(ring_segment_id, whole_mark_id)`. Let `R_k M` shift all slots by k. The unanchored symbolic object is the cyclic orbit `[M]={R_k M}`. Shift both tracks together; independently shifting ring and mark tracks destroys their relative arrangement. Reflection is a separate transformation: quotient by rotations only unless reflection invariance is explicitly part of the task.

For each checkpoint:

1. Freeze parameters, preserve the prior train/eval state, enter evaluation mode and run without gradients. Use a fixed meaning list and hard greedy messages for the primary deployed-code analysis.
2. Save meaning IDs, hard symbol IDs, original-listener predictions and the actual received-image condition. Do not infer symbol identity by treating every raster pixel as a token.
3. Separately sample repeated messages from a frozen policy to measure stochastic-channel behavior. Label this result “sampled”; do not mix it with greedy results or with training logs collected while the policy was changing.
4. Compute metrics on train, development and untouched test meanings separately. An additional full-meaning-space descriptive analysis is permissible after model selection, explicitly labeled; it does not train a model or choose a hyperparameter.
5. Move discrete analysis arrays to CPU and use real categorical attributes, not integer Euclidean distances or an accidentally flattened one-hot representation. Save distance definitions, normalizations, EOS/blank rules and random seeds.

### 2.2 Entropy and mutual information

EGG implements empirical plug-in entropy in **bits**:

`Ĥ(Z)=−Σ_z p̂(z) log₂ p̂(z)`,

and empirical MI:

`Î(U;V)=Ĥ(U)+Ĥ(V)−Ĥ(U,V)`.

Implementation: `egg/core/language_analysis.py:45` (`entropy_dict`), `:62` (`calc_entropy`), `:77` (`_hashable_tensor`), `:85` (`mutual_info`). Counts are normalized internally. A row is treated as one discrete outcome. The hash conversion handles scalar or one-dimensional tensor rows; a nested image tensor is not a suitable categorical outcome. `mutual_info` uses `zip` without validating equal lengths, so mismatched arrays can silently corrupt the joint distribution. Empty/all-zero counts and too-small samples need explicit handling.

`MessageEntropy` (`egg/core/language_analysis.py:103`) optionally argmaxes Gumbel messages and counts full rows. It does **not** itself truncate at EOS. Consequently padding/post-EOS samples can inflate entropy unless sanitized first. Training callback entropy can also aggregate messages from multiple policies across an epoch.

**Ring adaptation:** count canonical cyclic representatives, e.g. the lexicographically smallest whole-symbol rotation, to obtain `Ĥ([M])`. This canonicalization is appropriate for **identity counting**, even though it is not a neutral semantic anchor for positional analysis. Measure `Ĥ([M]|X)` with repeated frozen-policy draws and use `Î(X;[M])=Ĥ([M])−Ĥ([M]|X)`. Report raw orientation entropy separately if orientation is allowed to carry information. Empirical MI with a large alphabet is upward biased; compare against meaning-label permutations and show sample-size sensitivity. A deterministic injective lookup code has `I(X;M)=H(X)` while being wholly noncompositional.

A useful additional diagnostic is the optimal decoding ceiling from collisions. For a deterministic uniform code over N meanings, the best possible exact accuracy is `number_of_distinct_orbits/N`. More generally it is `Σ_m max_x p(x,m)`. Estimate stochastic collision/ambiguity behavior from repeated samples. This tests recoverability, not composition. Rasterization or noise may introduce additional collisions beyond symbolic orbit collisions.

### 2.3 Topographic similarity

For distinct meaning indices i<j, form semantic distances `a_ij=d_X(x_i,x_j)` and message distances `b_ij=d_M(m_i,m_j)`. Topographic similarity is

`TopSim = Spearman({a_ij}_{i<j}, {b_ij}_{i<j})`

`= Pearson(average_rank(a), average_rank(b))`.

For categorical attributes, a natural semantic distance is normalized Hamming:

`d_X(x,x′)=(1/A) Σ_a 1[x_a ≠ x′_a]`.

EGG's static implementation is `TopographicSimilarity.compute_topsim`, `egg/core/language_analysis.py:167`. It calls `scipy.spatial.distance.pdist` for each side (`:198`) and `scipy.stats.spearmanr(..., nan_policy='raise')` (`:201`). Supported named distances (`:174`) are:

- `edit`: Levenshtein distance divided by **mean sequence length**, `(len(x)+len(y))/2`, not maximum length;
- cosine;
- normalized Hamming;
- Jaccard;
- Euclidean.

The default pair is meaning Hamming / message edit. The callback flattens sender inputs (`:208`). On concatenated one-hot categorical meanings, Hamming differs by a constant scale from attribute mismatch when block sizes are fixed, and cosine likewise reflects overlap; for arbitrary embeddings or variable encodings that equivalence does not hold. Jaccard on nonbinary integer IDs is not a sensible default notion of categorical similarity. Euclidean distance between arbitrary category IDs invents an order. EGG's compo intervention code instead uses cosine over one-hot inputs and edit distance (`egg/zoo/compo_vs_generalization/intervention.py:108`, `:120`); do not silently mix conventions.

**Pitfalls:**

- Semantic distances have many ties. Use average ranks, not `argsort().argsort()` and random jitter. The familiar no-ties formula `1−6Σd²/[n(n²−1)]` is inappropriate without tie correction.
- A constant distance vector makes the statistic undefined. Report NA and the reason, not zero. `nan_policy='raise'` does not turn every constant-input warning into a useful result.
- Pair distances share observations. The number of pairs is not the number of independent experimental units; ordinary pairwise p-values are overconfident.
- Different distance choices answer different questions. Raw pixel distance combines typography, alignment and semantic coding. It should be a secondary perceptual statistic alongside a symbol-level statistic.
- Sampling messages adds within-meaning variation; evaluation mode and sampling must be specified. C20 and R22 do not use identical evaluation distributions or message-sampling protocols.
- A high value can be engineered without usable compositional rules, and a generalizing code can score low. Neither .5 nor the current project's .42 is a validated certification cutoff.

**Ring distances:** for fixed-length whole-symbol messages, use

`d_cyc-H(M,N)= min_{k∈{0,…,L−1}} (1/L) Σ_j 1[M_j ≠ N_(j+k mod L)]`.

This is a well-defined distance between cyclic orbits because Hamming distance is invariant under simultaneous cyclic shifts. A joint symbol can be scored as one mismatch; a secondary weighted sum of ring/mark mismatches is possible, but must be fixed before viewing results.

If edit distance is needed for variable-length strings, use a declared circular dissimilarity such as

`d_cyc-edit(M,N)= min_{r,s} Lev(R_r M,R_s N) / ((|M|+|N|)/2)`.

**Minimizing over only one cut is not generally enough for linear Levenshtein distance**, which is not invariant to simultaneously moving both string boundaries. The two-cut definition makes the score independent of the arbitrary serialization boundary; do not assert triangle inequalities for an arbitrary normalized variant without checking them. Empty strings need a stated convention. If angular gaps or slot spacing carry information, removing blank positions and comparing only occupied-mark strings changes the channel and is invalid as the primary metric.

For images, a separate rotationally aligned raster dissimilarity can minimize distance over a fixed angle grid. It measures a perceptual geometry, depends on interpolation/resolution, and is not a replacement for the symbolic cyclic metric. Use the same angle search and numerical fidelity for the line's rotation-robustness analysis.

**Inference:** calculate a meaning-label permutation null by permuting which meaning is associated with each complete message, then recomputing all pair distances/correlation. Do not independently shuffle the flattened pair-distance vector. Use repeated independent training seeds for claims about learned protocols. Report the seed distribution, not just a tiny p-value from hundreds of thousands of dependent pairs.

### 2.4 Positional disentanglement (posdis)

For each nonconstant position j, let `a₁(j)` and `a₂(j)` be the attributes with the largest and second-largest empirical MI with `M_j`. Then

`posdis = (1/|J|) Σ_{j∈J} [ I(M_j;X_a₁(j)) − I(M_j;X_a₂(j)) ] / H(M_j)`,

where `J={j:H(M_j)>0}`.

This is the C20 §3.1 positional measure. EGG implements the shared MI-gap calculation in `egg/core/language_analysis.py:20`, and exposes the static positional method at `:295`. It excludes constant positions from the denominator, not merely from the numerator. It requires at least two attributes. If every position is constant, its current division is 0/0; wrap that as NA plus an explicit collapse flag. The helper allocates its result tensor on CPU (`:21`), another reason to run discrete analysis on CPU.

**Interpretation/pitfalls:** a position scores highly if its variation is associated primarily with one attribute. The score does not ensure all attributes are represented, that distinct positions represent distinct attributes, or that the receiver uses those positions. Duplicating one attribute across the whole message can score well. Correlated input attributes blur the top-two gap; parity/XOR-like interactions can carry information with little univariate MI. Unequal attribute entropies, sparse samples, selection on converged runs and large symbol inventories all matter. A gap of zero can mean no information or equal information about two attributes; these are different cases.

**Ring adaptation:** there is no intrinsic “position 1” in an unanchored ring, so ordinary posdis is not rotation invariant. Report three distinct things rather than presenting a silently modified metric as the original:

1. Raw serialization posdis, explicitly labeled an orientation-dependent diagnostic.
2. Posdis after a **label-independent** canonicalization, with sensitivity to token-ID ordering. Lexicographic minimum rotation is valid for orbit identity but can move the apparent anchor whenever a symbol changes, creating or hiding positional structure.
3. If the protocol develops an identifiable anchor, fit an anchor detector using only training messages/images, freeze it, and report aligned posdis plus the proportion with unique/unambiguous anchors. Do not discard ambiguous test cases to inflate the score, and do not choose a shift using the gold held-out meaning.

No anchor is a substantive outcome, not a missing technical detail. Relative-offset symbol relations and cyclic n-gram histograms can be analyzed with MI, but those are **new diagnostics**, not the paper's posdis. An oracle alignment chosen to maximize attribute MI on the evaluation set leaks semantics into the score.

### 2.5 Bag-of-symbols disentanglement (bosdis)

For symbol v let `C_v(M)=Σ_j 1[M_j=v]`. For each nonconstant count variable, find its two highest-MI attributes. Then

`bosdis = (1/|V*|) Σ_{v∈V*} [ I(C_v;X_a₁(v)) − I(C_v;X_a₂(v)) ] / H(C_v)`,

where `V*={v:H(C_v)>0}`.

C20 §3.1 defines this count-based variant. EGG constructs histograms at `egg/core/language_analysis.py:284` and drops symbol 0 at `:291`, treating it as EOS. It counts across the provided entire sequence before dropping that column; it does **not** truncate all subsequent tokens at the first EOS. Sanitize variable-length messages first. Its illustrative doctest supplies a vocabulary smaller than some symbol IDs (`:249`); do not copy that as valid vocabulary handling.

**Ring adaptation:** counts are already invariant under cyclic shifts, but also under every permutation. Thus bosdis cannot distinguish a valid role-binding circle from a bag with the roles scrambled. For v2, include meaningful blank/absence counts, count joint `(ring,mark)` symbols and optionally report separate ring and mark marginals. Blank 0 is not EOS here. A 2,772-way joint alphabet yields extremely sparse count variables; show permutation bias and coverage, and prefer the deliberately smaller alphabet in the falsification experiment.

The same missing-attribute, redundancy, correlation and interaction pitfalls apply as for posdis. A high bosdis with low role-reversal accuracy is a plausible bag code, not evidence of a complete compositional language.

### 2.6 Remaining analysis utilities and useful complementary metrics

`language_analysis.py` contains entropy, MI, topographic similarity, positional/bag disentanglement and printing callbacks; it does **not** implement a general syntactic-compositionality test or TRE. `Disent.print_message` calls nonexistent `self.disent` for both scores (`egg/core/language_analysis.py:298`), so the callback is broken in this checkout even though its static functions are usable. `PrintValidationEvents` (`:328`) prints examples rather than defining another metric.

For our experiment also report:

- **Exact meaning accuracy:** `mean_i 1[∀a: x̂_ia=x_ia]`.
- **Attribute accuracy:** `mean_{i,a} 1[x̂_ia=x_ia]`, plus each attribute separately and the subject/object reversal confusion rate. Do not equate it with exact accuracy.
- **Unweighted predictive NLL:** `mean_i Σ_a −log q_a(x_ia|Y_i)`; optionally divide by A, explicitly. Keep this separate from entropy-regularized/RF surrogate loss, v2's dynamically weighted loss and ink penalties.
- **Acquisition:** new-listener exact/attribute accuracy against number of distinct training meanings and optimizer updates; normalized AUC over a common update interval. C20 motivates AUC, but its value depends on the architecture and training schedule.
- **Positive listening:** compare normal messages with shuffled-message, blank-image and targeted-message interventions (S20 §3.1). Shuffling messages across examples should substantially hurt a genuinely used channel; this alone does not establish composition.
- **Controlled component interventions:** change one whole mark or a prespecified relative-position pattern, test the predicted attribute change and preservation of every other attribute on held-out contexts. Fit the proposed correspondence on training data only. Include random substitutions and a hand-coded protocol.
- **Redundancy/partial reveal:** accuracy versus the number of visible whole marks, for contiguous and scattered masks; average over reveal origin and mask choices. Report retained ink as well as retained slots. This tests information distribution, not linguistic compositionality or “knowing the end before the beginning.”
- **Cost:** occupied marks, ring-segment variability, occupied slots, mean ink area, realized message entropy, collisions and nominal alphabet/length capacity. These quantities are not interchangeable.

The project's footprint localization (`v1/heptapod.py:373`) measures where pixel changes lie around the circle. It is descriptive and can report a localized footprint even for a negligible change; it does not demonstrate a reusable semantic constituent.

## 3. Capacity and a defensible ring-versus-line comparison

### 3.1 What the papers actually control

C20 fixes vocabulary and maximum/fixed message length in each setting, requires adequate nominal capacity and sweeps capacity relative to meaning-space size. It does not prove that equal `V^L` implies equal learnability. C19 explicitly accounts for EOS and variable length and compares expected lengths under a nonuniform prior. R22 fixes a small discrete channel across interventions; the capacity convention around forced EOS must be reproduced exactly. EGG19's examples illustrate both relaxed and discrete estimators. S20 warns that task context and continuous channels can substantially change what must be communicated.

For a reconstruction game, there is no candidate-set shortcut: the listener must identify the complete meaning. In a referential game with N candidates, the immediate selection problem can require only `log₂ N` bits conditional on the candidates, and shared visual information can reduce what the message supplies. Do not compare retrieval accuracy with full reconstruction as if their capacity demands were identical.

### 3.2 Nominal, quotient, empirical and perceptual capacity

For an ordered fixed-length line over K symbols, nominal capacity is `K^L` messages, or `L log₂ K` bits. For an unanchored oriented ring where cyclic shifts are equivalent, the number of necklaces is

`N_necklace(K,L) = (1/L) Σ_{d|L} φ(d) K^(L/d)`.

Periodic messages have smaller orbits, so simply dividing `K^L` by L is an approximation, not always an exact count. Do not quotient by reflections unless mirror images are also equivalent. A learned unique anchor may break the ambiguity, but the anchor consumes alphabet/slot resources and must be charged.

For v2's current nominal `K=33×84=2,772`, `L=12`, the ordered upper bound is approximately **137.2405 bits** and the cyclic-orbit bound approximately **133.6556 bits**, versus only **10.33985 bits** of uniform meaning entropy. These are combinatorial upper bounds before raster collisions/noise, not measurements of effective transmitted bits. Mark-count pressure does not eliminate the ring-segment channel. A soft mixture of symbols can exceed the intended discrete identity budget even if its argmax sequence looks small.

Realized entropy `H([M])` measures usage, not maximum capacity. Mutual information after noise measures recoverable information, not composition. Robustness depends on glyph separability, clipping, overlap, orientation, sampling and the receiver; these must be controlled or measured rather than folded into a single “bits” claim.

### 3.3 Concrete matching recipe

For an initial geometry test, simplify to **one fixed noninformative scaffold and one whole-mark alphabet**. Do not retain a freely chosen ring-segment track in one condition and omit it in the other.

1. Use the same hard symbol alphabet, blank convention, slot count, encoder output distribution and codebook in both layouts. Define the codebook as one canonical representative of each cyclic orbit. Both line and ring therefore have exactly the same number of permitted messages; the line is explicitly a **canonical-line control**. This tests rendering geometry at a common discrete channel. It does not represent every possible unconstrained linear language.
2. Also state the ecological comparison one might later want: unrestricted ordered lines versus unanchored rings with equal raw alphabet/slots. That comparison has different combinatorial capacities; report the difference or capacity curves instead of calling it bit-matched.
3. Render the **same intact mark templates**, at the same resolution, stroke mass and center-to-center path spacing. Place them on a circular path or straight path. Do not give the ring elaborate whole marks while giving the line arbitrary bars, or unwrap the raster in a way that slices marks at seams.
4. Use the same image canvas, CNN architecture and parameter budget, losses, augmentations, learning budgets, checkpoint rule and random seeds. Match rotations in physical angle; give the line arbitrary global rotations too if that is the comparison. Test clean and corrupted channels separately.
5. Measure surviving ink/marks under occlusion. A square masking the same fraction of canvas can erase very different fractions of a short central circle and a long line. For the primary reveal experiment, erase a fixed number of whole marks, with mask origins randomized; report pixel masks as a secondary, physically motivated test.
6. Evaluate only hard discrete messages in the primary result. If a Gumbel relaxation is used for training, report the relaxed-to-hard gap and ensure mixture weights, logits, latent IDs and coordinates never reach the receiver.
7. Verify rendered collisions and positive-control decoding before interpreting a learned-language failure. A capacity count is not enough if two nominal marks become identical after downsampling.

A capacity-matched codebook is an experimental design choice proposed here, not a method tested by these five papers. Its virtue is an exact, auditable count; its limitation is the common canonicalization constraint. A later study should vary rate and renderer scale to establish whether any observed advantage survives that choice.

## 4. EGG reuse audit

### 4.1 Core components: usable pieces and necessary boundaries

“As-is” below means the specific function or small component under its stated contract, not a recommendation to install the entire dependency stack and trust every callback. No runtime/MPS compatibility test was performed during this read-only assignment.

| Component | Reuse decision | Evidence / changes needed |
|---|---|---|
| Empirical entropy and MI | Copy the small functions with MIT notice, after validating nonempty aligned CPU categorical arrays | `egg/core/language_analysis.py:45`, `:62`, `:85`; finite-sample bias and nested-image hashing remain caller responsibilities |
| Topographic static function | Reuse for CPU arrays with custom distance callables and explicit NA handling | `egg/core/language_analysis.py:167`; use cyclic distances, average ranks already supplied by SciPy, and a separate permutation procedure |
| Positional MI-gap helper | Reuse for prepared anchored categorical messages, at least two attributes and at least one nonconstant representation | `egg/core/language_analysis.py:20`, `:295`; guard all-constant case |
| Bag measure | Copy and change EOS/blank handling and truncation | `egg/core/language_analysis.py:284`; ring blank 0 must not be automatically dropped |
| Disentanglement callback | Do not use unchanged | `egg/core/language_analysis.py:298` calls missing `self.disent` |
| `Interaction`, `LoggingStrategy` | Useful schema, with disciplined field population | `egg/core/interaction.py:16`, `:59`; `__add__` assumes dictionary-like `aux_input` (`:103`), while `from_iterable` is the safer aggregation path (`:162`) |
| Baselines | Running-mean baseline is a reusable estimator; checkpoint its state explicitly | `egg/core/baselines.py:41`; state is not an `nn.Module` buffer. Leave-one-out built-in baseline has a small-batch special case (`:67`) |
| Symbol-level REINFORCE game | Useful starting point for a categorical **whole-codebook** sender and differentiable image receiver | `egg/core/reinforce_wrappers.py:22`, `:121`; a plain wrapper's argmax dimension assumes one categorical action. Multi-slot policy requires correct joint log probability |
| Symbol-level Gumbel game | Useful pattern if rendering is in the receiver path and gradients traverse it | `egg/core/gs_wrappers.py:15`, `:116`; its logged message length is one (`:196`), so replace logging for a multi-slot image protocol |
| Deterministic wrapper | Reusable interface around an ordinary receiver where applicable | `egg/core/reinforce_wrappers.py:95` |
| Population samplers | Small pair-sampling patterns are useful, not a ready protocol-training experiment | `egg/core/population.py:12`, `:36`, `:69`; avoid global NumPy seeding, save sampler/RNG state and evaluate all partner pairs |
| Trainer | Can run a game with an explicit device, but needs audit/adaptation | `egg/core/trainers.py:51`, `:154`, `:165`, `:200`; losses are averaged per batch rather than per example, logs can be costly, CUDA AMP/distributed assumptions remain |
| Console logging / interaction saver | Useful interfaces after deciding which fields to retain | `egg/core/callbacks.py:68`, `:269`; don't save every 192×192 image at every step without a storage budget |
| Checkpoint saver | Insufficient alone for exact continuation/provenance | `egg/core/callbacks.py:180`; add RNGs, baseline, sampler, split IDs, full config and library hash; trainer loading at `egg/core/trainers.py:321` needs device-aware loading |
| Reconstruction loss | Copy the idea, write a small variable-cardinality version for 6/6/6/3/2 | `egg/core/losses.py:31`; do not import fixed/equal-attribute assumptions into our five heads |

For a slot-factorized RF sender, the score-function term is the **sum** of log probabilities of all sampled active decisions, multiplied by a detached per-example advantage. The receiver's CE is differentiated normally. Entropy regularization and baselines should not be confused with the reported task NLL. The core variable-length implementation sums effective log probabilities (`egg/core/reinforce_wrappers.py:580`) and masks using EOS lengths (`:563`); it also includes differentiable loss terms in its optimized scalar (`:602`). Copying the optimized scalar into a “language NLL” plot would be wrong.

### 4.2 What changes for an image channel

EGG's standard RNN receiver consumes symbol embeddings; ours must consume **only the rendered and corrupted image**. The necessary boundary is

`meaning → discrete whole-symbol message → renderer → image channel → CNN → five attribute heads`.

Store the pre-render symbols in the interaction for analysis, but never pass them to the listener. The renderer must enforce discrete alphabet identity, whole marks and fixed capacity. If using RF, the renderer need not be differentiable with respect to the sampled symbols; if using GS, gradients must traverse the renderer while hard evaluation is checked separately.

Do not reuse the variable-length GS game unchanged. It treats token 0 as EOS and forms an expectation over per-step receiver losses (`egg/core/gs_wrappers.py:418`, `:513`). The RNN sender appends a forced EOS (`egg/core/gs_wrappers.py:357`; RF equivalent `egg/core/reinforce_wrappers.py:222`). Our blank slot is not a stopping decision, and the CNN reads the completed image once. A custom fixed-size game is simpler and less error-prone than pretending a ring is an EOS-terminated sentence.

The core `dump_interactions` enters/restores evaluation mode, which is a useful pattern (`egg/core/interaction.py:281`, `:299`, `:332`). But its default device chooses CUDA/CPU, it can argmax and truncate EOS-bearing sequences, and its accumulation can encounter optional-field assumptions (`:304`, `:316`, `:320`, `:326`). Write a small explicit image-protocol dump rather than relying on every default.

### 4.3 MPS: source-level concerns, not unperformed runtime claims

- `egg/core/util.py:133` selects CUDA if available and otherwise CPU; it does not auto-select MPS. Pass an explicit `torch.device('mps')` through a compatible custom runner or adapt initialization. `Trainer` does accept an explicit device in its non-distributed path (`egg/core/trainers.py:51`, `:154`).
- `RelaxedEmbedding` checks concrete CPU/CUDA long tensor classes (`egg/core/gs_wrappers.py:229`). An MPS integer tensor can take the matrix-multiplication branch instead of the embedding branch. Replace backend-specific type checks with dtype checks before reuse.
- CUDA AMP and CUDA distributed paths are not MPS features (`egg/core/trainers.py:33`, `:160`, `:214`; `egg/core/interaction.py:230`). Disable these paths for a single-device MPS run; do not treat CUDA `.is_cuda` tests as a general accelerator abstraction.
- Discrete SciPy/edit-distance analysis belongs on CPU; some MI-gap allocations are CPU-only (`egg/core/language_analysis.py:21`). RNN packing explicitly transfers lengths to CPU (`egg/core/rnn.py:72`); test the chosen runtime rather than assuming all historical packing/backward combinations work identically on MPS.
- The project already contains a manual straight-through Gumbel implementation with an MPS-related comment (`v2/heptapod_ink.py:145`). The comment is not a verified statement about every current PyTorch release. A finite-gradient/finite-loss smoke test on the actual machine is required before choosing an estimator.
- Record Python/PyTorch/macOS versions and hardware. Bitwise reproducibility across CPU, CUDA and MPS is not promised by seeding. Seed Python, NumPy and Torch; save RNG states. Core `_set_seed` covers those families (`egg/core/util.py:239`), unlike the current v2 training entry point.

### 4.4 Zoo audit: actual behavior versus the name or paper

#### `compo_vs_generalization`

Useful: meaning enumeration/one-hot conversion, exact and per-attribute reconstruction metrics, frozen-sender receiver retraining, message interventions, systematic channel sweeps (`egg/zoo/compo_vs_generalization/data.py:13`, `:66`; `egg/zoo/compo_vs_generalization/train.py:93`, `:166`, `:175`, `:323`; `egg/zoo/compo_vs_generalization/intervention.py:144`).

Important departures/defects:

- `split_holdout` excludes every meaning containing zero from the main pool, puts meanings with exactly one zero into a special holdout, and discards meanings with multiple zeros (`egg/zoo/compo_vs_generalization/data.py:76`). A separate seeded random split is applied to the remaining pool (`:91`). This is more than “a random 90/10 split.”
- The normal validation loader is the **unscaled training data**, while a custom callback reports held-out data (`egg/zoo/compo_vs_generalization/train.py:205`, `:213`, `:275`, `:301`). Early stopping on this validation is effectively training-accuracy stopping. Do not reproduce this naming in our experiment.
- Special OOD loss logic selects zero-valued examples and drops the associated attribute from evaluation (`egg/zoo/compo_vs_generalization/train.py:115`). It is not full reconstruction of the held-out value, and `.squeeze()` creates empty/single-example edge cases.
- `PlusOneWrapper` increments all symbols, including forced EOS (`egg/zoo/compo_vs_generalization/archs.py:69`; instantiated `egg/zoo/compo_vs_generalization/train.py:255`). The result has a constant terminal symbol and no zero EOS. Account for that extra array position without treating it as an extra free decision.
- Fresh-receiver retraining gates on original training accuracy >.99, whereas the paper's selected-language discussion conditions on generalization ≥.8 (`egg/zoo/compo_vs_generalization/train.py:323`). Its architecture variants and feedforward activation differ from the paper's description. Seeds 17–19 occur in the local retraining loop (`egg/zoo/compo_vs_generalization/train.py:410`).
- Old receiver-statistics access assumes a dictionary where the current trainer returns an `Interaction` (`egg/zoo/compo_vs_generalization/train.py:363`); the feedforward receiver signature is incompatible with the newer four-argument receiver call. These paths need repair before they constitute a replication.
- The intervention evaluator changes mode and returns to train mode, and averages batch accuracies without weighting unequal batch sizes (`egg/zoo/compo_vs_generalization/intervention.py:144`, `:195`). Capture state and aggregate counts instead.

The README points to an older EGG version for reproduction. Pinning the present checkout is necessary for this audit but not sufficient to reproduce the published experiments.

#### `compo_vs_generalization_ood` and `learning_alone`

`get_data` deliberately exposes the all-zero and selected single-zero examples in training before assessing zero-valued generalization (`egg/zoo/compo_vs_generalization_ood/train.py:72`). At 100 values and two attributes, its split yields **8,824 train, 980 uniform holdout, 196 special OOD**; at 50 values, **2,164 / 240 / 96**. This is an exposure design, not pure zero-shot introduction of entirely unseen input embeddings. It is directly relevant to correcting our proposed role holdout.

The OOD training loss disables the other game's attribute-dropping evaluation (`egg/zoo/compo_vs_generalization_ood/train.py:162`), but its normal validation still points to training data (`:106`) and OOD reporting every epoch (`:173`) invites tuning leakage if used for selection. The encoder reshapes batch/sequence axes with `.view` rather than transposing (`egg/zoo/compo_vs_generalization_ood/archs.py:30`), which can mix example contents; do not reuse it unchanged. Attention code has backend-specific scalar/device handling (`egg/zoo/compo_vs_generalization_ood/archs.py:67`).

`egg/zoo/compo_vs_generalization_ood/learning_alone/data.py:15` creates hand-specified symbol mappings, duplication and EOS/padding; this is supervised acquisition of a constructed language, not iterated learning. Its training script uses Adam defaults despite an exposed learning-rate option (`egg/zoo/compo_vs_generalization_ood/learning_alone/train.py:113`). The results notebook's first code cell fills absent results with zeros and carries forward incomplete trajectories; a later aggregation averages over both runs and epochs rather than simply final performance (`egg/zoo/compo_vs_generalization_ood/ntb-results-full.ipynb:50`, `:64`, `:161`). Treat those notebook plots as analysis choices, not authoritative final-result extraction.

#### `channel`

This is the relevant atomic-identity/length-efficiency game: power-law sampling, reconstruction CE, uniform and frequency-weighted evaluation (`egg/zoo/channel/train.py:210`, `:152`, `:160`). It subtracts one from the requested maximum length to compensate for the wrapper's appended EOS (`:200`), exactly the kind of bookkeeping that matters for capacity. It trains over the same identity inventory rather than held-out compositions. Its checked-in parameter file is a short example configuration, not C19's 2,500×100×5,120 experiment. The README's version guidance matters.

#### `language_bottleneck`

This directory concerns entropy/bottleneck experiments associated with another paper, *Entropy Minimization in Emergent Languages*, not an extra result from the five assigned papers.

- `guess_number` sends a prefix of a binary vector while the receiver can see a suffix; the relevant information demand is conditional on receiver side information. Its BCE/accuracy logic is useful, but the old `np.int` use is obsolete (`egg/zoo/language_bottleneck/guess_number/features.py:115`; `egg/zoo/language_bottleneck/guess_number/train.py:125`).
- The intervention code has an old five-argument loss API, and in a variable-message branch stores shuffled messages alongside original labels, invalidating an MI interpretation if copied literally (`egg/zoo/language_bottleneck/intervention.py:121`, `:141`, `:143`, `:164`). Its empirical oracle/majority calculation is a useful diagnostic concept (`:284`).
- `AlwaysRelaxedWrapper` sends softmax vectors even in evaluation (`egg/zoo/language_bottleneck/relaxed_channel.py:24`). This is a continuous channel, not an alphabet-sized hard-symbol protocol.
- `mnist_classification` uses paired digits but `(10×first+second) % n_labels`; with default ten labels the target depends only on the second digit (`egg/zoo/language_bottleneck/mnist_classification/data.py:31`; `egg/zoo/language_bottleneck/mnist_classification/train.py:59`). It cannot establish that both digits were compositionally represented. Its old wrapper keyword is incompatible with current core APIs (`egg/zoo/language_bottleneck/mnist_classification/train.py:87`).
- `mnist_overfit` deliberately corrupts labels and defaults to a relaxed channel, with an explicit force-discrete option (`egg/zoo/language_bottleneck/mnist_overfit/data.py:10`; `train.py`). It illustrates overfitting but does not implement R22's listener reset experiment. `mnist_adv` likewise does not supply a complete adversarial generalization evaluation merely by its name.
- The style-transfer notebook contains a local vocabulary-size assignment that is not wired into the actual options and uses old trainer keywords (`egg/zoo/language_bottleneck/mnist-style-transfer-via-bottleneck.ipynb:182`, `:199`). Its image examples are not a quantified new-listener or compositionality result.

#### `basic_games`

The reconstruction and discrimination losses are small useful references (`egg/zoo/basic_games/play.py:145`, `:195`). Train/test sets are externally supplied files; there is no automatic semantic held-out split. The Perl data generators enforce certain uniqueness conditions, not systematic generalization, and do not establish a fixed reproducible seed by themselves. An architecture `.squeeze()` can remove a batch dimension for batch size one (`egg/zoo/basic_games/architectures.py:43`). Use our own explicit split and shape-safe loss.

#### `pop`: populations, not a ready Rita implementation

The README describes heterogeneous pretrained-model populations. `egg/zoo/pop/seq_train.py:66` starts with a pair; later additions train with the accumulated population (`:105`). This is not generational replacement through a finite transmitted corpus.

- `egg/zoo/pop/games.py:6` and `:32` import missing `scripts` modules; `egg/zoo/pop/data.py:20` imports a proximity sampler from a path where it is absent. Some related files reside under sanity checks instead. The discrete LSTM/GS setup reaches a `NotImplemented` branch (`egg/zoo/pop/games.py:258`). The default continuous experiment is not a discrete message-budget match.
- The game is in-batch retrieval with diagonal target labels (`egg/zoo/pop/games.py:43`), not full five-attribute reconstruction. Training uses an Adam/cosine setup (`egg/zoo/pop/train.py:76`) and assumes SLURM environment state (`egg/zoo/pop/train.py:44`).
- Pair selection samples random partners in training and cycles through selected pairs during validation, not a complete cross-product on every batch (`egg/zoo/pop/archs.py:565`, `:570`). `egg/zoo/pop/extract_com.py:64`, `:138` shows the more relevant idea of evaluating every sender–receiver pairing on a common dataset.
- `build_second_game` freezes old parameters and adds new partners (`egg/zoo/pop/games.py:289`), a useful conceptual control for acquisition. Frozen parameters do not by themselves guarantee all stateful training behavior has been frozen.
- Its RF implementation averages message log probabilities instead of using the sequence sum (`egg/zoo/pop/archs.py:661`); the logged message-length tensor is constructed from the second example and has the wrong shape (`:688`). The default device in another path is CUDA (`:709`). Do not lift this as a vetted fixed-length RF game.
- Data splits are random image partitions, not held-out semantic constructions; loaders drop final batches (`egg/zoo/pop/data.py:385`, `:413`, `:427`). The seeding helper treats zero specially and calls CUDA helpers (`egg/zoo/pop/data.py:205`). A weight-decay selector with an empty default skip-name matches every parameter name (`egg/zoo/pop/utils.py:365`).
- `egg/zoo/pop/build_cosine_data.py:48` calculates corresponding-row cosine similarities and broadcasts a scalar into a matrix block, rather than constructing all pairwise similarities. One classification sanity check aggregates outside the sender loop and reports only the last sender (`egg/zoo/pop/sanity_checks/test_classif.py:157`). The sparse-autoencoder helper penalizes reconstruction output rather than the latent activation (`egg/zoo/pop/homemade_sae.py:26`). These analyses should not be inherited as validated metrics.

Reuse the **experimental idea** of multiple partners and a cross-play matrix. Build a small population around our explicit game only after the single-pair baselines work. Zero-shot cross-play among independently trained pairs may fail because of arbitrary symbol permutations; that does not alone show a language is unlearnable. Report acquisition after matched examples separately from raw zero-shot compatibility.

#### Other directly relevant generalization examples

| Game | What it actually contributes | Caveat |
|---|---|---|
| `compositional_efficiency` | Hand-designed factorized versus mixed codes and receiver learning; useful negative control for assuming composition always learns fastest | Discrete transformations mix attributes modulo V (`egg/zoo/compositional_efficiency/discrete.py:83`, `:89`; `egg/zoo/compositional_efficiency/archs.py:31`, `:45`). The split uses Python's randomized string hash (`egg/zoo/compositional_efficiency/dataset.py:25`), so Torch seeds alone do not fix it. The continuous script constructs a test dataset then loads the **training dataset** as test (`egg/zoo/compositional_efficiency/continuous.py:64`). `CircleSender` means quantized 2D points, not a circular visual script (`egg/zoo/compositional_efficiency/archs.py:89`). |
| `emcom_as_ssl` | Frozen/learned visual representation games, in-batch communication, noise and clustering probes | `egg/zoo/emcom_as_ssl/train.py:52` uses SGD/momentum/cosine and optional LARC, not our reconstruction optimizer. Hardcoded OOD paths (`:98`) are not portable. Its contrastive mask retains the positive among “negatives” and then adds it again (`egg/zoo/emcom_as_ssl/losses.py:124`, `:128`, `:131`); related core NT-Xent code has the same issue (`egg/core/losses.py:143`). Candidate-set size changes the task. |
| `objects_game` | Attribute-structured object games and analytic shortcut baselines | Dataset uniqueness is of candidate tuples, not a held-out meaning inventory (`egg/zoo/objects_game/features.py:85`, `:139`). `egg/zoo/objects_game/util.py:31` computes a useful limited-information baseline. Success may reflect one attribute sufficient to select among distractors. |
| `summation` | Variable-message numerical task and length-sensitive behavior | Train/test use the same maximum-length distribution (`egg/zoo/summation/train.py:91`, `:96`), not extrapolation. A batch-reduced task loss (`:82`) is problematic inside a per-example EOS-weighted game. It also uses old NumPy/API conventions. |
| `simple_autoenc` | Small RF/receiver-backprop example | Train/test use the same finite meaning distribution; not compositional OOD evidence. |
| `mnist_autoenc` | Current one-symbol GS reconstruction demo | Different from EGG19's stated LSTM/REINFORCE demo. |
| `mnist_vae` | Continuous latent reconstruction baseline | `if self.train` tests a bound method, so evaluation still samples (`egg/zoo/mnist_vae/train.py:61`); BCE+KL (`:68`) is not a discrete-language metric. |
| `dsprites_bvae` | Factorized synthetic image data and attempted disentanglement callbacks | Same `if self.train` bug (`egg/zoo/dsprites_bvae/train.py:84`); continuous latent exact-count MI is unsuitable, the disentanglement callback is broken (`:197`), latent size is hardcoded, and every-100th-item subsampling can distort factor combinations (`egg/zoo/dsprites_bvae/data_loaders/data_loaders.py:14`). |

Additional core reasons to prefer a small copied/adapted runner: `egg/core/datasets.py:45` ignores a seed value of zero; its iterator boundary at `:88` can omit a final full batch; distractors at `:190` are sampled with replacement. `egg/core/continuous_communication.py:39` refers to an undefined hidden-size attribute for deeper senders and its message-length logging is not a reliable batch-length measure (`:125`). The legacy transformer decoder passes an old `static_kv` argument to attention (`egg/core/transformer.py:378`). None is needed for the proposed first experiment.

### 4.5 Licensing and archived-repository policy

The root `research/repos/EGG/LICENSE:5` grants the **MIT license**: copying, modification and redistribution are permitted subject to retaining the copyright and permission notice (`:13`). Preserve that notice in a vendored module and document the upstream commit and local changes.

There are embedded exceptions: `egg/zoo/emcom_as_ssl/LARC.py:9` carries a **BSD 3-Clause** notice with attribution, binary-redistribution and no-endorsement conditions. `egg/zoo/pop/sanity_checks/simplicial.py:1` identifies BSD-licensed upstream material and imports `xformers`; verify its upstream notice if copying it rather than assuming the root MIT text supersedes it. Neither file is necessary here. Dataset, pretrained-weight and Arrival-reference permissions are separate from EGG's code license; the project plan itself describes the reference material as CC BY-NC/local-use material (`v3/PLAN.md:7`). This audit does not expand those permissions.

The archived status argues for **vendoring a few audited functions**, not for discarding valid scientific ideas. `setup.py` reports version 0.1.0 and the dependency list includes old lower bounds and largely unpinned scientific/vision/logging packages. Importing core callbacks also imports optional-looking logging packages eagerly (`egg/core/callbacks.py:16`). Avoid the wholesale legacy zoo stack, old notebook APIs, SLURM orchestration, broken population imports, obsolete transformer paths and CUDA-specific distributed machinery. Preserve the paper/check-out distinction: a current script defect is not evidence that the historical paper's reported experiment had that defect.

## 5. Revised minimal falsification experiment

### 5.1 Claims and limits fixed before running

Test this operational claim:

> With a fixed hard whole-mark channel, an image-mediated protocol trained on a restricted set of the current five-attribute meanings generalizes to unseen subject–object combinations, remains learnable by fresh image listeners, and benefits from circular rendering on rotation-robust exact reconstruction relative to a capacity-matched canonical line.

Test local, reusable whole-mark semantics as a separate compositional-organization claim. Do not claim recursion, COGS-level structural generalization, human readability, arbitrary permutation invariance or unrestricted language from this experiment. C20 motivates separating generalization from compositional metrics; R22 motivates independent listeners; S20 motivates causal interventions; C19 motivates explicit channel accounting. **None supplies a universal 50%, 90% or topographic-score pass threshold.**

The following sizes, seeds, rendering choices and compute caps are a **new prespecified design**, not alleged settings from a paper. Where they differ from the papers, the reason is a small controlled test of our implementation. No timing or successful convergence is promised without measurement.

### 5.2 Exact meaning split

Enumerate all tuples `(s,v,o,t,p)` in lexicographic order, with `s,v,o∈{0,…,5}`, `t∈{0,1,2}`, `p∈{0,1}`. Split before rendering or augmentation:

| Partition | Rule | SVO triples | Full meanings |
|---|---|---:|---:|
| Test | `o=(s+1) mod 6` | 36 | **216** |
| Development | `o=(s+2) mod 6` | 36 | **216** |
| Training | All other subject–object offsets | 144 | **864** |

Every entity occurs as subject and object in training; every verb, tense and polarity occurs in every partition. All six grammatical variants of an SVO triple stay together. No training subject–object pair appears in development or test. The reversal of a test `(s,o)` pair has offset 5 and is in training; record both directions to detect role collapse. The receiver does not receive split identity or the modular rule. Also report the theoretical split-informed baselines, so exploiting the test relation cannot be mistaken for transmitting both roles independently.

This is stricter than the current random-triple split in one controlled respect: the missing relation is shared across all verbs. It is still a **finite recombination** test, not a new structural grammar. The 864/216/216 sizes are derived from an exact balanced partition of the existing space, not copied from R22's much sparser million-meaning experiment. Record coverage/density explicitly.

Optional follow-up, not needed to rescue a failed primary result: shuffle the 216 SVO triples with Python `random.Random(7)`, allocate the first 36 to test, next 36 to development and remaining 144 to training, again keeping all six grammatical variants together. This gives a matched-size random-combination split closer to C20. Report it as a different split, not a replacement chosen because it scored better.

### 5.3 Exact small channel and rendered controls

Use **K=7 symbols**: blank 0 plus six distinct whole-mark templates, and **L=6 slots**. Use one fixed scaffold per layout, with no learned ring-segment variations. Select templates before training, without looking at meaning labels: normalize the eligible whole marks into a common patch; select the first nonblank template by the smallest source-library index, then repeatedly select the template maximizing its minimum pixel distance from already selected templates, breaking ties by index. Record the six source IDs, source-file hash and actual rendered patches. This deterministic selection is a design proposal; inspect separability after the normalization rather than claiming it is guaranteed by farthest-first selection.

Define the common permitted codebook

`C = {lexicographically smallest cyclic rotation of w : w∈{0,…,6}^6}`.

It contains exactly

`|C|=(7^6 + 7^3 + 2×7^2 + 2×7)/6 = 19,684`

messages, or **14.26474 bits**. An unrestricted six-slot line has 117,649 strings, **16.84413 bits**, and is not the primary capacity match. Both experimental senders select only from C. The line renders the canonical representative in order; the ring renders the same representative around its cycle. Canonicalization must not depend on meaning labels. Store the codebook and hash. A renderer test must confirm that the 1,296 engineered-code messages remain distinguishable under the clean channel; count additional collisions over C if feasible.

A concrete geometry is a **192×192** grayscale canvas, six slot centers spaced by **24 pixels of path length**, and whole marks fitted inside **16×16** patches with the same resampling and ink normalization. For the circle use circumference 144 pixels, radius `144/(2π)≈22.92`; for the line use six equally spaced centers spanning 120 pixels, with 12-pixel margins defining the same 144-pixel allocated path. Rotate a whole mark's local patch with its local ring frame; place it without slicing it into angular wedges. Center both layouts. Scaffold stroke width and total path length are equal. The line/ring shape is known and constant, carrying no meaning. This deliberately changes v2's appearance to isolate geometry. A later full-library experiment can restore the original aesthetics after this control works.

Training channel: uniformly sampled global rotation in `[−π,π)`, bilinear interpolation with the same padding/background, no crop, and **no additional noise or occlusion** in the primary run. This isolates rotation and layout before combining disturbances. Keep the full mark bounding support inside the canvas for every rotation. The chosen canvas is larger than v2's current training canvas to avoid granting the circle a shorter path or smaller glyphs merely to fit an upright line.

Final evaluation conditions, shared random streams across paired layouts:

1. Clean upright, no channel noise.
2. Clean rotation: angles `2πk/24`, `k=0,…,23`, for every meaning.
3. Rotation plus zero-mean Gaussian image noise with standard deviations **.05 and .15** on the `[0,1]` ink scale, then clipping; four independent draws per angle, noise seeds **1000–1003**.
4. Whole-mark erasure of exactly **one** or **two** of six slots, averaging all 6 or 15 masks; also reveal curves retaining 0–6 slots, contrasting all contiguous cyclic masks with equally sized scattered masks. For the line, “contiguous” uses its actual ordered path, not an invented wraparound edge. Report the mask counts and retained ink; leave the fixed scaffold visible in both conditions.

Conditions 3–4 are secondary robustness diagnostics. Their numeric disturbance levels are declared test conditions, not literature-derived perceptual thresholds. Never label an evaluation “clean” while inheriting v1's default noise/occlusion.

### 5.4 Agents, optimization and budget

Use the current five role-specific input embeddings (32 dimensions each), concatenated through two 512-unit hidden layers, but output **6×7 energies** `u_j(a|x)`. For each permitted word `w∈C`, score

`S(w|x)=Σ_{j=1}^6 u_j(w_j|x)`.

Normalize these scores over C and sample one complete hard codeword during training. This shares the exact sender architecture and codebook between layouts, avoids a separate learned lookup entry for every meaning, and makes the capacity constraint explicit. The restriction can itself bias learnability; document it and keep the unrestricted-line study separate. At evaluation choose the maximum-score word, breaking ties by codebook order.

Use **REINFORCE for the hard categorical choice** plus ordinary receiver backpropagation, following the estimator family in C19/C20/R22. The categorical action is a whole codeword; its log probability is `log softmax(S)[w]`, not a sum of independently normalized slot probabilities after imposing C. Use a running-mean baseline, detached advantage, and sender entropy coefficient **.01**. This is an RF design with a potentially difficult 19,684-way action space; the convergence gate below is essential. Do not assume a short compute cap guarantees success.

Receiver: the same four-block CNN family as `v1/heptapod.py:210`, with two 3×3 convolutions per block, channels 32/64/128/128, BatchNorm, GELU and max pooling, adaptive 3×3 pooling, 384-unit trunk and five output heads. Both layouts get the same architecture and parameter count. The receiver sees only the image. Keep BatchNorm identically in both; R22's “no layer norm” result is not justification for silently changing this CNN.

Loss: mean of the five ordinary attribute cross-entropies; no dynamic difficulty weights, ink target or mark-count cost in this first fixed-rate experiment. Report sum and mean NLL with explicit units. Adam **5×10⁻⁴**, β=(.9,.999), batch **128**, **8,000 joint updates**, no LR schedule and no weight decay in the primary baseline. Learning rate and entropy coefficient follow R22's scale; batch and update cap are computational choices, not a reproduction of its 1,024-example batches. Sample uniformly from training meanings, with replacement, and give the paired layouts the same meaning and angle streams. Save independent RNG streams for training, validation and probes.

Evaluate greedy development NLL on all 216 development meanings at the 24 fixed angles every **250 updates**; choose the earliest checkpoint attaining the lowest development NLL. Do not examine test scores during training. Also retain the final checkpoint to show whether selection matters. Count all failed runs in unconditional summaries. A source-inspired secondary summary may condition on **>99.9% clean training exact accuracy**, following C20, but must show the denominator and may not replace the unconditional result. With 864 meanings, that threshold requires perfect greedy clean training reconstruction.

Use **ten paired training seeds, 0–9**, matching C20's initialization count rather than the earlier review's three. Initialize corresponding ring/line parameters identically and use separate reproducible channel/action streams. This is **20 joint-training runs**, not a claim of sufficient statistical power for every possible effect. Benchmark one development-only run before launching the set; if the budget must change, change and record it for all conditions before inspecting test performance. Do not extend only losing seeds or one layout.

### 5.5 Baselines that make failure interpretable

**Engineered compositional code:** encode each meaning as the cyclic orbit of

`[6, s, v, o, t, p]`.

Symbol 6 is a unique anchor; the remaining IDs are 0–5. The anchor costs one of six slots and one of seven symbols. Blank 0 is a legitimate value-bearing absence at a known position. Canonicalize into C for both layouts. This yields an injective code for all 1,296 meanings. Train the same image listener on training meanings only, under the same image channel, optimizer and 8,000-update budget, using seeds 0–9. It checks whether the image receiver can acquire and generalize a deliberately compositional protocol under this split. It is not evidence that the learned sender discovered it.

**Random injective codebook:** for each seed 0–9, use a seeded uniform permutation of C and assign its first 1,296 entries to the lexicographically ordered meanings. Freeze this mapping, then give the listener only training examples. The generation of a complete random mapping is not exposure to held-out labels: no held-out pair is used in fitting or selection. Its purpose is to test a stable, informative but unstructured convention. Use the same mapping for ring and line and the same listener budget. Match its subset of C to the engineered/learned code's costs only in a separately labeled sensitivity analysis; post-hoc matching must not replace the primary baseline.

**No-image and shuffled-message controls:** train/evaluate a listener with constant images under the same split, and permute messages among test meanings at evaluation, repeating 100 fixed permutations. Report empirical controls, not only a guessed chance value. Under the full uniform meaning prior, uniform exact guessing is 1/1,296, **0.07716%**, and uniform per-head guessing has mean attribute accuracy `(1/6+1/6+1/6+1/3+1/2)/5 = 26.6667%`. On the constrained test support, a split-informed constant joint guess can attain **1/216**, **0.4630%**; marginally uninformative predictions need not have that joint rate. A model given some roles can also exploit the deterministic test subject–object relation. Report full reconstruction, role errors and the role-reversal contrasts to expose such shortcuts.

**Symbolic/orbit diagnostic:** directly inspect whether distinct test meanings collide in hard cyclic code space. For the engineered code, an exact symbolic parser provides a 100% reference without learning or rasterization. This is a channel-integrity check. It must not be described as the visual listener's accuracy. A lookup decoder trained only on observed messages is a separate baseline and cannot assign meanings to unseen arbitrary strings by magic.

These controls add inexpensive receiver-only fits; the “20 runs” count above refers only to joint sender–listener training. Report total optimizer updates and wall time across **all** controls and probes, not just the headline runs.

### 5.6 Fresh-listener test and diagnosis of overfitting

For each selected learned speaker, freeze the sender and all rendering choices. Train **three fresh CNN receivers**, initialization seeds **17, 18, 19**, on the **864 training meanings only**, with the same channel and optimizer, for **2,000 updates**, development selection every 100. The 2,000-update cap is a resource choice; show the curve, its endpoint and whether the positive code has converged within that budget. Report every probe and average within each training seed before comparing layouts. Do not count 30 nested probes as 30 independent learned languages.

Record exact and attribute accuracy, NLL, and normalized training-accuracy AUC over the common 0–2,000-update interval, using measurements every 100 updates and trapezoidal integration divided by 2,000. Report development/test learning curves descriptively only at the frozen analysis stage; test values do not choose the checkpoint. C20 motivates the acquisition comparison; R22 motivates evaluating a new listener instead of assuming the original partner is optimal.

A smaller receiver (same structure with channels **16/32/64/64**, trunk **192**) is a prespecified follow-up if the main acquisition test succeeds. It tests architecture transfer, like C20's different receiver capacities, but does not need to run before establishing that the original-sized fresh receiver can learn anything.

For a distinct-example acquisition curve, use nested training sets of **72, 216, 432, 864 meanings**, with all six grammatical variants per chosen SVO triple. Start with 12 triples `{(s,s,s), (s,(s+1) mod 6,(s+3) mod 6): s=0,…,5}`; these are training triples and cover every subject, verb and object value. Shuffle the remaining 132 training triples with `random.Random(17)` and append enough to reach 36, 72 and 144 triples. Keep the optimizer-update budget fixed across subset sizes and report both unique examples and total presentations. This optional curve extends the primary full-training-set probe; it must not silently replace it if a subset is favorable.

Interpret the comparisons:

- Original receiver poor, fresh receiver good: evidence of partner/acquisition overfitting or optimization failure; the message may support a better decoder.
- Both poor, engineered control good: the learned protocol does not support the demonstrated transfer under this learner/budget, even though the visual task is learnable.
- Both good, local semantic interventions fail: good systematic/acquirable communication, but no evidence yet for the claimed local whole-mark composition.
- Engineered control poor: the experiment is not calibrated to falsify the sender's compositionality; investigate perception, optimizer, split/interface or budget using training/development data.

If overfitting is present, the first prespecified remedy extension is **listener-only Adam weight decay .01**, leaving speaker decay zero, paired over the same ten seeds and both layouts. This copies the direction and scale of R22's tested regularizer, not a guaranteed effect size. A separate reset extension should implement the actual reset-after-each-speaker-update inner loop, with development early stopping and explicit total receiver-update accounting. An occasional reset every few epochs is a different intervention. A listener population is another separately budgeted intervention; R22 does not provide a population effect estimate.

True iterated learning would require a generation to learn from a restricted corpus produced by a previous generation and then become a producer for the next one. Fresh-receiver evaluation, joint multi-partner training and receiver resets are three different mechanisms; none alone implements that transmission process. There is no complete, verified generational implementation among the audited EGG paths that should be imported as-is.

### 5.7 Measurements and analysis fixed in advance

Primary performance measurements:

1. Original-listener exact accuracy on the 216 test meanings averaged across 24 clean rotations, with the paired **ring minus line** contrast as the geometry endpoint.
2. Fresh-listener test exact accuracy under the same rotations, averaged over probe seeds within each training seed; compare learned versus random-code controls with the same probe budget. Train corresponding random-code and engineered-code probe controls for the same 2,000 updates; do not compare a 2,000-update learner to an 8,000-update baseline.
3. Held-out semantic intervention success: expected target attribute changed and **all other attributes preserved**, compared with random substitutions.

Also report clean upright exact accuracy; per-attribute scores; reversal errors; NLL; hard-code collision ceilings; occupied-mark counts/ink; `H([M])`; cyclic-Hamming topographic similarity; bosdis with blank included; and any justified anchored posdis. Compute exact all-pairs metrics within train, development and test (the full space has 839,160 unordered pairs, but those are not independent observations). Run **1,000 meaning-label permutations**, seeded 2000–2999, for metric null distributions. Sampled-policy analyses use 20 messages per meaning from a frozen checkpoint and are labeled secondary; the primary is greedy.

For semantic interventions, derive candidate rules using training-only minimal pairs differing in one attribute. Search for a single whole-symbol replacement after a fixed label-independent alignment rule; record the source symbol, replacement symbol, relative/anchored location, and predicted attribute change. Rank by training support, breaking ties lexicographically, and freeze the ten most supported rules (or all if fewer). Apply them to every applicable held-out source message; canonicalize the changed codeword into C before rendering. The alignment rule cannot use a held-out target meaning to choose a favorable rotation. Score exact agreement with the predicted modified meaning, including preservation of all unchanged attributes, and compare with frequency-matched random replacements at the same location. Report applicability/coverage and each rule separately; a rule with no held-out applications is untested. If there are no single-mark rules, report that outcome rather than inventing one: it argues against **local single-mark** composition but does not exclude a distributed compositional representation.

Statistical units and uncertainty:

- Average augmentation draws and probe initializations **within** each training seed. Ten independently trained protocols, not pixels, distances or repeated renders, are the main units for learning-algorithm comparisons.
- Report paired seed differences and 95% intervals from a paired bootstrap over seeds, **10,000 resamples**, analysis seed **3000**. With ten seeds these intervals have limited precision; show individual points.
- Use the exact **2¹⁰=1,024 paired sign-flip assignments** for the three prespecified directional contrasts (geometry, fresh-listener advantage over random code, semantic-intervention advantage over random substitution). State the exchangeability/symmetry assumption of this paired randomization-style test; it is not guaranteed merely by using the same seed numbers. Apply Holm correction across the three primary tests at familywise α=.05. Assign an untestable primary contrast (for example no applicable substitution rules) no positive evidence, retaining it in the family as p=1. Report unadjusted estimates/intervals alongside adjusted p-values rather than disguising an unadjusted interval as multiplicity-adjusted.
- Use a separate paired interaction contrast, `(ring−line)_rotated − (ring−line)_upright`, descriptively to distinguish a rotation-specific benefit from an overall geometry difference. Noise/reveal curves, metric correlations, smaller receivers and alternative splits are secondary unless a new experiment is preregistered.
- The 95%/α=.05 convention controls statistical error; it is not a literature-derived language threshold. Ten seeds do not guarantee power, and non-significance is not equivalence. If a practical noninferiority margin is desired, the user must define a meaningful error/compute tolerance; these papers do not supply one for ink scripts.

### 5.8 Pass, fail, and inconclusive outcomes

Use an evidence ledger with separate claims, not a single binary “real language” verdict.

| Claim | Positive evidence required | Falsification / limit |
|---|---|---|
| The implementation can communicate | Clean training exact reconstruction reaches the C20-inspired convergence gate, with no hidden side channel and distinguishable rendered controls | Collision of different intended meanings in the actual received representation is a direct obstruction to perfect reconstruction. Failure of all controls is a setup failure, not a language result. |
| It systematically generalizes under this split | Held-out exact performance and fresh acquisition exceed the random/constant controls with a positive paired effect supported by the prespecified uncertainty analysis; role contrasts are reported | If the upper confidence bound for the relevant learned-minus-control contrast is ≤0 while the engineered control succeeds, the claimed advantage over that control is rejected under this budget. A wide interval crossing zero is **inconclusive**, not a demonstrated failure or pass. |
| It uses reusable local whole-mark semantics | Training-derived substitutions predict the intended held-out semantic change while preserving the other attributes, outperforming matched random substitutions; coverage is disclosed | Repeated counterexamples refute a claimed universal substitution rule. No eligible/applicable rules means this local hypothesis has not been supported; low posdis alone does not refute every form of composition. |
| Circular rendering improves rotation-robust decoding | Positive ring-minus-line exact-accuracy effect, 95% paired interval above zero and Holm-adjusted directional test passing | A 95% interval entirely below zero supports a linear advantage and rejects circular superiority here. An interval including zero does not establish an advantage, equivalence or noninferiority. Both layouts at ceiling require a harder prespecified follow-up, not a claim that the ring won. |
| The protocol is transferable | Fresh-listener curves show a benefit over arbitrary codes at the same data/update budget; original and fresh performance are both disclosed | Success of the original partner alone is insufficient. Poor fresh transfer with a good engineered control rejects the practical transfer claim for this learner/budget, but does not prove no imaginable decoder can learn the code. |

A single erroneous deterministic minimal pair can falsify an **explicit universal rule** such as “this mark always negates without changing any other attribute.” Statistical aggregate success supports a narrower approximate claim. Do not convert “above random” into “fully compositional”; report the gap to the engineered code and the actual errors. The paper-based rationale is comparative and mechanistic: C20 supplies train-convergence and new-listener precedents, R22 supplies overfitting controls, S20 supplies intervention logic, and C19 supplies capacity/cost accounting. The papers do not justify the old plan's >50% meaning-piece or >90% photo thresholds.

Stop the expansion if the engineered visual code cannot be learned on development data, if the learned code cannot fit training data within the matched budget, or if a capacity/renderer bug is discovered. Preserve failed seeds. Any changed optimizer, codebook, split or compute cap starts a clearly versioned experiment; it must not retroactively erase the negative attempt. Only after the finite protocol survives these checks should the project add recursive meanings, COGS logical forms, larger alphabets, population training or transmission generations.

## 6. Deliverables for the eventual experiment

Before executing the design, create a run manifest containing the code commit, all parameters, mark-library and selected-template hashes, canonical codebook hash, exact split lists, initialization/channel/analysis seeds, device/runtime versions and the full checkpoint-selection rule. Save optimizer, baseline and RNG state along with models. Keep a compact frozen-code export mapping every evaluated meaning to its hard cyclic message, plus received-image condition and predictions. This makes capacity, mode errors, collisions and metric calculations independently inspectable.

The final results table should have one row per seed and condition, showing train/development/test exact and attribute accuracy, original/fresh receiver, clean/rotated/noisy channel, update count, convergence status, topographic distance convention, bosdis/posdis applicability, empirical entropy and collision ceiling. Show paired ring–line effects and acquisition curves. Keep theoretical capacity, empirical code usage, physical robustness and semantic generalization in separate columns.

**Reading conclusion:** the existing results establish promising finite-task communication, with unresolved evaluation/provenance issues. The cited literature supports a more discriminating experiment, not a shortcut from attractive ring images or one correlation to a language claim. The strongest immediate improvements are hard cyclic channel accounting, a balanced semantic holdout, independent-listener probes, causal whole-mark interventions, and a genuinely matched linear rendering.
