# Drawing games: deep-reading notes

Sources read in full: Mihai & Hare 2021a (*Learning to Draw*, NeurIPS 2021, arXiv 2106.02067v2), Mihai & Hare 2021b (*Shared Visual Representations of Drawing for Communication*, SVRHM workshop, arXiv 2110.08203v2), Qiu et al. 2022 (*Emergent Graphical Conventions in a Visual Communication Game*, NeurIPS 2022, arXiv 2111.14210v4), Hawkins, Sano, Goodman & Fan 2021 (*Visual resemblance and communicative context constrain the emergence of graphical conventions*, arXiv 2109.13861v1). Code read: `research/repos/LearningToDraw` (all Python files), the `dsketch` rasterizer it depends on (fetched read-only from `jonhare/DifferentiableSketching` at commit `8778b7a`, 2023-08-09, into the session scratchpad; not vendored), and the analysis/experiment code of `research/repos/graphical_conventions`.

Convention in this file: **[S]** = what the source shows; **[I]** = my interpretation or inference. Paper references are section/table/figure; code references are `file:line`. Paths for LearningToDraw are relative to `research/repos/LearningToDraw/`, for dsketch relative to the `DifferentiableSketching` repo root, for Hawkins relative to `research/repos/graphical_conventions/`.

---

## 1. Mihai & Hare 2021a, *Learning to Draw: Emergent Communication through Sketching*

### 1.1 Setup [S]

- **Game.** Image referential game derived from Havrylov & Titov (Sec 3.1). Sender sees photo P_s and draws; receiver sees the sketch plus K distractors and a target P_r, and must pick P_r. Three variants:
  - *Original*: P_r = P_s, distractors random from the pool, **K = 99** (Sec 3.1).
  - *OO-game same*: P_r = P_s, all receiver photos from different classes.
  - *OO-game different*: P_r is a **different photo of the same class** as P_s; all receiver photos from different classes. STL-10 has 10 classes, so OO games have K = 9 (Sec 4.1).
- **What the sender draws with.** 20 black, constant-width straight lines on a white canvas the size of the input (Sec 3.2; 5–50 lines in App A). Each line = 4 numbers (start, end) in [-1, 1] from a tanh. Rendering is the authors' **differentiable rasteriser** (Mihai & Hare, arXiv 2103.16194): per-line squared Euclidean distance transform D²_seg, raster R_i = exp(-D²/σ²) with **σ² = 5e-4** (Eq. 1), composed by an inverted soft-or S = ∏(1 − R_i) (Eq. 2), greyscale replicated to 3 channels. Fully differentiable, trained end to end with backprop (no RL).
- **Architectures** (Sec 3.2, Fig. 1).
  - Shared "early visual system": **VGG16 up to ReLU5_3** (before the last max-pool and FC layers), **torchvision ImageNet weights, frozen**, inputs ImageNet-normalised (sketches too). Feature map flattened and linearly projected to 64-d.
  - Sender: 64-d code → 3-layer MLP (64, 256, then 4·n_lines, ReLU, ReLU, tanh) → line coordinates → rasteriser.
  - Receiver: same backbone → 2-layer MLP (64 → 64, ReLU on the first) for both sketch and each photo; scores = dot product sketch·photo; argmax is the guess.
- **Losses.** Game loss: Weston–Watkins multi-class hinge (multi-margin) loss on the score vector (Eq. 3); cross-entropy reportedly also works. Optional **perceptual loss** (Sec 3.4, Fig. 2, Eq. 4): VGG16 feature maps at relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 for sketch S and the sender's photo P, each unit-normalised across channels, squared difference summed over channels, spatially averaged, weighted average over layers with w_l (default all 1). Equivalent to the content loss of neural style transfer; unlike LPIPS no per-channel learned weights.
- **Channel noise.** None. The channel is the clean rendered sketch. (There is an optional random affine + intensity jitter on the *sender's input photo* only, `--random-transform-sender`, not on the sketch.)
- **Data.** STL-10 at 96×96 (main), Caltech-101 at 128×128 (Secs 4.4, 4.5), CelebA centre-cropped to 112 px (Sec 4.6); also MNIST/CIFAR/TinyImageNet supported in code.
- **Training.** Adam, lr 1e-4, batch = K+1 (100), **250 epochs**; batches of games share one receiver pool, targets shuffled each batch (Sec 3.3). Hardware: mix of GTX1080, RTX2080, Quadro RTX8000, RTX-Titan. Throughput ~488 games/s (~10 s/epoch) on STL-10, ~175 games/s (~5 min/epoch) for Caltech-101 at 128 px (Sec 3.3). All numbers averaged over **10 seeds**; sample sketches from one seed (Sec 4).

### 1.2 Key results [S]

| Where | Result |
| --- | --- |
| Table 1 (STL-10, 20 lines) | Comm. rate, l_game vs l_game + l_perceptual: original **71.8 % (±6.1) vs 69.57 % (±2.6)**; OO-same 95.46 vs 96.04; OO-diff 82.72 vs 81.09. Perceptual loss "significantly improves human interpretability … at almost no cost" (Sec 4.1). Animals/birds stay poorly drawn even with it. |
| Table 2 | Perceptual loss on a single VGG block: blocks 1–2 give unrecognisable drawings; blocks 3–5 give increasing structure. Original game rates 68.4 / 69.6 / 71.1 / 76.4 / 60.5 %. "Later layers contribute the most" to human-understandable protocols (Sec 4.2). |
| Sec 4.3, App F | OO-different pushes sketches toward class-level "symbols"; original game keeps instance variation. |
| Table 3, App G | Wide model (1024-d) on Caltech-101 OO-diff: 50.46 → 64.99 %; wider model draws instance-specific detail again. |
| Table 4, App H | Stylized-ImageNet (shape-biased) VGG weights: 77.09 % vs 78.46 % (ImageNet) on Caltech-101 original game, but drawings more faithful to object shape. |
| Fig. 3, App I | CelebA: 98.9 % (ImageNet) / 99.6 % (SIN) comm. rate; SIN sketches capture pose, hair, glasses, headwear. |
| App A, Tables I–II | Fewer lines do not hurt: 5 lines 73.41 %, 20 lines 69.57 %, 50 lines 69.21 % (original). More lines look more like the object to humans; the agents do not need that. |
| App B, Table III | Point clouds instead of lines: 10 points → 75 % but meaningless; 50/100 points → 71/66 % with contours emerging. |
| App C, Table IV | Restricting the perceptual loss to fewer layers (applied to both agents): relu4_3 → **19.09 %**, relu3_3 → **1.0 %** (no convergence). Adding more layers (Table V) changes nothing. |
| App D, Fig. I | **Arbitrary perceptual target**: forcing all sketches to look like one dog image while playing on CelebA. λ = 1 → 69.2 %, λ = 0.01 → 98.9 %. "Humans stand no chance" of understanding them. |
| App E, Figs II–III | Out-of-distribution: CelebA-trained agents on STL-10: 15.8 % (ImageNet weights) vs 30 % (SIN). STL-10-trained on Caltech-101: 22.2 % (l_game) vs 26.7 % (+ l_perceptual). |
| App J, Fig. VIII | Overlay of 10 seeds: seeds agree on key strokes, differ in detail. |

**Human evaluation (Sec 4.7, Table 5, App K).**
- Protocol: pilot; a trained sender (single fixed seed per config) paired with a human receiver through a web UI. Human sees a sketch and a grid of 10 photos (**K = 9**, chance 10 %; agents were trained with K = 99 in the original game), clicks one. 30 games per config × 5 configs = 150 games per participant. No time limit; response time logged. Games sampled at random from the STL-10 test set. Two disjoint groups of **6 participants** each (ages 20–35): no feedback (Table 5) vs feedback after each trial (correct target shown, Table IX). Also reports *class communication rate* (picked an image of the right class). Significance by t-tests (App K.3).
- Results without feedback (Table 5): original + l_game, 20 lines: agent 100 %, **human 8.3 % (±5.4)**, class 15.0 %. Original + perceptual, 20 lines: agent 93.3 %, **human 38.3 % (±2.5)**, class 55.6 %. 50 lines: 37.2 %. OO-diff + perceptual: 23.9 % (20 lines), 38.9 % (50 lines). Perceptual vs no perceptual: p < 0.0001.
- With feedback (Table IX): l_game-only rises 8.3 → **13.9 % (±3.0)** (p < 0.0001); the other configs do not improve significantly. Authors note 30 games may be too few for humans to learn the agent's strategy (App K.2). Humans spent 1:00–2:30 per sketch in the l_game-only condition.

### 1.3 Iconic vs symbolic [S]
No formal measure. Iconicity is judged qualitatively (Tables 1–4) and by human gameplay. The OO-different game is described as producing more "symbol-like" class depictions (Sec 4.3). Without the perceptual loss, Mihai & Hare 2021b describe the result as "the visual equivalent of a hash code".

### 1.4 Stated limitations [S]
- Communication success alone does not yield human-decipherable sketches "irrespective of any visual system preconditioning" (Sec 1, Table 1).
- Animals and birds remain poorly drawn (Sec 4.1).
- Open question whether interpretability comes from ImageNet *labels* baked into the backbone; plan to test self-supervised backbones (Barlow Twins) and independent visual systems per agent (Sec 5).
- Fixed number of straight strokes; learned perceptual weights are tricky because the network could learn zero weights (Sec 3.4, Sec 5).
- Human study is a pilot (6 + 6 people, one seed, 30 games per setting), K = 9 not 99 (App K).

---

## 2. Mihai & Hare 2021b, *Shared Visual Representations of Drawing for Communication*

### 2.1 Setup [S]
- Same game, original variant only, K = 99, 20 lines, STL-10 (Sec 3).
- Two frozen backbones, **shared between sender and receiver** (App A): ImageNet VGG16 (as in 2021a; 96 px; MLPs 64/256 → 80; receiver 64/64; lr 1e-4) and **CLIP ViT-B/32** (224 px; sender MLP hidden sizes **1024/1024** because 64/256 "would not work well"; **lr 1e-3**, which makes VGG diverge).
- Two perceptual losses: (a) l_perceptual, layer-wise normalised squared feature differences, here after every transformer residual block for ViT, uniform weights (Eq. 1); (b) **l_clipdraw**, CLIPDraw-style negative cosine similarity between the final embedding of the photo and of 4 randomly augmented copies of the sketch (random perspective + random resized crop) (Eq. 2). The augmentation is what makes the cosine loss produce perceptually similar drawings.

### 2.2 Key results [S]
- **Fig. 2** (agent comm. rate, K = 99): ViT l_game 69.8 %, +l_perceptual 63.4 %, +l_clipdraw 61.1 %; VGG l_game 75.7 %, +l_perceptual 72.1 %, +l_clipdraw 51.8 %. Perceptual losses cost agent accuracy.
- **Table 1** (humans, same pilot protocol: 6 participants, 30 games per model, K = 9, chance 10 %): VGG l_game 8.3 % / class 15.0 %; VGG +perceptual 38.3 / 55.6; VGG +clipdraw 34.0 / 49.3; ViT l_game **5.6 %** / 15.6; ViT +perceptual 45.3 / 63.3; **ViT +clipdraw 62.7 % (±11.6) / class 83.3 % (±9.2)**.
- **Table 2, CLIP prompt probe** over all 8,000 STL-10 test games: prompts "a drawing of a {class}." and "a photo of a {class}."; c(I) = class of the nearest prompt. c(sketch) == ground truth: VGG l_game 7.3 %, VGG +perceptual 24.0 %, VGG +clipdraw 41.2 %, ViT l_game 9.4 %, **ViT +perceptual 96.4 %, ViT +clipdraw 96.6 %**. CLIP labels sketches as drawings 96–100 % of the time. The authors flag the bias: the probe is the same CLIP model used to produce the ViT sketches (Sec 4).
- Class identity is not needed for the self-supervised game; object-centric drawings are a side effect of the representational loss (Sec 4).

### 2.3 Stated limitations [S]
Perceptual losses reduce agent communication; the probe is biased toward CLIP-trained models; agents succeed far above the 10 % that class-only information would allow, so the sketches carry extra information whose nature (semantic content vs "neural hash code") is unknown; proposed causal interventions on stroke bundles (Sec 5).

---

## 3. Qiu et al. 2022, *Emergent Graphical Conventions in a Visual Communication Game*

### 3.1 Setup [S]
- **Game** (Sec 3, Fig. 2): multi-step, category-level referential game. Sender sees image I_S; receiver sees M context images that do **not** include I_S but include one target I*_R of the same class. Shared canvas starts blank. Each step the sender adds **5 strokes**; the receiver then either picks an image or **waits**. Shared reward ±γ^t (γ = 0.85) on termination, nothing for waiting: an implicit cost on length. Max T = 7 steps.
  - Paper says **M = 3** (Sec 5.1) but quotes chance as **25 %** (Sec 5.2.2), and the official README trains with `--game_size=4`. [I] Most likely 4 options (target + 3 distractors); the text is internally inconsistent.
- **Sender** (Sec 4.1): two stages. f_S = pretrained photo→pixel-sketch model (Kampelmühler & Pinz 2020), used as is. g_S = stroke policy built on Huang et al. 2019 (*Learning to Paint*), conditioned on target sketch + current canvas, outputs 5 quadratic-curve strokes (x0, y0, x1, y1, x2, y2) ∈ (0,1)^6. Gaussian policy N(μ, c·I) during training, c = 0 at test. Strokes go through a **pretrained neural renderer G** (Huang et al.), fully differentiable. g_S pretrained on **Sketchy**.
- **Receiver** (Sec 4.2): pretrained **VGG16 → 4096-d**, two separate linear heads (key/query); Boltzmann policy over dot products between the current canvas and the M images **plus the previous canvas C_{t−1}** as a "wait" option (Eq. 3): waiting means the new strokes did not move the canvas closer to any option. Whether the VGG is frozen is not stated in the text.
- **Learning** (Sec 4.3, App C): a model-based multi-agent policy-gradient surrogate. V(X_t) is differentiated through the renderer for the sender (reparametrisation point estimate) and analytically over the categorical receiver; value baseline v_φ regressed onto an episodic eligibility-trace target V_λ (Eqs. 5–7, Alg. 1). REINFORCE baseline used for ablation.
- **Losses**: RL return only. No perceptual loss; iconicity comes from the pretrained sketch pipeline.
- **Channel noise**: none, apart from the sender's Gaussian exploration noise during training.
- **Data**: Sketchy, **40 categories** picked where the sketcher behaves well (App A); 30 training categories (10 images each for training; 10 more per category held out as "unseen instance"); 10 unseen categories. Test set: 300 + 100 targets.
- **Budget**: batch 64, Adam lr 1e-4, **30k iterations**, 2k-iteration warm-up for the sender, exponential decay; **one RTX A6000, ~20 h per run**, **5 seeds** (Sec 5.1).
- **Controlled settings** (Table 1): *complete* (early decision + sender updates + multi-step); *max-step* (no early decision); *sender-fixed* (frozen sender); *one-step* (a single step); *retrieve* (no early decision, no sender update: unevolved sketches, an iconicity upper bound).

### 3.2 Key results [S]
- **Table 1** (accuracy % ± SD, avg. steps): complete seen 98.07 (1.03 steps), unseen instance 70.37 (2.36), unseen class 39.40 (3.76). max-step 86.27 / 67.93 / 38.40 (always 7). sender-fixed 99.60 / 71.80 / 45.40. **one-step 22.87 / 14.07 / 9.60**. retrieve 99.47 / 76.80 / 48.00 (7 steps).
- **Fig. 3b, Sec 5.2.1**: "systematic reduction": average steps first rise, then fall (complete and sender-fixed). complete ends with simpler drawings than sender-fixed, credited to mutual adaptation.
- **Fig. 3c**: accuracy when receivers see sketches cut at 1/3/5/7 steps; in *complete*, 1-step accuracy catches up with 3-step but never exceeds it, read as the pair moving along the accuracy/complexity Pareto front. With REINFORCE, 1-step overtakes 3-step; REINFORCE also converges more slowly and is unstable at γ = 0.95 (Fig. 3a–b).
- **Fig. 4** (symbolicity; bar values read off the plot, approximate): images ~76 %, **complete ~79 %**, max-step ~55 %, sender-fixed ~54 %, one-step ~43 %, retrieve ~56 %.
- **Table 2** (semanticity correlation): complete 0.43 ± 0.02, max-step 0.41 ± 0.13, sender-fixed 0.36 ± 0.06, one-step 0.31 ± 0.08, **retrieve 0.55 ± 0.00**. The text says semanticity "can be better retained in the complete setting compared with the retrieve baseline", which the table contradicts (retrieve is higher). [I] The defensible reading: among the *evolved* settings, complete keeps the most semanticity; evolving costs some semanticity relative to unevolved sketches.
- **Fig. 5, App B**: t-SNE of fine-tuned VGG features. Complete sketches form cleaner class clusters than raw images, with semantic neighbours kept close (cow/deer/horse/camel vs burger/apple). Other settings show no clear boundaries.
- **Fig. 6, Apps D (Figs A2–A4)**: evolution toward salient parts (rabbit ears, giraffe neck, deer antlers, three vertical lines for horse).
- Unseen classes: complete and sender-fixed agents use **longer episodes** (3.76, 4.75 steps); the authors read this as falling back to iconic communication for unconventionalised referents (Sec 5.2.2).

### 3.3 Definitions of iconicity, symbolicity and semanticity [S] (Sec 1, Secs 5.2.2–5.2.4)
- **Iconicity**: "the drawings exhibiting a high visual resemblance with the corresponding images, such that they are proximal to the latter when measured on the high-level embedding of a general-purpose visual system." *Measurement*: because the receiver's policy is a softmax over sketch–image similarities against a context of distractors, **communication accuracy on unseen images** (unseen instances and unseen classes) is used as a normalised resemblance measure. Higher generalisation means more iconicity; retrieve is the upper bound.
- **Symbolicity**: "the drawings being consistently separable in the high-level visual embedding, which facilitates new communication partners to easily distinguish between categories without grounding them to referents." *Measurement*: take the 300 final canvases (10 per training category); a pretrained VGG16 acts as a **new learner**, only its last FC layer is fine-tuned to classify into the 30 categories; **70/30 train/test split**; test accuracy = symbolicity (Fig. 4).
- **Semanticity**: "the topography of the high-level visual embedding space of drawings being strongly correlated to that of images, such that semantically similar instances and categories lie close to each other." *Measurement*: featureA = word2vec of category names; featureB = embeddings of evolved sketches from the fine-tuned VGG of the symbolicity test; **correlation between all pairwise distances** in A and in B (Table 2). [I] This is topographic similarity with a word-embedding metric on the meaning side and a probe network on the signal side.
- The authors stress that these are one way to operationalise the concepts (Sec 1).

### 3.4 Stated limitations [S]
- The two-stage pretrained sender is a limitation: an ideal sender would not first turn images into pixel-level sketches (Sec 6).
- Limits of the pretrained sketching module restrict the categories (40 chosen for satisfactory sketching) and therefore the discrimination demands (Sec 5.1, Sec 6).
- High communication accuracy does not imply a good convention (Sec 1, Sec 5.2.1, citing Bouchacourt & Baroni).
- The definitions are not the only possible ones (Sec 1).

### 3.5 Official code [S]
- Located on the project page (https://sites.google.com/view/emergent-graphical-conventions, "[code]" link): **https://github.com/Janetalready/VisualCommunication**. arXiv abstract page has no code link; the article itself is CC BY 4.0.
- **No license file** (GitHub API reports `license: null`), so it is all rights reserved by default. Not cloned.
- README (read via API): torch==1.11.0, opencv, wandb; data and pretrained models on a Google Drive folder; the sender is pretrained with megvii's ICCV2019-LearningToPaint "with the colour output reset to all ones"; pixel sketches come from kampelmuehler/synthesizing_human_like_sketches; entry points `train_dreamer_add_class_gen.py` (complete/max/sender-fixed/one-step) and `train_original_reward_retrieve_5stroke_accum.py` (retrieve, cumulative-reward baseline); `--n_games=50000`, `--game_size=4`, `--num_stroke=5`, `--discount=0.85`. Last push 2022-10-12.

---

## 4. Hawkins, Sano, Goodman & Fan 2021 (human study)

### 4.1 Setup [S]
- **Task** (Sec 2, Fig. 2, Methods 4.1): dyads on MTurk (Hawkins 2015 web framework). Shared context = 4 objects drawn from 8 visually similar ShapeNet chairs (two clusters, "dining" and "waiting-room", found with **VGG-19 fc6 + PCA + k-means**, 70 clusters, over 3,096 chairs). Sketcher draws the target with the mouse (300×300 canvas, 5 px black pen, **no deletion**, 30 s limit); strokes appear live for the viewer, who can **answer at any time** (Pictionary-style); both get immediate feedback. Bonuses for accuracy and speed.
- **Design**: 4 *repeated* objects drawn in 6 repetition blocks; 4 *control* objects drawn once in a pre phase and once in a post phase (repeated objects also appear in pre and post): 8 + 24 + 8 = 40 trials.
- **Samples**: 138 participants (69 dyads); plus 130 (65 dyads) as an internal replication in which the viewer must wait for a "Done" button (removes the viewer-interruption confound) and contexts are fixed for the diagnosticity analysis. Recognition experiment: 245 recruited, **223 kept** (106 yoked, 117 shuffled) after attention checks (≥ 4/5 catch trials correct); replication 225 (100 / 125). Annotation: **443** annotators for drawing–object part correspondences (≥ 1 high-quality map for all 2,600 drawings), **117** for diagnosticity maps (≥ 30 ratings per object pair).

### 4.2 Measures and key results [S]
- **Efficiency** = balanced integration score (BIS, Liesefeld & Janczyk 2019): z-score accuracy and response time within each interaction, BIS = z(accuracy) − z(RT) (Sec 2.1). Improves with repetition, b = 0.5, t = 13.5 (Fig. 3); RT alone b = −1.5, t = −11.5; accuracy alone b = 0.46, z = 6.5. First-time accuracy **76 %** (chance 25 %).
- **Strokes** fall with repetition, b = −0.216, t = −6.00 (Fig. 4A).
- **Object-specificity**: pre-phase accuracy equal (75.7 % repeated vs 76.1 % control); phase × condition interaction on efficiency b = −0.16, t = −3.17, p = 0.002; raw accuracy gain control +7.1 %, repeated +14.5 % (Sec 2.2).
- **Within-dyad convergence**: Pearson correlation of VGG-19 fc6 features between successive drawings of the same object rises, b = 0.53, t = 5.03 (Fig. 4B).
- **Interaction-specificity** (Sec 2.3, Fig. 5): naive *yoked* viewers (all 40 drawings of one dyad, in order) improve more than *shuffled* viewers (drawings from 10 different dyads, kept in their original repetition block): interaction b = 0.10, t = 4.9; accuracy gain yoked **+15.8 %** vs shuffled **+5.6 %**. Original viewers vs yoked: overall accuracy 88 % vs 75 %; accuracy gains similar (+14.5 vs +15.8); the difference is RT (10.9 → 5.84 s vs 4.66 → 3.31 s).
- **Between-dyad divergence**: similarity of the same object's drawings across different dyads *decreases* with repetition, b = −1.4, t = −2.5; permutation CI [−0.57, 0.60], p < 0.001 (Fig. 4C). "Graphical conventions discovered by different dyads were increasingly opaque to outside observers."
- **Non-arbitrariness / diagnosticity** (Sec 2.4, Figs 6–7): one annotator group paints, per stroke, which object part it depicts (binary mask per drawing); another paints the target part most different from each foil (graded heat map, averaged over foils). Diagnosticity of a drawing = mean heat-map value over drawn pixels. Repeated drawings become more diagnostic as they get sparser, relative to controls: b = −0.05, t = −3.4, p < 0.001.
- Replication (Appendix) reproduces all effects (e.g., BIS b = 0.47, t = 12.5; yoked/shuffled interaction b = 0.09, t = 4.9; between-dyad b = −2.0, t = −4.99).

### 4.3 Definitions and findings on iconic vs symbolic [S]
- Two accounts (Sec 1): **image-based / resemblance** (drawings mean what they visually resemble, via general-purpose vision) vs **symbol-based / convention** (shared but arbitrary, socially learned associations). They propose an integrative account: viewers combine current visual evidence with learned associations.
- Predictions: (1) resemblance dominates without history, but new associations form fast, so sparser drawings that share key features still work; (2) resemblance **constrains** which conventions form: the most diagnostic features survive.
- Findings: conventions are referent-specific and interaction-specific (opaque to outsiders), but drawings **did not drift toward arbitrariness**; they kept the most diagnostic features (Abstract, Sec 3, Fig. 8).

### 4.4 Stated limitations [S] (Sec 3)
- Only visual object identity was communicated. It is "unclear whether the same general-purpose visual processing mechanisms will be sufficient" for **non-visual concepts** (semantic associations, number, causal mechanisms); conventionality may matter more there ("42" or "→").
- Why drawings fall at different points of the resemblance–convention continuum is open; production cost is a candidate; individual differences in implicit cost.
- The first design confounded sketcher and viewer decisions (viewer interruptions), fixed in the replication.

### 4.5 Repository `graphical_conventions` [S]
- The local clone (`cogtoolslab/graphical_conventions`, MIT, `LICENSE`) is the **CogSci 2019 snapshot** (`README.md:4`). The journal-version code is `hawkrobe/graphical_conventions` (MIT, has `data/` and `requirements.txt`; not cloned). The human-annotation diagnosticity analysis of Sec 2.4 is **not** in the local snapshot.
- Reusable pieces:
  - BIS: per-game z-scoring `analysis/python/df_generation_helpers.py:407-446`, `bis = z(outcome) − z(dv)` at `:450-452`, with dv = `drawDuration` or `numStrokes` (`:456-475`). [S] The code uses sketch draw duration as the time variable; the paper says response time.
  - Feature extractor: VGG-19 split by block, `layer_index = 5` → fc6 (4096-d) `analysis/python/embeddings.py:24-78`; loader pads 10 px, centre-crops/resizes to 224, **no ImageNet normalisation** (`:105-109`); CUDA-only device handling (`:90`, `:113-118`).
  - Similarity metrics: `analysis/rmd/helpers.R` — `get_sim_matrix` (Pearson or Euclidean, `:36-47`), within-dyad successive convergence `compute_within_convergence` (`:74-83`), drift from first repetition `compute_within_drift` (`:85-94`), across-dyad similarity `make_across_df`/`compute_across_similarity` (`:96-110`, note default `method='euclidean'`, called with `'cor'` in the Rmd).
  - Mixed models and permutation baselines: `analysis/rmd/analyze_refgame_dataframe.Rmd` (BIS model `:98-105`, phase × condition `:190-201`, within `:254-259`, across `:343-351`, drift permutation `:454-478`). The within/across permutation baselines are commented out (`:264-294`, `:353-392`) while the plotting chunk at `:393` uses an undefined `d.toplot`, so [I] the Rmd does not knit end to end as is.
  - Model-based "lesion" analyses (exploratory, not in the paper): stroke-deletion similarity curves `analysis/python/analysis_helpers.py:322-436`; patch-occlusion importance heat maps against distractors `analysis/python/analysis_helpers.py:520-640` and `analysis/python/generate_diagnosticity_df.py:124-169`.
  - Recognition experiment (jsPsych + MongoDB + socket.io): `experiments/recog/` (`js/setup.js` inserts one catch trial per block of 8, `:120`), version history and the yoked/scrambled designs in `experiments/README.md`; exclusion rule (≥ 4 catch trials correct, all 5 shown) `analysis/python/generate_recog_dataframe.py:81-104`.
  - Data: `results/csv/graphical_conventions_group_data_*.csv` (133 MB) contain per-trial **base64 PNG and SVG strokes** of every human sketch, plus outcome, draw duration and stroke counts: a ready corpus of human convention formation.
- Code is Python 2 era (`print '...'` in `generate_recog_dataframe.py:107`, `DataFrame.ix` in `df_generation_helpers.py:377`, `.decode('base64')` at `:396`) and depends on a private MongoDB; feature files referenced by the Rmd (`../../data/sketch_features/…`) are not in the repo.

---

## 5. LearningToDraw repository

License: **BSD-3-Clause** (`LICENSE`, `setup.py`), © 2021 Mihai & Hare. The rasteriser dependency `DifferentiableSketching` (dsketch) is also **BSD-3-Clause** (GitHub API). Last push 2021-11-10.

### 5.1 Architecture walk-through [S]
- **Entry point** `commgame.py`. `main` (`:352-365`) picks image pairing (`pair_images` or `pair_images_tranform_sender`). `train` (`:48-64`) builds loaders and a torchbearer `Trial`. `build_trial` (`:82-151`): seeds (`:83-84`), Adam over trainable params (`:96`), callbacks for snapshots and image grids, then adds the perceptual loss (`:126-139`), the "like a dog" loss (`:141-142`) and the CLIP loss (`:144-145`) as `add_to_loss` callbacks. The metric `commrate` (`:25-45`) is argmax accuracy per game.
- **Model assembly** `build_model` (`commgame.py:191-245`). Defaults: encoder `VGG16BackboneExtended`, decoder `SinglePassSimpleLineDecoder`, receiver `SimpleReceiverMLP` (`:249-254`). **Sender and receiver share one encoder object, including its trainable linear projection, unless `--separate-encoders`** (`:193-196`). Eval mode can **swap agents across two trained pairs**: sender (encoder + decoder) from pair 1, receiver (encoder + MLP) from pair 2 (`:205-231`, flags `:309-311`). The paper does not report this experiment.
- **Game forward** `model/game.py:29-62`: photo → sender encoder → decoder → sketch; `1 − sketch` if `--invert` (`:36-37`); replicate to 3 channels (`:39-41`); store un-normalised sketch in state (`:43-44`); ImageNet-normalise (`:46-47`); receiver encoder + MLP on photos and sketch (`:49-50`); scores = sketch·photo dot products, per game block when OO games pack several games in a batch (`:52-62`).
- **Backbones** `model/agents.py`:
  - `createVGG16` (`:54-79`): torchvision `vgg16(pretrained=…)`, or Stylized-ImageNet weights from a **hard-coded path** `/ssd//vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth` (`:60`) that must be downloaded manually.
  - `VGG16BackboneExtended` (`:130-189`): `vgg16.features[:layidx]` up to the chosen ReLU (`VGG16_LAYERS`, `:122-127`; relu5_3 → index 30, 512 maps, stride 16); frozen with `--freeze-vgg` (`:140-142`); trainable `Flatten → Linear(H·W·C → latent)` (`:144-147`); forward hooks store relu1_2…relu5_3 and the projection into `fstate` for the perceptual loss (`:150-156`, `:168-172`).
  - `CLIPVisualEncoder` (`:271-331`): `clip.load('ViT-B/32')`, optional freeze, linear projection of the final image embedding; hooks on every residual block, permuted to (N, tokens, D) (`:290-308`). `import clip` is **module-level** (`:251`), so OpenAI CLIP is required even for VGG runs.
  - `SimpleReceiverMLP` (`:28-50`): Linear → ReLU → Linear (defaults 64/64 via args).
- **Sender decoder** (dsketch `dsketch/experiments/characters/models/single_pass_decoders.py:318-365`): a coordinate grid in [−1, 1]² as a buffer (`:323-327`); MLP `Linear(64→64) → ReLU → Linear(→256) → ReLU → Linear(→4·nlines) → Tanh` (`:329-336`); reshaped to `[batch, nlines, 2, 2]` (`:352-359`); `line_edt2` (`:361-364`). `Decoder.forward` (dsketch `…/models/model_bases.py:81-91`) runs params → σ² → EDT² → `raster_soft` → `× contrast`.
- **Data/game construction** `utils/game_data.py`: `build_game_loader` (`:14-53`) permutes the receiver images inside a batch and stores the inverse permutation as the target labels. Original game: a plain DataLoader where batch = pool, so batch 100 → 99 distractors (`:231-249`). OO games: `GameSampler` (`:85-168`) packs one image per class per game; `collate_oo_diffmixed` (`:171-181`) pairs each sender image with a different same-class image. `RANDOM_TF` (`:258-261`) is the optional sender-side augmentation.
- **Losses** `utils/game_loss.py`: `HingeLoss` = `F.multi_margin_loss(...)` **computed on CPU** (`:16-25`); PyTorch divides the summed margins by the number of classes, a constant rescaling of Eq. 3. `CELoss` (`:47-56`). `reconstruct_loss` (pixel MSE, `:40-44`).
- **Logging/analysis** for the human study: `utils/game_logging.py:13-109` (`GameLogger` writes every test game as PNGs + `games.json`/`games.csv`); `sample_gamelogs.py` samples 30 games (`:15`, matching 30 games per setting); `render_game.py` renders one game to HTML; `clip_analysis.py:35-75` (CLIP prompt probe, `index % 10` = class, `index // 10` = template) + `analyse_logs.py` (Table 2 counters); `dump_hashes_classes.py` maps image hashes to classes; `render_sketches_seeds.py` builds the 10-seed overlay (App J). **The human-study web UI itself is not in the repo.**

### 5.2 How the differentiable rasteriser works [S]
1. **Distance transform per primitive.** `line_edt2` (dsketch `dsketch/raster/disttrans.py:52-108`): for segment a→b and every grid pixel p, t0 = ((p−a)·m)/(|m|² + ε) with m = b − a; d² = |p−a|² if t0 ≤ 0, |p−b|² if t0 ≥ 1, otherwise the squared perpendicular distance (`:95-104`). Fully vectorised, `[batch, nlines, H, W]`. Also `point_edt2` (`:21-49`) and `curve_edt2_polyline` (`:379-423`), which samples a parametric curve (cubic Bézier by default, Catmull–Rom available) into `segments` straight pieces and takes the min of their EDT².
2. **Relaxed rasterisation.** `exp(dt2, σ²) = exp(−d²/σ²)` (dsketch `dsketch/raster/raster.py:80-96`). σ² sets both visible width and how far gradients reach (paper: 5e-4 on a [−1, 1] grid). A hard version (`nearest_neighbour`, `:44-58`) is used only for metrics.
3. **Composition.** `softor(R) = 1 − ∏(1 − R_i)` (dsketch `dsketch/raster/composite.py:10-25`), applied in `Decoder.raster_soft` (`…/models/model_bases.py:64-66`); `over` (`composite.py:52-80`) for coloured strokes. The game inverts to black-on-white (`model/game.py:36-37`), which gives Eq. 2.
4. Gradients flow from receiver loss → raster → EDT² → endpoint coordinates → MLP. Only pixels within a few σ of a stroke receive meaningful gradient, which [I] is why σ² matters for optimisation as much as for appearance.
- [S] Detail: in current dsketch `main`, `get_callbacks` that anneals σ² is defined at **module level** (`…/models/model_bases.py:173-183`), not inside `AdjustableSigmaMixin`, so `--sigma2-step` schedules are inert. LearningToDraw does not use annealing, so the paper results are unaffected.

### 5.3 Perceptual losses in code [S]
- `make_perceptual_loss` (`utils/game_loss.py:94-121`): takes the sender-photo feature maps captured during the forward pass (`state[FEATURE_MAPS]`), re-encodes the (normalised) sketch with the **same frozen sender encoder** (`:102-105`), unit-normalises each map along dim 1 (`normalize_tensor`, `:89-91`), squared difference summed over dim 1, spatially averaged (`spatial_average`, `:79-86`), weighted by `--sop-weights` (default `[1,1,1,1,1]`) and scaled by `--sop-coef` (`:115-119`).
- Iteration over weights stops at `min(len(res), len(weights))` (`:116`). For VGG the 6th key is the trainable `projection`, so the 5 default weights cover exactly relu1_2…relu5_3. [I] For the CLIP encoder the dict holds resblock0–11, `CLIPvisual` and `projection`; with 5 default weights only **resblocks 0–4** enter the loss unless `--sop-weights` is passed with more entries, while the paper says all blocks are weighted equally. Also for ViT features shaped (N, tokens, D), `normalize_tensor` normalises along the **token** axis (dim 1), not the feature axis (`agents.py:303-304` + `game_loss.py:89-91`). Neither point can be checked against the authors' command lines, which are not given for the ViT runs.
- `make_dog_perceptual_loss` (`:127-172`): the same loss against one fixed image (App D).
- `make_CLIP_loss` (`:175-226`): per sample, 4 augmentations (RandomPerspective(0.5) with white fill, RandomResizedCrop(scale 0.7–0.9)) (`:181-184`, `:194`), loss = Σ(1 − cos) between the photo embedding and each augmented sketch embedding (`:219-220`). Python loop over the batch, summed rather than averaged, so its scale grows with batch size.
- `LearnableWeightedPerceptualLoss` (`:290-323`, `--learn-sop-weights`): learnable non-negative unit-norm layer weights. Not used in the paper.

### 5.4 Dependencies and Apple MPS [S + verified]
- `requirements.txt`: `torch~=1.7.0`, `torchvision~=0.8.1`, `torchbearer~=0.5.3.dev0` (installed from GitHub source), `pillow~=8.0.1`, `clip~=1.0.0` (the module used is OpenAI CLIP), `DifferentiableSketching` from GitHub. dsketch's own requirements add `lpips`, `pyx`, `quickdraw`, `scikit-image~=0.17`, `scikit-learn`, `matplotlib`.
- **The repo as pinned cannot use MPS**: torch 1.7 predates the MPS backend (added in 1.12). Upgrading breaks it:
  - `utils/game_data.py:259` calls `transforms.RandomAffine(..., fillcolor=None)`. **Verified**: the installed torchvision 0.28 has no `fillcolor` parameter, and the transform is built at import time, so importing `utils.game_data` fails.
  - torchbearer 0.5.3 is a 2020-era trainer that relies on old torch internals; I did not test it on torch 2.x. [I] Expect friction.
  - Default devices are `'cuda:0' if available else 'cpu'` (`commgame.py:267-268`); `--device mps` would be passed through as a string. `HingeLoss` already moves to CPU (`game_loss.py:25`), which works but syncs every step. The SIN weight path is hard-coded (`agents.py:60`).
- **The core math does run on MPS.** I re-implemented `line_edt2` + exp raster + inverted soft-or + a VGG16 `features[:30]` forward/backward + multi-margin loss in a scratch script (scratchpad only, nothing in the project touched) with torch 2.13 / torchvision 0.28 on this Mac: forward and backward work, gradients reach the line MLP, **~0.70 s per iteration** for batch 100, 20 lines, 96 px (random VGG weights, sketch branch only). [I] With the photo branch and the perceptual pass, a full STL-10 replication (~50 iterations/epoch × 250 epochs) is roughly 4–6 h on the M4 Pro. That is an estimate, not a measurement.

### 5.5 What we could reuse [I]
- **Rasteriser primitives** (BSD-3, ~60 lines): `line_edt2`, `curve_edt2_polyline` with `cubic_bezier`, `exp`, `softor`/`over`. They map directly onto a ring renderer: express tendrils and blots as Bézier strokes in polar coordinates (θ, r) converted to Cartesian, or compute the EDT on a polar grid. A per-stroke σ² is a natural width parameter, and per-stroke amplitude before soft-or gives ink weight. This gives v1-style continuous strokes, but **analytically**, without v2's cut-and-paste of film marks.
- **Frozen pretrained listener trunk + small trainable head** (`VGG16BackboneExtended`, `CLIPVisualEncoder` pattern): VGG16 to relu5_3 or CLIP ViT-B/32, frozen, with only a linear projection and MLP trained. Pretrained weights now come from torchvision (`VGG16_Weights.IMAGENET1K_V1`) and open_clip, not the repo.
- **Perceptual-loss code** (`make_perceptual_loss`): reusable as is for any "make image A look like image B to VGG" term.
- **Agent-swap evaluation** (`commgame.py:205-231`): the exact mechanism we need for "private code vs shared language" (sender of seed A read by listener of seed B).
- **Game logging + CLIP probe pipeline** (`GameLogger`, `sample_gamelogs.py`, `clip_analysis.py`, `analyse_logs.py`): a template for exporting decipherment kits and probing logograms with an external model.
- **Do not adopt** torchbearer, the pinned environment, or the shared sender/receiver encoder default.

---

## 6. Synthesis for Heptapod Ink

### 6.1 What the evidence says about our core questions [I, tied to S]
1. **Successful agent communication predicts nothing about outsider readability.** Mihai & Hare: agents at 93–100 % while humans sit at chance (8.3 % vs 10 % with VGG, 5.6 % with ViT) without a representational bias (2021a Table 5, 2021b Table 1). Feedback over 30 trials barely helps (13.9 %, App K.2). Our v1/v2 logograms are trained only on the game loss plus ink costs, so the default expectation for the planned blind decipherment is "near chance" unless we add a bias that ties form to meaning.
2. **A perceptual prior toward a fixed look does not create readability.** App D: forcing sketches to resemble one dog image keeps agents communicating while humans "stand no chance". Our v2 already forces the film's ink style (real marks), which is the same situation: it buys Arrival aesthetics, not decipherability. A style loss toward the 38 film logograms would behave the same way.
3. **Readability came from grounding form in a referent the human can see.** Every interpretability gain in these papers comes from making the drawing resemble the **photo of the referent** (perceptual or CLIPDraw loss) or from a sketch pipeline pretrained on human sketches (Qiu). Our meanings are symbolic sentences with no visual referent. Hawkins et al. (Sec 3) flag exactly this as open: for non-visual concepts, conventions should rely more on learned association than on resemblance. Their human data show that conventions without resemblance support are **interaction-specific and opaque to outsiders** (yoked +15.8 % vs shuffled +5.6 %; between-dyad divergence, Fig. 4C).
4. **Conventions differ by run.** Hawkins: different dyads diverge for the same object. Mihai App J: seeds agree on key strokes but differ in detail. [I] For us, an independently trained run is a "different dyad"; if logograms for the same sentence share nothing across seeds, any outsider learning is learning one private code.
5. **Order and sequencing matter.** Qiu: the one-step game fails (22.87 % seen) while the sequential game with early stopping produces the most symbolic and compact conventions (Table 1, Fig. 4). [I] This conflicts with the order-free claim of our script. A stroke-by-stroke reveal with early stopping would push information to the front and create a privileged start. The partial-reveal curve should be measured without that training pressure, or both regimes compared explicitly, as the Codex review suggests (averaging over reveal origins).

### 6.2 Concrete things to adopt
Each item lists the evidence and how it maps onto our setup.

1. **Cross-seed listener swap as the first "private code" test.** *Evidence*: the swap code exists (`commgame.py:205-231`); Hawkins shows interaction-specific conventions are opaque to outsiders (Sec 2.3). *Do*: train N ≥ 3 seeds; read each speaker with every other seed's frozen listener (zero-shot), then with a fresh listener head trained on k = 10, 50, 200 examples (acquisition curve). Report the diagonal (own pair) vs off-diagonal matrix.
2. **Qiu's symbolicity test as the "fresh learner" protocol.** *Evidence*: Sec 5.2.3; a frozen VGG16 with only the last FC fine-tuned, 70/30 split, separated settings from ~43 % to ~79 % (Fig. 4). *Do*: freeze a speaker, render logograms, train a fresh frozen-backbone + linear probe per meaning slot on seen combinations, and test on held-out combinations and rotations. Use two probes (frozen ImageNet/CLIP backbone and a small from-scratch CNN, as Codex warned that natural-image features may miss ink detail). This answers "can a newcomer learn it from limited examples" better than retraining our own listener.
3. **Hawkins-style convergence measurements.** *Evidence*: `analysis/rmd/helpers.R:74-110`, Figs 4B–C. *Do*: (a) within-run convergence: similarity between logograms of the same sentence at successive training checkpoints; (b) between-run divergence: similarity of the same sentence's logograms across seeds per checkpoint, with a permutation baseline that shuffles seeds within (sentence, checkpoint). Use a **rotation-invariant** similarity (max over rotations, or features of the polar-unwrapped strip), because a rotated copy should count as identical.
4. **Semanticity as a sanity layer on topographic similarity.** *Evidence*: Qiu Table 2. *Do*: correlate meaning distances (role-aware, plus optionally word-embedding distances between words) with distances in a **fixed external encoder** that played no part in training, using average ranks and a permutation null (fixes the tie problem Codex found). Report it next to our own topsim; do not treat it as proof of compositionality.
5. **A pretrained, frozen listener trunk as an option, not the default.** *Evidence*: Mihai keeps the backbone frozen so the visual system "cannot collapse and remains universal" (Sec 3.2); ViT-B/32 + CLIPDraw gave the best human transfer (62.7 %, 2021b Table 1). *Do*: compare our CNN listener against frozen VGG16-relu5_3 / CLIP ViT-B/32 + trainable head under the same rotation/noise channel. [I] A frozen general-purpose reader pressures the speaker toward distinctions any vision system sees, which should help outsider models and possibly humans. Keep speaker and listener encoders **separate** (the repo's shared default merges both agents into one network).
6. **Grounding for readability, if we want it, must tie each word to something a reader already knows.** Options, in order of how much semantics they import: (a) iconic seeds: give each entity/verb a reference image or icon and add a Mihai-style perceptual loss between the rendered *mark for that word* and its icon (requires a per-word region or mark, so it presupposes compositional structure); (b) CLIPDraw-style loss between the rendered logogram (augmented) and the CLIP text embedding of the sentence. *Evidence*: perceptual / CLIPDraw losses raise human accuracy from chance to 38–63 % at a 6–20 point agent cost (2021a Table 1, 2021b Fig. 2). *Caveat*: (b) imports CLIP's semantics, so CLIP can no longer serve as the outside reader (the probe bias the authors flag in 2021b Sec 4); it also makes the "emergent" claim weaker and should be described that way.
7. **Human evaluation protocol.** *Evidence*: Mihai App K (forced choice with K = 9, 30 trials per condition, feedback vs no-feedback groups, instance vs class rate); Hawkins (yoked vs shuffled, attention checks ≥ 4/5, BIS combining accuracy and time, mixed models). *Do*: forced-choice reading with a small candidate set (e.g., 1 target + 9 distractors that differ in one slot), feedback and no-feedback groups, scoring both exact sentence and per-slot (role-aware) accuracy, plus a **yoked vs shuffled** arm: logograms from one seed vs logograms mixed across seeds. Use the `experiments/recog` jsPsych design as the template. Mihai's 6 people per group is a pilot; state uncertainty intervals.
8. **Outside-model reader with a held-out probe.** *Evidence*: 2021b Sec 4 (CLIP prompt probe, with its bias). *Do*: the planned vision-model decipherment should use a model that was never used in training or as a loss, matching the "kit" humans get.
9. **Rasteriser upgrade path.** *Evidence*: §5.2. *Do*: if v3 wants continuous, differentiable strokes with film-like ink, build them from dsketch's EDT primitives (Bézier strokes, per-stroke σ² and amplitude, soft-or) instead of v1's sector brush. Port the code (BSD-3, attribution) rather than installing the repo.
10. **Stroke/mark ablation as a causal probe.** *Evidence*: Mihai 2021b Sec 5 proposes stroke-bundle interventions; Hawkins' snapshot has stroke-deletion similarity analyses (`analysis_helpers.py:322-436`). *Do*: delete or swap individual marks and record which meaning slots the listener changes. This covers the "controlled mark substitutions that transfer across contexts" test in CODEX_REVIEW.md.

### 6.3 What would not transfer, and why [I]
- **Photo-referential iconicity.** Qiu's iconicity (generalisation to unseen images) and Mihai's perceptual loss both need a picture of the referent. Our sentences have none; "iconicity" has no direct analogue unless we add icons (6.2 item 6).
- **Qiu's training stack.** RL with a pretrained neural renderer and a two-stage sketch pipeline costs ~20 h per run on an A6000, has no license, and solves a problem we do not have (our renderer is already differentiable end to end). The *ideas* transfer (early-stop receiver, symbolicity and semanticity probes); the code does not.
- **Referential choice as the main objective.** Mihai and Qiu train discrimination among K candidates; picking 1 of N needs far fewer bits than full reconstruction (Codex review). It is fine for human evaluation (forced choice), not as a replacement for our reconstruction game.
- **Shared encoder between agents.** It makes sender and receiver one model and strengthens the private-code reading; we should not copy it.
- **ImageNet backbone as an unbiased reader.** Mihai's own Sec 5 question (are drawings shaped by ImageNet labels?) applies to us: a frozen ImageNet trunk may ignore ink texture that our marks rely on. Keep a from-scratch control.
- **Hawkins' resemblance-based optimism.** Their "non-arbitrary conventions" result relies on a visible object with diagnostic parts. For abstract sentences their own discussion predicts more arbitrariness, so their finding should not be cited as evidence that our script will be readable.
- **The LearningToDraw environment.** Pinned torch 1.7, torchbearer, module-level CLIP import and the `fillcolor` break make it a reading reference, not something to run on MPS.

### 6.4 Open discrepancies to remember [S]
- Qiu: M = 3 in the text vs chance 25 % and `--game_size=4` in the README; Table 2 vs text on retrieve vs complete semanticity.
- Mihai 2021a vs 2021b: VGG l_game original-game rate 71.8 % (2021a Table 1, 10-seed mean) vs 75.7 % (2021b Fig. 2). [I] Probably a different single run; neither paper explains it.
- LearningToDraw ViT perceptual loss vs paper description (layers used and normalisation axis), §5.3.
- Hawkins: the paper's BIS uses "response time"; the snapshot code standardises sketch draw duration (`df_generation_helpers.py:463`).
