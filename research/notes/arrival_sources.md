# Arrival sources: how the film's script was made, analysed, and what we can reuse

Scope: Christopher Wolfram's notebooks, Coon's and Cuskley's texts, three third-party repos, the Replicate model, and Wikipedia (secondary). Compared against our own `reference/prep_library.py`, `reference/prep_marks.py` and `reference/measure.py`.

Conventions:
- **[S]** marks what a source states. **[I]** marks my interpretation or my own measurement.
- Notebook cells are cited as `file.nb #N`, where N is the cell index in document order (Input, Output, Text, Section and Message cells all counted, starting at 0). I extracted cells by parsing the box expressions. Code inside RowBox was flattened to text, and embedded images were decoded from `CompressedData` (zlib + Mathematica's `!boR` dump; the last W*H bytes of each one are the raster). Quotes below are the flattened code.
- To identify embedded images I correlated each decoded raster against the 38 files in `reference/logograms/` (r = 1.00 means an exact match).

---

## 1. How the logograms were designed, and what counts as canonical

### Who made them, and how many

- **[S] Wikipedia (secondary), citing Wired [1], Gizmodo [5], Insider [6], a YouTube interview with Bertrand [7] and Thrillist [11]:**
  - Screenwriter Eric Heisserer started from the etchings on the One Ring, and his early drafts looked like Tengwar.
  - Production designer Patrice Vermette's team tried alphabet-like scripts based on human writing systems, which looked "too human".
  - Martine Bertrand, an artist married to Vermette, drew "inky and smoky" designs. Villeneuve approved them and asked her for "15 similar designs".
  - The article credits Vermette and Bertrand, and calls the result "a collaborative effort of the crew". Vermette's team "oversaw the semantics".
  - Jessica Coon "was asked to annotate the characters as if she were analyzing them in real life" [5][10].
  - There was "a dictionary of roughly 100 Heptapod B logograms, 71 of which were used in the film" [6][7][11].
  - Stephen and Christopher Wolfram used the Wolfram Language "to section each character into 12 different parts" and found repeating patterns [1][8][9].
- **[S] Cuskley ("Imagining Heptapod B", paras 2–4):** the designers first tried an alphabet based on human scripts, which came out clunky. Bertrand then "just designed something weird". Meaning was assigned afterwards: "They mapped them onto some utterances, and had computer scientists analyse them for patterns, post-hoc. Those patterns became the 'rules' of Heptapod B". She adds that the symbols "were designed to look cool, and that's pretty much it… it turns out there's naturally enough structure in looking cool to meaningfully analyse it."
- **[S] Notebook paths.** Every notebook loads `~/Dropbox/SOYL/ReceivedMaterial/HiRez Logograms/Final Scripted Logograms/ScriptLogoJpegs/` (`sectionBreaking-01.nb #1`, `unwrapping-01.nb #1`, `featureDetection-01.nb #2`, `ca-01.nb #46`). "SOYL" is *Story of Your Life*, the production codename. The full-set output in `arrivalLiveCoding.nb #48` has 38 images, the same 38 files we have in `reference/logograms/`.
- **[S] Embedded image metadata** (`arrivalLiveCoding.nb #49`): Photoshop CS6, 300 dpi, grayscale. Our files are all 3300×3300 at 300 dpi.
- **[I]** So we hold the 38 "final scripted" logograms Wolfram worked from. That is 38 of roughly 100 designed and 71 on screen. The ~100 and 71 figures come only from secondary press.

### What structure is claimed

- **[S] Wolfram's code** splits each logogram into 12 equal angular wedges (`sectionBreaking-01.nb #3`, `sectionCount=12`). The notebooks contain **no text claiming what any section means**. The only notebook text cells are "Animated (not very exciting)", performance notes, and "Will take about 2 hours (on Chris's computer)" (`arrivalLiveCoding.nb #31`).
- **[S] Wolfram's comparisons pair logograms that share a word.** The three hand-picked `ImageDifference` cells compare:
  - Earth1 with HepPurpEarth_1 (`arrivalLiveCoding.nb #7`)
  - AbbotIsDead1 with Abbott1 (`#9`)
  - OfferWeapon_1 with OfferWeapon1 (`#11`)

  The five pasted result images (`#49–#53`) are aligned-difference images that match these pairs:
  - |OfferWeapon1 − Weapon1| (r = 0.99)
  - |Abbott1 − WhereIsAbbott1| (0.63; the "Where is Abbott?" caption is visible)
  - |Human1 − Human_1| (0.98)
  - |OfferWeapon1 − OfferWeapon_1| (0.99)
  - |Time1 − ThereIsNoLinearTime1| (0.79)
- **[I] My measurement backs this up.** I took the share of logogram A's ink lying on B's ink (400 px, B dilated by 2 px, no alignment). The median over all 1,406 ordered pairs is 0.42. The top meaningful pairs are near 1:
  - Weapon ⊂ OfferWeapon: 1.00
  - Time ⊂ ThereIsNoLinearTime: 0.98
  - Earth ⊂ HepPurpEarth: 0.95
  - Abbott ⊂ WhereIsAbbott: 0.94
  - LouiseHasWeapon ⊂ MessageHereLouiseHasWeapon: 0.93

  So the production built compound logograms by **reusing the base word's exact ink at the same position and angle on the canvas, and adding marks elsewhere on the ring**. For example, OfferWeapon is Weapon plus one heavy cluster at the bottom. Small, thin logograms such as Ian and Louise also score above 0.8 inside large, heavy ones, so containment needs an ink-ratio guard.
- **[S] Variants exist.** Human1 and Human_1, and OfferWeapon1 and OfferWeapon_1, are near-duplicates (containment 0.86–0.98). These are two drawings of the same meaning.
- **[S] Tense, negation, and "complexity relates to size":** none of my allowed sources state any of these for the film. The Wikipedia summary explicitly does not cover tense or negation markers. The novella properties reported on Wikipedia, all citing Chiang, are:
  - size marks paragraph or page breaks [20]
  - noun declension is shown by stroke curvature, thickness and spacing [22]
  - subject and object are marked by orientation [23]
  - the writer knows the whole sentence before the first stroke [21]

  These are claims about the **book**. They are not properties of the film's logograms.
- **[I] Size.** Across the 38 files the ring radius is nearly constant: the RANSAC fits give r/width between 0.286 and 0.371 (median 0.352), from `fragments_index.json`. Complexity shows up as more ink and more clusters on a ring of the same size, not as a bigger ring. Compound / base ink ratios: Weapon/OfferWeapon 0.80, Earth/HepPurpEarth 0.82, Time/ThereIsNoLinearTime 0.91.

---

## 2. Wolfram notebooks, one by one

License: CC BY-NC 4.0 (`COPYING.md`). `Readme.md` maps each notebook to its timestamp in the live-coding video (youtu.be/r8nTifCIr0c). The working example `i` in every notebook is **CowrittenLogogram1** (3300 px, resized to 500; matched at r = 1.00). Every analysis runs at 500×500.

### sectionBreaking-01.nb: the "12 sections"

- `#5`: `breakLogogram[i_,c_]` uses `CirclePoints[Norm@ImageDimensions@i, c]` to build c triangles from the centre `{0,0}`. Each triangle becomes a white-on-black polygon mask, and each mask is multiplied with the logogram to give c pie-slice images.
- The wedges are anchored at the **image centre**, at a **fixed absolute angle**, and run to the image corner. They therefore cut straight through any cluster that straddles a boundary.
- `#8`, `#10`: the 12 boundary lines are drawn in red, with a `Manipulate` over a rotation offset `t ∈ [0, 2π/12]` to slide the grid by eye.
- `#11`, `#13`: each wedge is tinted with `ColorData[104]` to show the partition.
- `#15–#18`: the sections of all 38 logograms go into a 38×12 grid.
- `#19`: `straightSections = ImageRotate[#, k·(−2π/12)]` rotates wedge k back to a common orientation, so column-wise comparison is possible (`#20–#22`).
- **Output:** visual grids only. There is no statistic and no label.
- **[I]** The "12 parts" in the press reports is this code: a hand-chosen, content-blind, equal-angle partition used to eyeball which wedges repeat across logograms. Our `SLOTS = 12` (`prep_library.py:20`) happens to match it.

### unwrapping-01.nb: polar unwrap

- `#3`: `ImageTransformation[i,{ArcTan@@#,Norm@#}&]` is a first attempt at a direct polar transform.
- `#5–#6`: `PixelValuePositions[Binarize@ImageResize[i,200],0]` plus a `Manipulate` over the centre `(x,y) ∈ [50,150]²`. The ring is plotted in (angle, radius) coordinates, and the centre is **picked by hand** until the ring comes out flat.
- `#7`: the working method:

  ```
  unwrap[i_,width_:600] := ImageAssemble@List@Table[
    ImageTake[ImageRotate[i,t,All,Background->White],
              {0, h/2}, {Floor[w/2]}], {t,0,2Pi,2Pi/width}]
  ```

  It rotates the whole image 601 times and takes the single centre column from the top edge down to the centre each time. The centre is assumed to be the image centre, and the radius runs from 0 to h/2.
- `#10–#11`: `rotateImage` is a cyclic column shift, which shows that rotating the logogram is a horizontal shift of the strip.
- `#12–#17`: five random logograms are unwrapped and stacked so they can be compared under a shared shift.
- **Compared with ours:**

  | | Wolfram | Ours (`prep_library.py`) |
  | --- | --- | --- |
  | Centre | image centre (hand-tuned in an exploration cell) | RANSAC circle fit; score = angular coverage of 90 bins × 1000 + inliers, inlier band 0.006·n (`:27-53`) |
  | Radius normalisation | none (pixels from 0 to h/2) | radius relative to the fitted ring, from 0.45r to 1.65r (`:18`, `:59`) |
  | Sampling | 601 rotations, one column each | one `grid_sample` at 3600×420 (`:56-64`) |

  **[I]** Fitting matters. Our fitted centres sit a median of 7.4% of r away from the image centre, and up to 23% (AbbotIsDead1 0.233, WhereIsAbbott1 0.194, Abbott1 0.177). With Wolfram's centre assumption those rings would come out of the unwrap as waves of ±23% r. Nothing to adopt from his unwrap; ours is better on every count.

### featureDetection-01.nb: what the "analysis screens" look like

1. **`#5`:** `SkeletonTransform@ColorNegate@i`, the medial skeleton.
2. **`#8`:** `ImageCorners[i, 3, 0.1, 5]` with parameters r = 3, threshold 0.1 and minimum distance 5. A `Manipulate` sweeps r ∈ 0–10, t ∈ 0–0.13 and d ∈ 1–10 (`#11`); a text cell calls it "not very exciting".
3. **`#13–#21`:** the corners are joined with an all-pairs `Outer[...Line...]`. Edge weights are either thickness ∝ 1/distance (`#16`) or opacity = 3000/distance² (`#18`, animated in `#21`). Corners get green boxes.
   - **[I]** This is the "network of little dots connected across the symbol" that Cuskley describes from the film's screens ("Imagining Heptapod B", para 3). It is a visualisation, and not a parser.
4. **`#25`:** `ImageCorners[ColorNegate[i], n]` for n = 1..16, the top-n corners.
5. **`#27–#30`:** `Radon` transform at 350×350.
   - **[I]** There is a bug: the `Table` over `t` never uses `t`, so every frame is identical.
6. **`#32–#36`:** Sobel convolution, `GradientFilter[i,3]` and `GradientOrientationFilter[i,3]`.
7. **Overall:** no detected feature is ever turned into a symbol inventory. The README calls it "a quick look at an attempt to detect features and map symbols".

### arrivalLiveCoding.nb: the actual comparison work

- **`#7`, `#9`, `#11`:** `ColorNegate@ImageDifference[a,b]` on the shared-word pairs (see section 1).
- **`#13–#18`:** `FeatureExtraction[i2]` gives a default image feature extractor (a pretrained net in Mathematica 11.1). Its output is a real vector (`#17`).
- **`#19`:** `DimensionReduce[featureVectors, 2, Method->"LatentSemanticAnalysis"]`, plotted with each logogram as its marker.
- **`#21`, `#23`:** `ClusteringTree` with complete and average linkage. The output is a dendrogram image; the cluster memberships are not recoverable as text.
- **`#25`:** `alignedDifference[a,b] := ImageDifference[a, ImageAlign[a,b,Background->White]]`, which registers b to a and then takes the absolute difference.
- **`#29`:** `Nearest[i2, DistanceFunction -> (ImageMeasurements[alignedDifference[#1,#2],"Mean"]&)]`. The nearest neighbour is defined by the mean aligned pixel difference. Text `#31`: about 2 h; it was aborted (`#33`).
- **`#37–#43`:** `FeatureNearest[i2]` gives deep-feature nearest neighbours. `alignedDifference[#, fnf[#,2][[2]]]&/@i2` produces the difference with each logogram's nearest neighbour, for all 38 (`#48`).
  - **[I]** The pasted images `#49–#53` look like picks from this run. Every one is a shared-word or variant pair. In other words, generic image features plus alignment were enough to retrieve the logogram sharing a word.
- **`#54`:** `ImageMultiply@@i2` overlays all 38.
- **`#56–#59`:** the ink histogram. `Mean@ImageMeasurements[ColorNegate/@i2,"Mean"]` = **0.1041** mean ink coverage. Our `measure.py` gives median ink 0.099, p10 0.063 and p90 0.156 (`reference_stats.json`). The two agree.
- **`#60–#62`:** `ImageMesh` and `MorphologicalGraph` of the example logogram.

### Notebooks with nothing to reuse for the script

- **3d-01.nb:** relief and 3D plots of `DistanceTransform@ColorNegate@i` (`#5`, `#9`), multiplied by blended random noise to animate (`#12`, `#22`, `#26`). This is screen graphics only.
- **ca-01.nb:**
  - `#2`: `CellularAutomaton[{rule,{2,{{2,2,2},{2,1,2},{2,2,2}}},{1,1}}, ...]` is an outer-totalistic, 2-colour, 9-neighbour automaton seeded with the binarised logogram at 100 px plus 10 px of padding.
  - `#10`: rules are screened for being non-static, smaller than 80,000 px, and shrinking their borders.
  - `#16`: `goodRules={174826,…,256576}`.
  - `#47–#55`: a `MorphologicalGraph` of an automaton frame, drawn with random logograms as vertex shapes and exported as `graph-05.jpg`.
  - Growth visuals only.
- **maps-01.nb:** 12 `landingSites` as `GeoPosition` points (`#0`), a 3-nearest-neighbour graph with 1000-mile disks (`#5`), `FindShortestTour` (`#7`), and a sweep over map projections. `goodProjections` was chosen by clicking buttons (`#30–#32`). Not about the script.
- **shipChristopher.nb:** an OBJ previs mesh of the shell (`#0`, "From Erik Y"), vertex-normal tours, and exploded polygons. Volume 4.42×10⁹ (`#38`); bounding box 2295 × 4496 × 856 (`#42`). Not about the script.

### What to replicate

**[I]**
1. Keep our ring fit and polar unwrap; they are strictly better than Wolfram's.
2. Add Wolfram's **38×12 straightened-section grid** as a diagnostic for our script. Render each generated logogram's 12 slots side by side, de-rotated, to see which slots repeat across meanings. We can do this directly on our polar strips.
3. Replicate his **shared-word retrieval**: does a generic image embedding's nearest neighbour share a meaning element? Run it on the canon as a baseline (expect hits for Weapon/OfferWeapon, Time/ThereIsNoLinearTime, Earth/HepPurpEarth and so on), then on our script.
4. Use his **aligned difference images** for minimal pairs. We already diff minimal pairs in v1.
5. The corner-network picture is worth borrowing only as a film-style visualisation.

---

## 3. What Coon and Cuskley claim

### Coon, "Alien Speak" (Sloan Science & Film, reprinted in *Stages*, Jan 2019)

**[S]** This is an essay about fieldwork and Universal Grammar. **It makes no claim about how Heptapod B is structured** (nothing on circles, semasiography or non-linearity). Its points:

- Human languages vary "in limited and constrained ways" (p. 2, lines 43–52).
- For Heptapods, "even the most basic human language distinction, like the difference between 'nouns' and 'verbs', is no longer a given" (p. 2). "Even seemingly benign concepts, like asking a question, may have no direct correlate in Heptapod" (p. 3).
- The film draws on Sapir-Whorf. Among human languages the strong version "has been shown to be not only wrong, but dangerous" (p. 3).
- Banks "approaches the daunting task… as any good fieldworker would": rapport first, start with the basics, and accept slow progress (pp. 2–3).
- Coon was the film's scientific consultant in 2015 (bio).

**[S] Secondary.** Wikipedia cites a different Coon paper ("The linguistics of Arrival", 2020, *Language Invention in Linguistics Pedagogy*) for the claim that Heptapod A allows unlimited centre embedding. That paper is not in our folder.

### Cuskley, "Building Heptapod B" (blog, April 2025, CC BY-NC-ND 4.0)

**[S] Linguistic claims:**

1. **Heptapod B is a separate language, not a writing system for Heptapod A.** Heptapods are "bimodal bilinguals". The right analogue is a **signed language**, not writing: "Heptapod A is to Heptapod B as spoken English is to ASL or BSL; not as spoken English is to written English" ("The film and the story", paras 5–7).
2. **Semasiographic.** Quoting Chiang p. 108, the script conveys meaning without reference to speech, and "has its own system of rules for constructing sentences, like a visual syntax that's unrelated to the syntax for their spoken language."
3. **Non-linear.** "The whole point of Heptapod (A or B) is that it isn't linear". A font or a typed sequence "feels like cheating". The in-film touchscreen that assembles segments "feels like sequential humans making Heptapod B". "Heptapods build a symbol all at once" ("Imagining Heptapod B", paras 5–6).
4. **Many languages.** Expect many Heptapod languages, so variation between generations is welcome ("Growing Heptapod B", para 6).
5. The film's rules were found post-hoc in designs made to look cool (section 1).

**[S] Method** (the code is on OpenProcessing sketch 2607570, which returned 403 to WebFetch; I read the same files served from `ccuskley.github.io/interactives/heptapod/`):

- It is a p5.js particle system:
  - `heptapodsketch.js`: `setup`/`setQuote`. `textToPoints` renders the English quote in Reenie Beanie at size 70 with sampleFactor 0.5.
  - The number of points is the number of particles.
  - Three centre emitters spew particles with a semi-random wind.
  - Particles that reach the circle (radius = `csize/2`, `csize = windowWidth/3`) solidify as dots 3–15 px across.
- **The only quote-dependent parameter is the particle count**, and it sets an upper bound on protrusions:
  - Code: `nprotrusions = int(random(2, int(quotePoints.length/500)))`.
  - The post says "/400". The code uses /500.
  - Protrusions are spaced every `(N−2000)/nprotrusions` particles, starting at particle 500.
- **Protrusion** (`protrusion.js`):
  - core size `random(20, R/4)`
  - concentric bulge circles shrinking in steps of 5 px
  - `random(3,15)` tendrils
- **Tendril** (`tendril.js`):
  - target radius `random(random(0.7R, R−1.2·core), random(R+core, 1.5R+core))`; whether it ends up inside or outside the ring decides "innie" or "outie"
  - stroke width 3–7
  - a bulb with probability 0.4, size 2–15
- Afterwards the particles flock to their stored text positions and the symbol morphs into the English quote. Keys: `f` = flee, `j` = next quote.
- The quotes are 10 lines from the story (`quotes.json`).
- **Things she says it cannot do:** gaps in the ring, the 3D smoky look, appendage emergence.

**[S] Rules and grammar:** she defines none. The footnote "Tie quote" suggests mapping content words to protrusions, syntactic or thematic role (SVO, agent/patient) to protrusion properties, and tendrils to concreteness. She notes that a deterministic mapping would require "all of English morphosyntax and semantics". The footnote "More Complexity" says the link between length and complexity is mostly an artefact of the font.

**[S] Code license:** the post invites remixing "with attribution" on OpenProcessing. The served JS files carry no LICENSE, and the post itself is CC BY-NC-ND.

### What this means for evaluating our script

**[I]**
- **Semasiography.** Our script has to be judged as an independent visual language, so English word order must not leak in. Test this: the angular order of marks should not correlate with the slot order in the meaning representation. Compare positional disentanglement against bag-of-symbols disentanglement (v3 plan item 4).
- **The sign-language analogy** points to simultaneity: several meaning components expressed at once in space. The right measures are partial-reveal and occlusion curves. The claim to check is that meaning is spread over the ring, not that the ring is read in a fixed order.
- **Cuskley's "touchscreen" critique applies to v2.** Picking a segment and mark for each of 12 slots is exactly "selecting little segments". To be fair, the slot model is also how the canon behaves (fixed-position, additive word clusters; see section 1), so the critique is aesthetic, not grounded in the canon. Report both.
- **Coon's point about nouns and verbs** argues against judging our script by whether it has "words" in the English sense. Judge it by learnability and generalisation (COGS split, iterated learning), and by whether a new reader can learn it from paired examples. Banks' fieldwork method is the blind decipherment test (v3 plan item 5).
- **No source** gives us tense or negation markers in the film's canon. Do not claim fidelity there.

---

## 4. Third-party audit

### FlxB2 / arrival_logograms (MIT, Felix Burk 2018; one commit, May 2024)

**[S] What the code does:** a pure-PIL procedural drawer with no meaning input. `run.py` seeds `random` and calls `logogram((2048,2048), 10, 10, 100, (1,1))`. `logogram.py`:

1. Draws 99 jittered arcs of the ring with random start and end, a gap set by `angle` and `gapWidth = randint(0,90)`, and a segment width of `randint(-10,10)` (`circularStroke.py:49`; negative widths draw nothing).
2. Draws 1–4 clusters, each covering 0.4 rad: 69 random disks, some with a NodeBox-style random-walk `Tendril` (35–50 segments), plus a 19-arc "blob".
3. Scatters 69 small disks around the whole ring.
4. Blurs twice and thresholds at 200.

`index.html` runs the same code in Pyodide. **[I]** Bugs and quirks:
- `generatePNGs` passes an extra `seed` argument and raises a TypeError.
- The `middleAngle` formula does not point at the cluster centre (`logogram.py:61`), so the blobs land away from the disks.
- Nothing is trained, and there is no semantics.
- The output resembles the stackexchange "Where is Abbott" generators: clean circles with spray, nowhere near Bertrand's ink.

**Verdict: skip as a generator. Optionally borrow** it as an "Arrival-looking but meaningless" **distractor class** (MIT, safe), for example as negatives in a discriminator or a decipherment control.

### Selim Şeker / logogram-language-generator (inzva 2020; one commit, Dec 2020)

**[S] What the code does:**
- `configs/bbvae_CompleteRun.yaml`: a β-VAE (Burgess capacity loss "B", γ = 10, C_max = 25, latent 128, 64×64 input, 70 epochs) trained on **Omniglot** (`Omniglot.py`: the 50 Omniglot alphabets). **No Arrival logograms are used anywhere.**
- `logogram_language_generator.py`:
  - loads a fastText `cc.<lang>.128.bin` vector for each word
  - maps es/fr/it to en with UMWE matrices (`umwe_mappers/*.t7`, which are included)
  - standardises using random English embedding samples (the `.pickle` files are included), or uses a learned mapper (`MapperLayer.py`: a 3-layer MLP, `umwe2vae` with a smoothness-minus-contrast loss, or a triplet-loss multilingual mapper in `train_mapper_with_siamese.py`)
  - decodes the vector as a VAE latent, then post-processes it (saturation, gamma, contrast, threshold)
  - can produce word-to-word interpolation GIFs

**[I] Assessment:**
- The VAE checkpoint (`model_checkpoints/final_model_checkpoint.ckpt`), the mapper checkpoints and the fastText bins are **not in the repo**, so nothing runs out of the box.
- It uses deprecated PyTorch Lightning APIs (`pytorch_lightning.logging.TestTubeLogger`).
- Outputs are Omniglot-like glyphs, not rings.
- It encodes words, not sentences, and there is no listener.

**License:** no top-level LICENSE. The vendored `umwe/` is CC BY-NC 4.0 (from Facebook).

**Verdict: skip.** The one idea worth noting is "word embedding mapped into a pretrained drawing latent". That makes semantic similarity become visual similarity by construction: a topographic prior, not emergence. It is the opposite of what we want to test.

### kyegomez / HeptapodLM (MIT; package `nonlinear-transformer`)

**[S] What the code does** (`heptapod/model.py`):
- token embedding
- `depth` blocks of a Python double-loop `Local2DAttention` over a token grid (`:87-110`; O(H·W) Python loops, no projections, q = k = v = the raw features)
- an MLP
- an extra local attention after each block
- `Rotary2DEmbedding` that is **added after all blocks** (`:150`), so attention never sees position
- `to_logits`

`train.py` reshapes 1024 bytes of enwik8 into a 32×32 grid and trains next-token prediction "diagonally" (`at.py:112-115`, labels `x[:,1:,1:]`).

**[I] Bugs:**
- `README` and `example.py` call the model with `embed_size`, `heads` and `iterations`, which are not in the constructor (`:131`), so the example raises a TypeError.
- The rotary embedding concatenates per-axis frequencies, so it only works when H = W, and it is added rather than rotated.
- The blocks ignore `window_size`.
- Local attention with no mask lets a position see its own target, which leaks the label.
- The dataset calls `.cuda()` inside `__getitem__`.
- The README says "please help us train". There are no weights and no results.

**Verdict: skip.** The name is the only Arrival connection.

### Replicate: meghabyte/arrival-logograms

**[S] Model page:**
- "trained on a small dataset of logograms from the movie… not based on any 'real' language… mostly just for fun"
- public; 12,017 runs; version `3d58fdf6…`, created 2024-08-24
- runs on L40S at about $0.0058 per run
- default prompt "A logogram of creativity", `lora_scale` 0.6, `refine: no_refiner`, `high_noise_frac` 0.8, `K_EULER`, guidance 7.5
- the training schema shows a `TOK` token string, `caption_prefix "a photo of TOK"`, CLIPSeg masking and LoRA rank 32

**[I] Base model:** that is the signature of Replicate's **SDXL** fine-tuning cog, so this is an SDXL LoRA (inferred; the page does not name the base).

**Not stated on the page:** the license, the size of the training set, and any weights download. The page offers only API and Docker runs.

**Verdict: skip.**
- It is prompt-to-image style transfer with no compositional semantics.
- The license is unclear, and it was trained on copyrighted stills.
- We already hold better, cleaner ink.
- At most it could serve as a style-only distractor.

### Cuskley's generator

**Verdict: borrow the ideas, not the code.**
- Borrow the growth animation: ring first, then protrusions, then tendrils, then bulbs. It suits our "writing in time" idea.
- Borrow the tendril geometry (inside or outside the ring, radii up to 1.5R) as a sanity range.
- Borrow her "tie params to syntax" footnote as a contrast: a hand-designed baseline mapping of role to protrusion property, to compare against what emerges.
- The code license is unclear, and the code is p5.js.

---

## 5. Arrival-faithful properties to have or test

Each item gives the source, then **[I]** the test.

1. **One ring per complete utterance, at a nearly constant ring size.** Sources: Wikipedia (one logogram per sentence; dictionary of whole utterances); our fits (r/width 0.286–0.371 across all 38). Test: ring radius is invariant to meaning complexity; complexity has to show up in ink and marks.
2. **Composition by reuse.** A compound contains its component's ink at the same place, and new marks are added elsewhere. Source: Wolfram's shared-word `ImageDifference` and nearest-neighbour picks (`arrivalLiveCoding.nb #7–#11`, `#49–#53`); my containment measurement (Weapon ⊂ OfferWeapon 1.00, Time ⊂ ThereIsNoLinearTime 0.98, Earth ⊂ HepPurpEarth 0.95; median over all pairs 0.42). Test: for meaning pairs where one is a subset of the other, containment should be well above the all-pairs median, with an ink-ratio guard.
3. **Meaning lives in angular regions.** Source: Wolfram's 12-wedge sectioning (`sectionBreaking-01.nb #3–#22`), plus the press claim of "repeating patterns" (Wikipedia [1][8]). Test: a 12-wedge attribution map that shows which wedges change when one meaning slot changes, as in Wolfram's straightened-section grid.
4. **Absolute orientation in the canon.** Shared words keep the same absolute angle on the canvas, and Wolfram's sectioning is not rotation-invariant (source: item 2 and `sectionBreaking #5`). **[I]** Our random-rotation listener is a design choice that goes beyond the canon. Report a fixed-orientation condition alongside the rotated one, and do not call rotation invariance "faithful".
5. **Drawing-to-drawing variation for the same meaning.** Source: Human1/Human_1 and OfferWeapon1/OfferWeapon_1 in the 38 files; Cuskley on variation. Test: sample two logograms for the same meaning (dropout or noise) and check that they read the same while differing in pixels.
6. **Ink statistics match the canon.** Source: Wolfram's mean ink 0.104 (`#58`); our `measure.py` gives a median of 0.099 and also measures gap share and cluster counts. Test: generated logograms fall inside the canon's p10–p90 ranges (v2 at about 10 marks per logogram is already flagged as too busy).
7. **Retrievability by generic vision.** Logograms that share a meaning element are nearest neighbours under off-the-shelf image features. Source: Wolfram `FeatureNearest` (`#37–#48`). Test: retrieval@1 of a shared slot using a frozen pretrained encoder, with the canon as the reference score.
8. **Semasiographic, with a visual syntax unrelated to English order.** Source: Cuskley quoting Chiang p. 108. Test: no correlation between angular order and the slot or English word order; COGS structural generalisation.
9. **Non-linear and simultaneous.** Sources: Cuskley ("isn't linear", "all at once"); Chiang via Wikipedia [21]. Test: partial-reveal and occlusion curves (v3 plan), and the linear control at equal capacity.
10. **Learnable by a new reader from paired examples.** Sources: Coon's fieldwork framing; Cuskley's proposed communication game. Test: blind decipherment (v3 plan) and iterated learning.
11. **Not grounded in any source available here:** tense or negation marks, "complexity ∝ size", declension by stroke curvature, subject/object by orientation. The last two are novella-only per Wikipedia [22][23]. Treat them as optional extras, and do not claim them as film-canonical.
