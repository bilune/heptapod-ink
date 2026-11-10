# Debate 4, round 2: Gonzalo changes the target

Read research/debate4/context.md, your expert_codex.md, and research/debate4/step1_results.md (step 1 is done: results, references, fixes).

New objective from Gonzalo: not random/nonsense word combinations. The target is MEANINGFUL but UNCOMMON English: real web/news/dialogue sentences with names, numbers, casing, punctuation, varied topics (the EWT-like distribution), up to 12 words. Failing on nonsense is an accepted limitation; relying on the reader's knowledge of English is by design, as long as the ring carries what is not predictable.

Current state on EWT (dev panel, 256): plain writer 20.7% exact / 61.7% words; with sender search 66.8% / 84.7% (perturbed 61.7%); classical reference (Qwen2.5-1.5B arithmetic coding + RS(36,28) through the same eyes) ~98% clean / ~94% perturbed. Tatoeba: 68.8% plain, 96.5% search. Training so far used only Tatoeba (1.88M short simple sentences) + 50% random combos in stage 1 (30% in stage 2).

Constraints unchanged (trained models only, one ring, $25 cumulative budget of which ~$1.6 is spent; Modal H100 at $0.001097/s; M4 for tiny smoke tests only; stall watchdog; locked exam = UD GUM/PUD/LinES/ParTUT natural 2,000 + random sets, fingerprinted in v4/LOCKED_EXAM.json, evaluate once at the end; EWT is now a dev set).

Use web search as needed (licences and availability of diverse short-English corpora: FineWeb/FineWeb-Edu, C4, Wikipedia, OpenSubtitles/dialogue, news; domain adaptation for discrete codes).

Write research/debate4/round2_codex.md (<=1,000 words), acting as the same expert:
1. Is the retarget sound, and what would now surprise a reviewer (targets on EWT-like text, plain and with search, robustness, fresh-reader learnability)?
2. The concrete next experiment(s), replacing the old order: training data mix (sources, filtering to <=12 words, dedup against all eval sets incl. UD and Tatoeba, size), what to continue vs retrain (stage-1 VQ continuation from the v4 checkpoint vs fresh warm-up + VQ; eyes adaptation), hyperparameters, H100 minutes, local smoke tests, pass/fail rules on EWT dev panels, and whether rotation trick/SimVQ belong in this run.
3. Updated budget split for the remaining ~$23 and what to drop.
Be concrete.
