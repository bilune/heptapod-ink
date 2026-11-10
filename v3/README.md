# Heptapod Ink v3: a model that writes and reads Heptapod-style logograms

text → frozen MiniLM → **writer** → one ring logogram in the film's ink (24 slots, real marks cut from the 38 Arrival logograms, fixed orientation) → **reader** (polar eyes + text decoder) → text.

It is trained only on reconstructing the text. Grammar and meaning are not hand-coded.

## Final results (COGS IID test, 2,858 sentences never used for any decision)

| Reader | Reads alone | Read and verify (32 candidates) | Correct sentence among candidates |
| --- | --- | --- | --- |
| Original reader (run 3) | 77.1% | 92.4% | 93.1% |
| **Fresh reader, trained from scratch on the frozen code (run 4)** | **78.3%** | **93.2%** | 94.0% |

The 95% interval is about ±1.5 points for "alone" and ±0.9 points for "verified". The verifier picked a wrong sentence in 1.0% of cases.

- **Read and verify:** the reader proposes 32 candidates (beam search); the frozen writer redraws each one, and the candidate whose drawing matches the received logogram wins. It uses the writer at read time, so it is reported separately from "reads alone".
- **A fresh reader learns the language as well as the reader that co-invented it**, and slightly better. The script is learnable from paired examples, not a private code.
- COGS out-of-distribution generalisation categories remain near 0%. The script is not systematic in that sense.
- Scope is COGS only: 645 lemmas, simple past-tense sentences. Arbitrary English does not work (try `heptapod.py say`); v4 moved to T5 for that reason.

## How we got here (all runs on Modal H100)

| Step | Change | Dev exact |
| --- | --- | --- |
| pilots | plain CNN reader, decoder ignored the image | ~0% |
| + word dropout, bag-of-words loss, polar eyes, condensed marks | the reader can see the ring | – |
| base | soft→hard straight-through Gumbel, 30k steps | 44% |
| freeze25 | freeze the writer when the soft phase ends; train only the reader | 71% |
| r2_aug | reader on train-only noun/name substitution augmentation | 76% |
| r3_aug_verbs | + verb swaps (same frame and form), more steps | 77.5% |
| r4_fresh_reader | brand-new reader from scratch on the frozen code | 78.7% |

Key diagnostics: zero code collisions; errors concentrate on rare words (93% error for words seen 1–9 times in train); the writer spells rare and frequent words about equally consistently, so the gap was on the reader's side, which augmentation fixed.

## Commands

```bash
python v3/heptapod.py train --steps 30000 --freeze-writer --mark-scale 0.25      # writer + reader
python v3/heptapod.py train-reader --init <ckpt> --aug --steps 20000              # reader only
python v3/check_local.py <ckpt_dir> verify --split dev --beam 32                   # read and verify
python v3/augment.py --max 150000 --target 200 --verbs 60000                      # practice sentences
modal run --detach v3/modal_app.py --name <run> --args "<heptapod.py args>"   # same on an H100
python3 v3/dashboard.py                                                                     # live dashboard
```

`modal_app.py` runs any script on a rented H100 through [Modal](https://modal.com) (needs your own account; `requirements-modal.txt`). The v3 checkpoints are not published; only v4's are.
`PLAN.md` and `CODEX_REVIEW.md` are working notes from this stage.

## Data note

COGS and Tatoeba are not in git; `scripts/fetch_data.sh` downloads them into `v3/data/` and `v3/data_tatoeba/`.

The augmentation uses the COGS meaning graphs to know which words are nouns, names or verbs, and which roles and frames each was seen in, all from train only. This is information used to build practice data, not a rule inside the model. Variants that match any held-out sentence or meaning are dropped.
