# Heptapod Ink v4: open English in one ring

text → T5-small encoder → 36 slot vectors → one of 81 film-ink marks per slot (per-slot VQ codebooks) → one ring logogram → polar eyes → slot distributions → adapter → T5-small decoder → text.

Trained only on reconstructing the text. No grammar or word meanings are hand-coded. All information is in the marks; the ring line is the same in every slot. Architecture and training stages: [docs/how-it-works.md](../docs/how-it-works.md).

## Current checkpoint: `runs/s2/model.pt`

The first v4 checkpoint (`runs/final`, trained on Tatoeba) was continued on a mixed natural-English pool (FineWeb, Wikipedia, CC-News, Tatoeba; built by `prep_mixed.py`), and its eyes were adapted to the new code. With sender-side rereading (`search.py`), on the locked exam of 2,000 unseen sentences:

| Set | Plain exact | Reread exact | Reread words | Reread, unseen perturbation | Shuffled image |
| --- | --- | --- | --- | --- | --- |
| Natural (UD GUM/PUD/LinES/ParTUT) | 33.5% | **81.0%** | 92.9% | 75.5% | 1.2% |
| Random word combinations | 4.1% | 36.8% | 61.3% | 29.8% | 0.6% |

Protocol, breakdowns and probes: [docs/results.md](../docs/results.md). Raw files: `runs/exam/` (exam), `runs/s2/` (probes, figures).

## Earlier checkpoint: `runs/final` (Tatoeba test, run once after all dev decisions)

| Set | Reads alone, exact | Reads alone, words | Read and verify, exact | Read and verify, words | Shuffled-image control, exact |
| --- | --- | --- | --- | --- | --- |
| Natural sentences (3,000) | **62.7%** | **88.2%** | **88.2%** | 94.1% | 0.0% |
| Random word combinations (1,000) | 9.2% | **44.1%** | 29.0% | 48.3% | 0.4% |

- The eyes recognise 99.1% of the marks on the writer's rings (98.1% on random combinations).
- Read and verify: the reader proposes 32 candidates, the frozen writer redraws each, and the candidate whose ring matches the received pixels wins. The verifier chose a wrong sentence in 2.1% (natural) and 2.0% (random) of cases. The correct sentence was among the candidates in 90.1% / 30.0%. Sender-side rereading later replaced this.
- PNG round trip (`say` then `read`, 16-bit grayscale, 4× pixel blocks): 400/400 identical to the in-memory reading on dev.
- Weak spot: random word combinations. The ring keeps word shape but swaps rare words for similar ones ("marika my rest what advocate mary" → "maryland my stay what suggest mary"). On natural dev, 22% of sentences have another candidate that produces the identical ring: the code is near its capacity.
- On web text (UD EWT) this checkpoint read only 21% exact, which motivated rereading and the new training data.

## How we got here

| Step | Result (dev) |
| --- | --- |
| Categorical 82-way argmax-STE | information lived in soft probabilities; random combos ~0.5% words |
| FSQ 4×{-1,0,1} and a continuous 4-number control | both ~0% on random combos: 4 numbers per slot is too narrow to learn through |
| Continuous 256-wide slots | 99% exact after 4,000 H100 steps |
| + per-slot VQ (k-means init, EMA, commitment .25, dead-entry reseed) | 50% exact natural, 33% random words after 15 min |
| + per-slot batch-entropy term (collapsed slots revived) | 60% exact, 43% random words on paper |
| Stage 2: eyes trained on the frozen code, then text through the frozen symbolic reader | 64% exact natural, 45% random words from the image |
| Sender-side rereading (`search.py`) | EWT web dev 21% → 67% exact |
| Stage 1 continued on mixed data + eyes adapted (`runs/s2`) | EWT web dev 42% plain, 80% reread |

Stage 0 measures the eyes' confusions on random rings (91% per mark) and keeps the 81 most readable marks; stage 1 trains through that confusion channel.

## Commands

Run from the repo root. `HEPTAPOD_OUT` points at the run folder that holds `model.pt`.

```bash
HEPTAPOD_OUT=v4/runs/s2 python v4/heptapod_t5.py say "I left my keys at home."
HEPTAPOD_OUT=v4/runs/s2 python v4/heptapod_t5.py read <png>
HEPTAPOD_OUT=v4/runs/s2 python v4/search.py eval --split dev --n 300        # plain vs reread, PNG round trip
HEPTAPOD_OUT=v4/runs/s2 python v4/check_v4.py png --n 400
python v4/heptapod_t5.py train --smoke                                       # tiny local run of every stage
python3 v4/live.py                                                           # live view of a run's logs (localhost:8791; logs in $HEPTAPOD_LIVE_LOGS, default v4/runs/live_logs/)
modal run --detach v3/modal_app.py --name <run> --script v4/heptapod_t5.py --args "train --quant vq ..."
```

Use `HEPTAPOD_DEVICE=cpu` to force the CPU and `HEPTAPOD_CHUNK=32` to keep rereading light on a laptop. Full training needs a GPU (it ran on a rented H100 through Modal) and the datasets (`scripts/fetch_data.sh`).

## Files

| File | What |
| --- | --- |
| `heptapod_t5.py` | Model, staged training, `say` / `read` / `eval` |
| `search.py` | Sender-side rereading |
| `prep_mixed.py` | Mixed natural-English training pool |
| `locked_exam.py`, `run_locked.py`, `LOCKED_EXAM.json` | Locked exam: build, fingerprints, single evaluation |
| `fresh_reader.py`, `fresh_exam.py` | A new reader learning the frozen code |
| `probe_structure.py`, `anatomy.py`, `zones_causal.py`, `content_probe.py`, `grammar_probe.py` | What the slots do |
| `semantic_probe.py`, `analogy_probe.py`, `visual_analogy.py` | Meaning, analogies |
| `cliff_test.py` | Follow-up on long, subword-heavy sentences |
| `distill.py` | Attempts to distill rereading into the writer (did not work) |
| `export_web.py`, `export_search.py`, `export_web2.py` | ONNX export for [web/](../web/README.md) |
| `check_v4.py`, `references.py` | Read-and-verify, PNG round trip, classical reference codec |
| `live*.py`, `run_*.py` | Live dashboards and job wrappers for Modal runs |

Checkpoints are not in git; see the [top-level README](../README.md#weights). Only `runs/s2/model.pt` is published.

## Next, if more capacity is wanted

The loss sits in the continuous → marks step (99% → ~65%), not in vision (99% marks read). Options: longer stage 1, a larger alphabet (e.g. each mark inside or outside the ring: 162 symbols), or two rings per sentence.
