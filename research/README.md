# research/

The lab notebook. Nothing here is needed to run the models; it is the record of how decisions
were made, kept mostly as written at the time (including dead ends and numbers that were later
corrected).

## Where to start

- [debate4/final_results.md](debate4/final_results.md): the locked exam and the probes, in one page.
- [SYNTHESIS.md](SYNTHESIS.md): an early review of the literature and of what was wrong with the
  v2 measurements.
- [presentation_plan.md](presentation_plan.md): how prior work compares, and the claims we think are
  honest to make (and which ones not to make).

A cleaned-up version of all this is in [docs/](../docs/).

## What is in each folder

| Path | What |
| --- | --- |
| `notes/` | Long literature notes: emergent-communication methodology, the film's script and what can be measured in it, drawing games, meaning spaces and a survey of emergent-language papers |
| `debate/` | v3: why the reader ignored the image, and the writer-freezing fix |
| `debate2/` | Moving from COGS to open English: which pretrained models to reuse (led to v4 on T5) |
| `debate3/` | Raising v4's accuracy: sender-side search, and why distilling it into the writer failed |
| `debate4/` | From "it works" to a locked exam and probes: new target and data, the exam, the subword cliff, semantic errors, relation to T5's geometry. Also the scripts and raw results of some of these analyses (`round5_*`) |

## About the debates

Most design decisions were argued out between two AI assistants: Claude (Anthropic), which wrote
most of the code and ran the experiments, and Codex (OpenAI), prompted to act as an outside expert
reviewer. Each round has a prompt file (`*_prompt.md`, what was asked, with the evidence at that
point) and an answer file (`*_codex.md` or `*_claude.md`). Gonzalo set the goals, the budget and
the constraints, and picked what to try. The reviewer's suggestions were often wrong or too
expensive, and sometimes caught real mistakes; read them as a notebook, not as conclusions.

## Not committed

Papers, cloned repositories (COGS, ReCOGS, EGG, the Wolfram Arrival repo and other Arrival-related
projects) and datasets (Tatoeba, UD treebanks) were kept locally in `papers/`, `repos/`, `tatoeba/`,
`ud_ewt/` and `ud_locked/`. They are not in git. [scripts/fetch_data.sh](../scripts/fetch_data.sh)
downloads the ones the code actually reads into the same paths; the notes cite the papers with
links.
