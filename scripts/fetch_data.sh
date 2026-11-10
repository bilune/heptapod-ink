#!/usr/bin/env bash
# Fetch the external inputs the code expects, into the paths the code reads.
#
#   bash scripts/fetch_data.sh            # everything below (about 150 MB)
#   bash scripts/fetch_data.sh logograms  # only some parts: logograms repos tatoeba ud
#
# What goes where:
#   reference/logograms/*.jpg   38 logograms from WolframResearch/Arrival-Movie-Live-Coding
#                               (ScriptLogoJpegs/, list in reference/urls.txt; CC BY-NC 4.0, see reference/COPYING.md)
#   research/repos/COGS         najoungkim/COGS, read by v3/cogs_graphs.py (research/repos/COGS/data/*.tsv)
#   research/repos/ReCOGS       frankaging/ReCOGS, background reading only (no code imports it)
#   research/repos/Arrival-Movie-Live-Coding   full Wolfram repo, optional (FULL_ARRIVAL=1)
#   research/tatoeba/eng_sentences.tsv         Tatoeba English export (CC BY 2.0 FR)
#   research/ud_ewt/*.conllu    UD English-EWT dev+test (fresh panel, v4/fresh_exam.py)
#   research/ud_locked/*.conllu UD English GUM, PUD, LinES, ParTUT (locked exam, v4/locked_exam.py)
#
# Pins: the git repos are checked out at the commits this project used. Tatoeba exports change
# weekly and UD `master` moves with each UD release, so rebuilt splits will not be byte-identical
# to the ones the published checkpoint was trained/evaluated on. Override UD_REF (e.g. r2.16) to pin UD.
#
# Nothing heavy runs here. The derived datasets are rebuilt with the commands printed at the end.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PARTS="${*:-logograms repos tatoeba ud}"
UD_REF="${UD_REF:-master}"

want() { [[ " $PARTS " == *" $1 "* ]]; }

clone_at() {  # clone_at <url> <dir> <commit>
  local url="$1" dir="$2" rev="$3"
  if [ -d "$dir/.git" ]; then
    echo "skip $dir (already cloned)"
  else
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" checkout --quiet "$rev"
  echo "ok   $dir @ ${rev:0:7}"
}

if want logograms; then
  mkdir -p reference/logograms
  while read -r url; do
    [ -z "$url" ] && continue
    f="reference/logograms/$(basename "$url")"
    [ -s "$f" ] || curl -fsSL --retry 3 -o "$f" "$url"
  done < reference/urls.txt
  echo "ok   reference/logograms ($(ls reference/logograms | wc -l | tr -d ' ') files)"
fi

if want repos; then
  mkdir -p research/repos
  clone_at https://github.com/najoungkim/COGS.git   research/repos/COGS   165a7b669eade971fa47bf568a2e51925360fed8
  clone_at https://github.com/frankaging/ReCOGS.git research/repos/ReCOGS 1b6eca8ff4dca5fd2fb284a7d470998af5083beb
  if [ -n "${FULL_ARRIVAL:-}" ]; then
    clone_at https://github.com/WolframResearch/Arrival-Movie-Live-Coding.git \
      research/repos/Arrival-Movie-Live-Coding 45f40667bad53ed5dc9489d31b3b2e2c7bea4aa0
  fi
fi

if want tatoeba; then
  mkdir -p research/tatoeba
  if [ ! -s research/tatoeba/eng_sentences.tsv ]; then
    curl -fL --retry 3 -o research/tatoeba/eng_sentences.tsv.bz2 \
      https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
    bunzip2 -kf research/tatoeba/eng_sentences.tsv.bz2
  fi
  echo "ok   research/tatoeba/eng_sentences.tsv"
fi

if want ud; then
  ud() {  # ud <repo> <file> <dest dir>
    mkdir -p "$3"
    [ -s "$3/$2" ] || curl -fsSL --retry 3 -o "$3/$2" \
      "https://raw.githubusercontent.com/UniversalDependencies/$1/$UD_REF/$2"
    echo "ok   $3/$2"
  }
  ud UD_English-EWT    en_ewt-ud-dev.conllu     research/ud_ewt
  ud UD_English-EWT    en_ewt-ud-test.conllu    research/ud_ewt
  ud UD_English-GUM    en_gum-ud-dev.conllu     research/ud_locked
  ud UD_English-GUM    en_gum-ud-test.conllu    research/ud_locked
  ud UD_English-PUD    en_pud-ud-test.conllu    research/ud_locked
  ud UD_English-LinES  en_lines-ud-dev.conllu   research/ud_locked
  ud UD_English-LinES  en_lines-ud-test.conllu  research/ud_locked
  ud UD_English-ParTUT en_partut-ud-dev.conllu  research/ud_locked
  ud UD_English-ParTUT en_partut-ud-test.conllu research/ud_locked
fi

cat <<'MSG'

Inputs fetched. Derived data is rebuilt from the repo root with (order matters, some steps are heavy):

  # ink library (only if you change it: reference/fragments.npz and reference/marks.npz ship with the repo)
  .venv/bin/python reference/prep_library.py   # logograms -> fragments.npz, fragments_index.json
  .venv/bin/python reference/prep_marks.py     # fragments -> marks.npz, marks_index.json
  .venv/bin/python reference/measure.py        # reference_stats.json

  # v3 COGS meaning graphs (v3 only)
  .venv/bin/python v3/cogs_graphs.py && .venv/bin/python v3/augment.py && .venv/bin/python v3/embed_lemmas.py

  # Tatoeba splits used by v3 and v4 -> v3/data_tatoeba/{train,dev,test,*_random,random_vocab}.json
  .venv/bin/python v3/prep_tatoeba.py
  # v4 evaluation panels -> v3/data_tatoeba/{fresh_*,locked_*,dev_uniform10}.json
  .venv/bin/python v4/fresh_exam.py
  .venv/bin/python v4/locked_exam.py           # also rewrites the v4/LOCKED_EXAM.json fingerprint
  # v4 mixed training pool -> v3/data_tatoeba/mixed_{train,val}.json, v4/mixed_data_report.json
  # streams HuggingFaceFW/fineweb (sample-10BT), wikimedia/wikipedia (20231101.en) and
  # vblagoje/cc_news from the Hugging Face Hub with `datasets` (streaming, nothing stored in full),
  # plus a Tatoeba replay; excludes every UD/Tatoeba/evaluation sentence (see its docstring)
  .venv/bin/python v4/prep_mixed.py --total 500000
MSG
