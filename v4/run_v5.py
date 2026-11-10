#!/usr/bin/env python3
"""One Modal job: search at send time on the fresh exam, distill it into the writer,
then the final fresh-exam measurement of the distilled writer (alone and with search)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["HEPTAPOD_OUT"])
INIT = sys.argv[1]
os.symlink(INIT, OUT / "v4_final.pt") if not (OUT / "v4_final.pt").exists() else None


def run(*args):
    print({"event": "run", "cmd": " ".join(args)}, flush=True)
    r = subprocess.run([sys.executable, *args], cwd=HERE.parent)
    print({"event": "exit", "code": r.returncode}, flush=True)
    return r.returncode


SKIP_BASE = "skip-base-eval" in sys.argv
for split in () if SKIP_BASE else ("fresh_natural", "fresh_random"):
    run(str(HERE / "search.py"), "eval", "--ckpt", "v4_final.pt", "--split", split, "--n", "1000", "--batch", "64", "--max-sec", os.environ.get("EVAL_SEC", "600"))
run(str(HERE / "distill.py"), "--init", str(OUT / "v4_final.pt"), "--rounds", os.environ.get("ROUNDS", "2"), "--gen-sec", os.environ.get("GEN_SEC", "1200"),
    "--train-sec", os.environ.get("TRAIN_SEC", "1500"), *(["--targets", os.environ["TARGETS"]] if os.environ.get("TARGETS") else []),
    "--search-rounds", os.environ.get("SEARCH_ROUNDS", "20"), "--gen-batch", os.environ.get("GEN_BATCH", "64"))
if (OUT / "distilled.pt").exists():
    for split in ("fresh_natural", "fresh_random"):
        extra = ["--plain-only"] if os.environ.get("PLAIN_ONLY") else []
        run(str(HERE / "search.py"), "eval", "--ckpt", "distilled.pt", "--split", split, "--n", "1000", "--batch", "64",
            "--max-sec", os.environ.get("EVAL_SEC", "600"), *extra)
print({"event": "all done"}, flush=True)
