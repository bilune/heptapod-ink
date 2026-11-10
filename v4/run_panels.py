#!/usr/bin/env python3
"""Modal job: paired search evaluation (writer as is vs writer rereads) on development panels."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["HEPTAPOD_OUT"])
ckpt = sys.argv[1]
if not (OUT / "base.pt").exists():
    os.symlink(ckpt, OUT / "base.pt")
n = os.environ.get("PANEL_N", "256")
for split in os.environ.get("SPLITS", "dev dev_random dev_uniform10 fresh_natural fresh_random").split():
    print({"event": "run", "split": split}, flush=True)
    r = subprocess.run([sys.executable, str(HERE / "search.py"), "eval", "--ckpt", "base.pt", "--split", split,
                        "--n", n, "--batch", "64", "--seed", "0", *sys.argv[2:]], cwd=HERE.parent)
    print({"event": "exit", "split": split, "code": r.returncode}, flush=True)
print({"event": "all done"}, flush=True)
