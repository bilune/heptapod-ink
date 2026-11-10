#!/usr/bin/env python3
"""Modal job: adapt the eyes to the mixed-data code (stage 2), then paired search panels."""
import os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
init = sys.argv[1]
def run(*a):
    print({"event": "run", "cmd": " ".join(a)}, flush=True)
    r = subprocess.run([sys.executable, *a], cwd=HERE.parent)
    print({"event": "exit", "code": r.returncode}, flush=True)
run(str(HERE / "heptapod_t5.py"), "train", "--quant", "vq", "--resume", init, "--pool", "mixed_train", "--s2-random", "0",
    "--s2-sec", os.environ.get("S2_SEC", "900"), "--s2-steps", "200000", "--lr-eyes", "2e-4", "--text-after", "3000",
    "--decoder-after", "3000", "--dev-sets", "fresh_natural dev", "--select", "exact_cased", "--eval-n", "256", "--eval-every", "1000")
for split, panel in (("fresh_natural", "A"), ("fresh_natural", "B"), ("dev", "A")):
    run(str(HERE / "search.py"), "eval", "--ckpt", "model.pt", "--split", split, "--panel", panel, "--n", "256", "--batch", "64", "--seed", "0")
print({"event": "all done"}, flush=True)
