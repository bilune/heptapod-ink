#!/usr/bin/env python3
"""The locked final exam: verify fingerprints, evaluate one split once, refuse to repeat."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["HEPTAPOD_OUT"])
ckpt, split = sys.argv[1], sys.argv[2]
lock = json.loads((HERE / "LOCKED_EXAM.json").read_text())
data = (HERE.parent / "v3" / "data_tatoeba" / f"{split}.json").read_text()
assert hashlib.sha256(data.encode()).hexdigest() == lock[split]["sha256"], "locked file changed"
marker = OUT / f"LOCKED_DONE_{split}"
assert not marker.exists(), f"{split} already evaluated once"
marker.write_text("started")
if not (OUT / "exam.pt").exists():
    os.symlink(ckpt, OUT / "exam.pt")
print({"event": "fingerprint ok", "split": split, "n": lock[split]["n"]}, flush=True)
r = subprocess.run([sys.executable, str(HERE / "search.py"), "eval", "--ckpt", "exam.pt", "--split", split,
                    "--n", str(lock[split]["n"]), "--batch", "64", "--seed", "0"], cwd=HERE.parent)
marker.write_text(f"done exit {r.returncode}")
print({"event": "all done", "code": r.returncode}, flush=True)
