"""Print one line per new evaluation (dev exact) or final result in the Modal runs."""
import re, subprocess, sys, time, ast
from pathlib import Path

runs = {}
for f in Path("v3/runs").glob("modal_*.launch.log"):
    m = re.search(r"ap-[A-Za-z0-9]+", f.read_text())
    if m:
        runs[f.stem.replace("modal_", "").replace(".launch", "")] = m.group(0)
seen = {n: set() for n in runs}
finished = set()
while len(finished) < len(runs):
    for name, app in runs.items():
        if name in finished:
            continue
        try:
            out = subprocess.run(["modal", "app", "logs", app], capture_output=True, text=True, timeout=20).stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        for line in out.splitlines():
            if "dev_exact" in line and "'step'" in line:
                try:
                    row = ast.literal_eval(line[line.index("{"):])
                except Exception:
                    continue
                if row["step"] not in seen[name]:
                    seen[name].add(row["step"])
                    phase = "whole pieces" if row["hard"] else "blending"
                    print(f"{name}: step {row['step']} ({phase}) -> unseen sentences exact {row['dev_exact']*100:.1f}%", flush=True)
            if "exit code" in line or '"gen"' in line:
                finished.add(name)
                print(f"{name}: done", flush=True)
    time.sleep(180)
