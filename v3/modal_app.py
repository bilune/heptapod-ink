"""Run Heptapod Ink v3 on a Modal GPU.

The container gets the same code and data as the local project; each run
executes `python v3/heptapod.py ...` exactly as locally, writing to a Modal
Volume mounted at v3/out/<run name>.

  modal run --detach v3/modal_app.py --name base --args "train --steps 30000"
  modal volume get heptapod-runs base ./v3/runs/modal/base   # fetch results
"""
import shlex
import subprocess

import modal

ROOT = "/root/project"
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "transformers", "matplotlib", "numpy", "pillow", "sentencepiece", "protobuf")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_file("v1/heptapod.py", f"{ROOT}/v1/heptapod.py")
    .add_local_file("v2/heptapod_ink.py", f"{ROOT}/v2/heptapod_ink.py")
    .add_local_file("reference/marks.npz", f"{ROOT}/reference/marks.npz")
    .add_local_file("reference/fragments.npz", f"{ROOT}/reference/fragments.npz")
    .add_local_dir("v3/data", f"{ROOT}/v3/data")
    .add_local_file("v3/heptapod.py", f"{ROOT}/v3/heptapod.py")
    .add_local_file("v3/check_local.py", f"{ROOT}/v3/check_local.py")
    .add_local_dir("v3/data_tatoeba", f"{ROOT}/v3/data_tatoeba")
    .add_local_file("v4/heptapod_t5.py", f"{ROOT}/v4/heptapod_t5.py")
    .add_local_file("v4/search.py", f"{ROOT}/v4/search.py")
    .add_local_file("v4/distill.py", f"{ROOT}/v4/distill.py")
    .add_local_file("v4/run_v5.py", f"{ROOT}/v4/run_v5.py")
    .add_local_file("v4/run_panels.py", f"{ROOT}/v4/run_panels.py")
    .add_local_file("v4/run_s2.py", f"{ROOT}/v4/run_s2.py")
    .add_local_file("v4/fresh_reader.py", f"{ROOT}/v4/fresh_reader.py")
    .add_local_file("v4/run_locked.py", f"{ROOT}/v4/run_locked.py")
    .add_local_file("v4/cliff_test.py", f"{ROOT}/v4/cliff_test.py")
    .add_local_file("v4/LOCKED_EXAM.json", f"{ROOT}/v4/LOCKED_EXAM.json")
)
runs = modal.Volume.from_name("heptapod-runs", create_if_missing=True)
cache = modal.Volume.from_name("heptapod-cache", create_if_missing=True)
app = modal.App("heptapod-field", image=image)


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/runs": runs, "/cache": cache})
def run(name: str, args: str, script: str = "heptapod.py"):
    import os
    # v3/heptapod.py writes to v3/out; point it at this run's folder on the volume
    os.makedirs(f"/runs/{name}", exist_ok=True)
    out = f"{ROOT}/v3/out"
    if not os.path.islink(out):
        os.symlink(f"/runs/{name}", out)
    os.environ["HEPTAPOD_OUT"] = f"/runs/{name}"  # v4 writes here directly
    path = f"{ROOT}/{script}" if "/" in script else f"{ROOT}/v3/{script}"
    toks = shlex.split(args)
    while toks and "=" in toks[0] and toks[0].split("=")[0].isupper():  # leading KEY=VALUE tokens become env vars
        k, v = toks.pop(0).split("=", 1)
        os.environ[k] = v
    cmd = ["python", path, *toks]
    with open(f"/runs/{name}/console.log", "a") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        proc.wait()
    runs.commit()
    cache.commit()
    return proc.returncode


@app.local_entrypoint()
def main(name: str, args: str, script: str = "heptapod.py"):
    print("exit code", run.remote(name, args, script))
