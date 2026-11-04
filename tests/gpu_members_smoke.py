"""training.members on GPUs (utils/devices.py), for a node with one GPU or more. On the analytic
Gaussian example with boosts from network 1, so that networks 2 and 3 have members:

1. member_device auto: rank 0 and every member train on a GPU, those of a node in turn over its
   GPUs (both used when there are two), and none fails (memory growth: no process reserves a GPU
   whole);
2. the same with the pilot: the members' posterior chains run on their GPUs as well;
3. member_device cpu: the members train on the CPU, rank 0 still on its GPU.

It prints the members' training seconds of every run, and exits 1 when no GPU is visible.

    python tests/gpu_members_smoke.py [--ranks 4] [--members 4]     (it starts mpiexec; CLIENT_TEST_MPI_ARGS adds flags)
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MPI = [shutil.which("mpiexec") or "mpiexec", *shlex.split(os.environ.get("CLIENT_TEST_MPI_ARGS", ""))]
BOOST = dict(n=24, start_iteration=1, walkers=32, cool_steps=50, steps=200, components=1)
PILOT = dict(member_pilot=12, member_pilot_steps=200, member_pilot_walkers=16)
NETWORKS = (2, 3)                    # those trained on boost points


def gpus():
    """The GPUs this job sees, by nvidia-smi (TensorFlow would start CUDA in this process)."""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    found = len(re.findall(r"^GPU \d+:", listing, re.M))
    return min(found, len([v for v in visible.split(",") if v.strip()])) if visible else found


def config(d, name, members, **training):
    cfg = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    cfg["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    cfg["training"].update(n_epochs=40, members=members, member_threads=1, **training)
    cfg["acquisition"]["boost"] = dict(BOOST)
    cfg["convergence"]["max_iterations"] = 4
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run(arguments, ranks):
    result = subprocess.run([*MPI, "-n", str(ranks), sys.executable, str(ROOT / "client.py"), *map(str, arguments)],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                     TF_NUM_INTRAOP_THREADS="2", TF_CPP_MIN_LOG_LEVEL="2"))
    if result.returncode != 0:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout[-6000:]}")
    return result.stdout


def records(directory):
    (run_dir,) = Path(directory).iterdir()
    return [json.loads((run_dir / "trained_models/members" / f"members_it_{n}.json").read_text()) for n in NETWORKS]


def check(recs, log, members_on, label):
    rank0 = re.search(r"TensorFlow on rank 0: (.*)", log)
    assert rank0 and rank0.group(1).startswith("GPU"), (label, rank0 and rank0.group(1))
    assert "members trained on:" in log, label
    used = set()
    for rec in recs:
        devices = {m["member"]: m.get("device") for m in rec["members"]}
        assert all(m.get("error") is None for m in rec["members"]), (label, rec["members"])
        assert str(devices[0]).startswith("GPU"), (label, devices)
        others = [d for k, d in devices.items() if k]
        assert all(str(d).startswith(members_on) for d in others), (label, devices)
        used |= {d.split(" (")[0] for d in devices.values() if str(d).startswith("GPU")}
        seconds = [round(m.get("seconds", float("nan")), 1) for m in rec["members"] if m["member"]]
        print(f"   {label}, network {rec['iteration']}: {devices}; members' training seconds {seconds}")
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--members", type=int, default=4)
    args = ap.parse_args()
    n_gpus = gpus()
    if not n_gpus:
        raise SystemExit("gpu members smoke: no GPU visible (nvidia-smi -L); run it on a GPU node")
    print(f"{n_gpus} GPU(s) visible; {args.ranks} ranks, {args.members} members")
    with tempfile.TemporaryDirectory() as temporary:
        d = Path(temporary)
        log = run([config(d, "auto", args.members), "-i", 4, "-o", d / "auto"], args.ranks)
        used = check(records(d / "auto"), log, "GPU", "1. auto")
        if n_gpus > 1 and args.members > 1:
            assert len(used) > 1, f"two GPUs or more, and the members used only {used}"
        print(f"1. member_device auto: rank 0 and the members on {sorted(used)}")

        log = run([config(d, "pilot", args.members, **PILOT), "-i", 4, "-o", d / "pilot"], args.ranks)
        check(records(d / "pilot"), log, "GPU", "2. pilot")
        assert "Members' pilot errors:" in log
        print("2. the pilot on the GPUs: chosen, no member failed")

        log = run([config(d, "cpu", args.members, member_device="cpu"), "-i", 4, "-o", d / "cpu"], args.ranks)
        check(records(d / "cpu"), log, "CPU", "3. cpu")
        print("3. member_device cpu: the members on the CPU, rank 0 on its GPU")
    print("gpu members smoke: OK")


if __name__ == "__main__":
    main()
