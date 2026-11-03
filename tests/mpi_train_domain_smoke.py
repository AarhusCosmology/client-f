"""The training domain, the region's quantile reach and the audit's posterior pilot through
hpc/, two ranks (acquisition.boost.train_domain, region_quantile; acquisition.audit.posterior).

Run by itself: it starts mpiexec. What a test of the settings on the cluster does, on the
analytic two-parameter Gaussian: a run through the launcher (hpc/launch.py) into
its boosts, a branch at the boosts' start with the settings (analysis/branch_run.py --set
acquisition.boost.KEY=VALUE), the branch continued through the launcher, and continued again.
Checked: the run without the settings has no training domain; the branch builds it at its
first training after the start, commits it with that training and reads it at every later
one, also after a restart; it reports the share of the region outside the domain and the
quantile reach.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "hpc"
MPI = [shutil.which("mpiexec") or "mpiexec", *shlex.split(os.environ.get("CLIENT_TEST_MPI_ARGS", "")), "-n", "2"]
TAG = f"smoke{os.getpid()}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", CLIENT_REPO=str(ROOT), SLURM_JOB_ID=TAG)


def run(command, cwd, timeout=600):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, env=ENV)
    if result.returncode:
        raise AssertionError(f"exit {result.returncode}: {command}\n{result.stdout[-6000:]}")
    return result.stdout


def launch(cwd, *arguments):
    return run([*MPI, sys.executable, str(KIT / "launch.py"), "domain_bounds.json", *map(str, arguments)], cwd)


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    (d / "domain_bounds.json").write_text(json.dumps(dict(x=[-5, 5], y=[-5, 5])))
    config = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    config["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    config["acquisition"].update(
        audit=dict(chain=True, starts=16, steps=60, epochs=2, rounds=1, k=2),
        boost=dict(n=24, start_iteration=2, walkers=32, cool_steps=50, steps=200, components=1,
                   background="quadratic", region=2.0))
    config["convergence"]["max_iterations"] = 8
    (d / "client.yaml").write_text(yaml.safe_dump(config))

    said = launch(d, "client.yaml", "-n", "smoke", "-o", "results", "-i", 3)       # networks 0 to 2
    (base,) = (d / "results").iterdir()
    assert "Training domain" not in said and not list(base.glob("boost/train_domain_it_*")), said[-3000:]

    said = run([sys.executable, str(KIT / "analysis" / "branch_run.py"), str(base), "--at", "2",
                "--set", "acquisition.boost.train_domain=1.0", "--set", "acquisition.boost.region=1.5",
                "--set", "acquisition.boost.region_quantile=0.99", "--set", "acquisition.audit.posterior=6",
                "--set", "acquisition.audit.posterior_tolerance=1.0e-6"], d)
    (branch,) = [p for p in (d / "results").iterdir() if p != base]
    settings = yaml.safe_load((branch / "client.yaml").read_text())["acquisition"]["boost"]
    assert (settings["train_domain"], settings["region"], settings["region_quantile"]) == (1.0, 1.5, 0.99), settings

    said = launch(d, branch, "-i", 2)                                              # networks 3 and 4
    print("\n".join(line for line in said.splitlines() if "Training domain" in line or "Region:" in line
                     or "outside the training domain" in line))
    assert "Training domain (train_domain_it_3.npz, built from this training set)" in said, said[-4000:]
    assert said.count("Training domain (train_domain_it_3.npz): ") == 1, said[-4000:]        # network 4 reads it
    assert "of the region volume (to its edge, inside the box) lies outside the training domain" in said, said[-4000:]
    assert "volume-weight ESS" in said, said[-4000:]
    assert "(the 0.99 quantile of its points' distances)" in said, said[-4000:]
    assert said.count("posterior pilot 0: 6 cooled T = 1 walkers evaluated") == 2, said[-4000:]     # networks 3 and 4
    committed = json.loads((branch / "phases" / "training_it_3.json").read_text())["artifacts"]
    assert "boost/train_domain_it_3.npz" in committed, committed
    for iteration in (3, 4):
        report = json.loads((branch / "audit" / f"audit_it_{iteration}.json").read_text())
        if report["posterior"]["tuned"]:
            assert report["final_search"]["stage"] == "post_pilot", report
            assert report["highest_excess"] == report["final_search"]["highest_excess"], report
            assert report["clear"] == (report["final_search"]["distinct_above_tau"] == 0), report

    said = launch(d, branch, "-i", 1)                                              # network 5, after a restart
    assert "Training domain (train_domain_it_3.npz): " in said and "built from" not in said, said[-4000:]
    assert [p.name for p in branch.glob("boost/train_domain_it_*")] == ["train_domain_it_3.npz"]
    # The audit's batches go through the launcher as well: its journal counts them.
    journal = KIT / "logs" / f"evaluations_{TAG}.jsonl"
    batches = [json.loads(line) for line in journal.read_text().splitlines()]
    assert sum(b["points"] for b in batches if b["kind"] == "audit") > 0, batches
    for kept in (journal, KIT / "logs" / f"backend_{TAG}.json"):
        kept.unlink(missing_ok=True)
    print("mpi_train_domain_smoke: passed")
