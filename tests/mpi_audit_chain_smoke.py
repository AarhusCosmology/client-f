"""The audit's chain search in the main loop under MPI, two ranks (acquisition.audit.chain).

Run by tests/run_mpi_smokes.py, or by itself: it starts mpiexec. In a boost iteration a round of
the audit whose own climbs flag nothing runs the network's tempered chain on the master, while
the workers wait in the audit's collective, and looks where its cooled walkers climb to. When
the audit then changed nothing, the sampling step takes that chain and does not run it again.
The unit tests script the search; this is the loop with its broadcasts, twice: with a threshold
no maximum reaches (the chain is reused), and with one every maximum exceeds (the chain's
summits are evaluated on both ranks and the chain is run again for the changed network).
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
LAUNCHER = [shutil.which("mpiexec") or "mpiexec", *shlex.split(os.environ.get("CLIENT_TEST_MPI_ARGS", "")), "-n", "2", sys.executable]
REUSED = "Sampling: the chain of the audit's last look is this network's; it is not run again"


def run(arguments, timeout=300):
    result = subprocess.run([*LAUNCHER, str(ROOT / "client.py"), *map(str, arguments)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
    if result.returncode:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout[-6000:]}")
    return result.stdout


def settings(directory, name, **audit):
    config = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    config["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    config["acquisition"].update(
        audit=dict(chain=True, starts=16, steps=60, epochs=2, rounds=1, k=2, **audit),
        boost=dict(n=24, start_iteration=1, walkers=32, cool_steps=50, steps=200, components=1))
    file = directory / f"{name}.yaml"
    file.write_text(yaml.safe_dump(config))
    return file


def searches(run_directory, iteration):
    report = json.loads((run_directory / "audit" / f"audit_it_{iteration}.json").read_text())
    return [r["search"] for r in report["rounds"]], report


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)

    # Nothing is ever flagged: the chain search runs once in each boost iteration, finds
    # nothing, and its chain is the sampling step's.
    said = run([settings(d, "quiet", tau=1.0e6), "-i", 3, "-o", d / "quiet"])
    (quiet,) = (d / "quiet").iterdir()
    assert "(chain search)" in said and said.count(REUSED) == 2, said[-4000:]
    assert searches(quiet, 0)[0] == ["full"], searches(quiet, 0)          # network 0: before the boosts, no chain search
    for iteration in (1, 2):
        kinds, report = searches(quiet, iteration)
        assert kinds == ["chain"] and report["true_calls"] == 0 and report["clear"], (iteration, kinds, report["true_calls"])
    assert (quiet / "boost" / "boost_it_1.json").exists() and (quiet / "trained_models" / "model_it_2.keras").exists()

    # Every maximum is flagged: the climbs use their round, the chain search has its own, its
    # summits are evaluated by both ranks, and the network it sampled is not the final one.
    said = run([settings(d, "loud", tau=1.0e-6), "-i", 3, "-o", d / "loud"])
    (loud,) = (d / "loud").iterdir()
    for iteration in (1, 2):
        kinds, report = searches(loud, iteration)
        assert report["true_calls"] > 0 and len(kinds) >= 2, (iteration, kinds, report["true_calls"])
        assert report["call_budget"] == 4, report["call_budget"]           # k = 2 for the climbs' round, 2 for the chain's
    assert (loud / "boost" / "boost_it_1.json").exists() and (loud / "trained_models" / "model_it_2.keras").exists()
    stages = json.loads((loud / "objective_calls" / "summary.json").read_text())["stages"]
    assert "audit/it1" in stages and stages["audit/it1"]["attempts"] == stages["audit/it1"].get("finite", 0), stages

    # The posterior pilot (acquisition.audit.posterior), with a tolerance every network exceeds:
    # cooled walkers evaluated by both ranks, a fine-tune, a second pilot round on a new chain,
    # and after the last fine-tune a look at the final network, whose chain the sampling step takes.
    said = run([settings(d, "pilot", tau=1.0e6, posterior=6, posterior_tolerance=1.0e-6, posterior_rounds=2),
                "-i", 3, "-o", d / "pilot"])
    (pilot,) = (d / "pilot").iterdir()
    assert "posterior pilot 0: 6 cooled T = 1 walkers evaluated" in said and "after the last fine-tune" in said, said[-4000:]
    assert said.count(REUSED) == 2, said[-4000:]
    for iteration in (1, 2):
        _, report = searches(pilot, iteration)
        rounds = report["posterior"]["rounds"]
        assert [r["fine_tuned"] for r in rounds] == [True, True] and report["true_calls"] == 12, (iteration, report)
    assert (pilot / "boost" / "boost_it_1.json").exists() and (pilot / "trained_models" / "model_it_2.keras").exists()
print("PASS the audit's chain search and posterior pilot in the loop, two ranks", flush=True)
