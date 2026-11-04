"""training.members in the main loop (training/members.py), on the analytic Gaussian example with
boosts from network 1, so that networks 0 and 1 are the tempered phase's and 2 and 3 are trained
on boost points:

0. the messages alone: a task of 18 MB (a real training set) reaches its member rank;
1. three ranks, members: 3: networks 0 and 1 are trained once (no member file); from network 2
   on, members 1 and 2 train on ranks 1 and 2 while rank 0 trains its own, every network's three
   members are saved with their record, and the training phase lists them;
2. one process, members: 3: the two extra members train on rank 0 after its own;
3. members: 1 against no members key, one process each: the same training sets, bit for bit;
4. three ranks, members: 3 with member_pilot: each member's posterior points are evaluated
   once with the true likelihood, the choice is by them, and they join the next training set;
5. one process, the same;
6. a pilot run stopped after its first training and continued: the pilot's points are added
   to the continued run's training set again.

Run by itself (it starts mpiexec): python tests/mpi_members_smoke.py
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MPI = [shutil.which("mpiexec") or "mpiexec", *shlex.split(os.environ.get("CLIENT_TEST_MPI_ARGS", ""))]


def run(arguments, ranks=1, timeout=900):
    launcher = [*MPI, "-n", str(ranks), sys.executable] if ranks > 1 else [sys.executable]
    result = subprocess.run([*launcher, str(ROOT / "client.py"), *map(str, arguments)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                     TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3"))
    if result.returncode != 0:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout[-6000:]}")
    return result.stdout


BOOST = dict(n=24, start_iteration=1, walkers=32, cool_steps=50, steps=200, components=1)
TEMPERED, BOOSTED = (0, 1), (2, 3)      # the networks of the tempered phase, and those trained on boost points


def config(d, name, **training):
    cfg = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    cfg["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    cfg["training"].update(n_epochs=4, **training)
    cfg["acquisition"]["boost"] = dict(BOOST)
    cfg["convergence"]["max_iterations"] = 4
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def only_run(directory):
    (run_dir,) = Path(directory).iterdir()
    return run_dir


def check_tempered_once(run_dir):
    """The tempered phase's networks are trained once: no member file, no pilot."""
    for n in TEMPERED:
        assert (run_dir / "trained_models" / f"model_it_{n}.keras").exists(), n
        assert not list((run_dir / "trained_models/members").glob(f"*_it_{n}*")), n
        assert not list((run_dir / "training_history").glob(f"history_it_{n}_member*")), n


def check_members(run_dir, networks, log):
    check_tempered_once(run_dir)
    for n in networks:
        record = json.loads((run_dir / "trained_models/members" / f"members_it_{n}.json").read_text())
        assert [m["member"] for m in record["members"]] == [0, 1, 2], record
        assert all(m.get("error") is None for m in record["members"]), record
        assert record["selection_points"] > 0 and record["kept"] in (0, 1, 2), record
        for m in range(3):
            assert (run_dir / "trained_models/members" / f"model_it_{n}_member{m}.keras").exists()
            assert (run_dir / "training_history" / f"history_it_{n}_member{m}.csv").exists()
        phase = json.loads((run_dir / "phases" / f"training_it_{n}.json").read_text())
        listed = json.dumps(phase)
        assert f"members_it_{n}.json" in listed and f"model_it_{n}_member2.keras" in listed, phase
    assert "Members (" in log and "Kept:" in log, log[-3000:]
    return record


PILOT = dict(members=3, member_pilot=12, member_pilot_steps=200, member_pilot_walkers=16)

STOP_AT_ACQUISITION = """import sys
sys.path.insert(0, {root!r})
import run.storage
begin = run.storage.PhaseStore.begin
def stop(self, phase):
    if phase.startswith("acquisition_it_2"):
        raise RuntimeError("stopped after the first training with members")
    return begin(self, phase)
run.storage.PhaseStore.begin = stop
import client
client.main()
"""


def check_pilot(run_dir, networks, log):
    import pandas as pd

    check_tempered_once(run_dir)
    for n in networks:
        record = json.loads((run_dir / "trained_models/members" / f"members_it_{n}.json").read_text())
        assert record["choice"] == "pilot" and record["kept"] in (0, 1, 2), record
        assert all(np.isfinite(m["pilot_error"]) for m in record["members"]), record
        pilot = pd.read_csv(run_dir / "trained_models/members" / f"pilot_it_{n}.csv")
        assert len(pilot) == record["pilot_points"] == 3 * 12, (len(pilot), record)
        assert sorted(set(pilot["member"])) == [0, 1, 2], pilot["member"].unique()
        assert {"pred_member0", "pred_member1", "pred_member2"} <= set(pilot.columns), pilot.columns
        following = run_dir / "training_data" / f"data_it_{n + 1}.csv"
        if following.exists():
            names = [c for c in pilot.columns if c not in ("member", "loglkl") and not c.startswith(("pred_", "derived:"))]
            data = pd.read_csv(following)[names].to_numpy(np.float32)
            rows = {r.tobytes() for r in data}
            assert all(r.tobytes() in rows for r in pilot[names].to_numpy(np.float32)), f"pilot {n} not in {following.name}"
        phase = json.dumps(json.loads((run_dir / "phases" / f"training_it_{n}.json").read_text()))
        assert f"pilot_it_{n}.csv" in phase, phase
    assert "Members' pilot errors:" in log and "pilot points added" in log, log[-3000:]
    assert "Members (" not in log, log[-3000:]          # no held-out selection set with the pilot
    return record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MESSAGES = """import sys
sys.path.insert(0, {root!r})
import numpy as np
from mpi4py import MPI
from training import members
comm = MPI.COMM_WORLD
members.train_member = lambda task: dict(member=task["member"], rows=len(task["inputs"]), rank=comm.Get_rank())
if comm.Get_rank() == 0:
    big = np.zeros((200000, 23), np.float32)          # 18 MB, as a real training set
    members.dispatch(comm, {{1: dict(member=1, inputs=big), 2: None}})
    results = members.collect(comm, [1])
    assert results == [dict(member=1, rows=200000, rank=1)], results
    print("messages OK", flush=True)
else:
    members.serve(comm)
"""


def messages(d):
    """0. rank 0's task of megabytes reaches its member rank, and the result comes back."""
    script = d / "messages.py"
    script.write_text(MESSAGES.format(root=str(ROOT)))
    result = subprocess.run([*MPI, "-n", "3", sys.executable, str(script)], cwd=ROOT, text=True, timeout=300,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
    if result.returncode != 0 or "messages OK" not in result.stdout:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout[-4000:]}")


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    messages(d)
    print("0. an 18 MB task reaches its member rank, the result comes back")
    log = run([config(d, "three", members=3), "-i", 4, "-o", d / "mpi"], ranks=3)
    record = check_members(only_run(d / "mpi"), BOOSTED, log)
    ranks = sorted({m["rank"] for m in record["members"]})
    assert ranks == [0, 1, 2], record
    print("1. three ranks: networks 0 and 1 trained once; members on ranks", ranks, "; last kept", record["kept"],
          "-", record["reason"])

    log = run([config(d, "alone", members=3), "-i", 4, "-o", d / "alone"])
    record = check_members(only_run(d / "alone"), BOOSTED, log)
    assert {m["rank"] for m in record["members"]} == {0}, record
    print("2. one process: all members on rank 0; last kept", record["kept"])

    run([config(d, "one", members=1), "-i", 4, "-o", d / "one"])
    run([config(d, "none"), "-i", 4, "-o", d / "none"])
    a, b = only_run(d / "one"), only_run(d / "none")
    for n in range(5):
        name = f"training_data/data_it_{n}.csv"
        if (a / name).exists() or (b / name).exists():
            assert digest(a / name) == digest(b / name), name
    assert not (a / "trained_models/members").exists()
    print("3. members: 1 = no key: training sets identical, no member files")

    log = run([config(d, "pilot", **PILOT), "-i", 4, "-o", d / "pilot"], ranks=3)
    record = check_pilot(only_run(d / "pilot"), BOOSTED, log)
    assert sorted({m["rank"] for m in record["members"]}) == [0, 1, 2], record
    print("4. three ranks, the pilot from network 2: last kept", record["kept"], "-", record["reason"])

    log = run([config(d, "pilot1", **PILOT), "-i", 4, "-o", d / "pilot1"])
    record = check_pilot(only_run(d / "pilot1"), BOOSTED, log)
    print("5. one process, the pilot: last kept", record["kept"])

    stopper = d / "stop.py"
    stopper.write_text(STOP_AT_ACQUISITION.format(root=str(ROOT)))
    stopped = subprocess.run([sys.executable, str(stopper), str(config(d, "stopped", **PILOT)), "-i", "4", "-o",
                              str(d / "stopped")], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=900,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                      TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3"))
    assert stopped.returncode != 0 and "stopped after the first training with members" in stopped.stdout, \
        stopped.stdout[-3000:]
    run_dir = only_run(d / "stopped")
    log = run([run_dir, "-i", 1])
    assert "plus 36 pilot points from pilot_it_2.csv" in log, log[-3000:]
    check_pilot(run_dir, (2,), "Members' pilot errors: pilot points added")
    print("6. stopped after training and continued: the 36 pilot points are in the training set again")
print("mpi members smoke: OK")
