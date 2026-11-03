"""The main loop under MPI, two ranks: a job killed inside the acquisition of network 0 (os._exit
on the master, as a job at its time limit ends), continued, and continued once more.

Run by tests/run_mpi_smokes.py, or by itself: it starts mpiexec. The unit suite and the recovery
smoke run the loop on one process, and the other MPI smokes run pieces of it: this is the loop
itself with its broadcasts, where a rank waiting for another is the usual way to go wrong.
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

KILLED = """import os, sys
sys.path.insert(0, {root!r})
import client
original = client.evaluate_true
def killing(likelihood, samples):
    from utils.mpi_utils import is_master
    if likelihood._client_calls.stage.startswith('acquisition/') and is_master():
        sys.stdout.flush()
        os._exit(3)
    return original(likelihood, samples)
client.evaluate_true = killing
client.main()
"""


def run(arguments, script=None, succeeds=True, timeout=300):
    result = subprocess.run([*LAUNCHER, str(script or ROOT / "client.py"), *map(str, arguments)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout[-6000:]}")
    return result.stdout


def phases(directory):
    return {file.stem: json.loads(file.read_text())["status"] for file in (directory / "phases").glob("*.json")}


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    config = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    config["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    config["training"]["n_epochs"] = 2
    config["acquisition"].update(n_append=10, pool_factor=2)
    (d / "config.yaml").write_text(yaml.safe_dump(config))
    (d / "killed.py").write_text(KILLED.format(root=str(ROOT)))

    run([d / "config.yaml", "-i", 2, "-o", d / "runs"], script=d / "killed.py", succeeds=False)
    (directory,) = (d / "runs").iterdir()
    assert phases(directory) == dict(initial="complete", training_it_0="complete", acquisition_it_0="incomplete"), phases(directory)

    said = run([directory, "-i", 1])                       # the unfinished acquisition is set aside and done again
    assert "Phase acquisition_it_0 did not finish" in said, said[-3000:]
    assert (directory / "training_data/data_it_1.csv").exists() and (directory / "trained_models/model_it_1.keras").exists()
    assert set(phases(directory).values()) == {"complete"}, phases(directory)

    said = run([directory, "-i", 1])                       # and a continuation with nothing to recover
    assert "did not finish" not in said, said[-3000:]
    assert (directory / "trained_models/model_it_2.keras").exists()
    assert set(phases(directory).values()) == {"complete"}, phases(directory)
    stages = json.loads((directory / "objective_calls/summary.json").read_text())["stages"]
    assert stages["initial/design"]["attempts"] == 128 and "acquisition/it1" in stages, stages
print("MPI loop recovery smoke passed", flush=True)
