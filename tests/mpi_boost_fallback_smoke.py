"""Manual two-rank smoke: mpiexec -n 2 python tests/mpi_boost_fallback_smoke.py.

A boost iteration whose network has no mode: every rank must learn that the iteration is an
ordinary one (client.draw_boost_points), or the ranks part ways at the injection.
No likelihood evaluations and no training.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpi4py import MPI

import client
from sampling.modes import NoModeFound

comm = MPI.COMM_WORLD
assert comm.size == 2
master = comm.rank == 0
cfg = SimpleNamespace(seed=1, sampling=SimpleNamespace(temperature=7.0), acquisition=SimpleNamespace(boost=object()))
points = np.arange(12.0).reshape(4, 3)

with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp) / "boost" if master else None
    for outcome, expected in ((NoModeFound("no mode found"), False), ((points, {"label": np.zeros(4, int)}), True)):
        # only the master draws; a worker that reached draw_boost would fail on its missing arguments
        with mock.patch("dataset.boost.draw_boost", side_effect=[outcome] if master else AssertionError("a worker drew")):
            got, draw, boosting = client.draw_boost_points(cfg, None, None, None, None, 7, None, directory, master)
        assert comm.allgather(boosting) == [expected, expected], (comm.rank, boosting)
        assert (got is not None) == (master and expected)
        if master:
            assert (directory / "fallback_it_7.json").exists()
            print(f"PASS every rank {'boosts' if expected else 'falls back'}", flush=True)
