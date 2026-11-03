"""Manual two-rank smoke: mpiexec -n 2 python tests/mpi_audit_failure_smoke.py.

Checks static/chunked Python exception transport and collective termination.
No CLASS++ evaluations or network training.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpi4py import MPI
from client import audit_network
from config.config import AuditConfig

comm = MPI.COMM_WORLD
assert comm.size == 2
cfg = SimpleNamespace(acquisition=SimpleNamespace(audit=AuditConfig(failure_policy='penalize')))

for schedule in ('static', 'chunked'):
    os.environ['CLIENT_MPI_SCHEDULE'] = schedule
    for fatal in (False, True):
        def backend(x):
            if int(x[0]) == 1:
                if fatal:
                    raise TypeError('injected fatal failure')
                raise FloatingPointError('injected numerical failure')
            return [float(x[0]), 2.]
        likelihood = SimpleNamespace(n_derived=1, evaluate_with_derived=backend,
            split_evaluations=lambda v: (np.asarray(v).reshape(-1,2)[:,0], np.asarray(v).reshape(-1,2)[:,1:]))
        def audit(evaluate, **kwargs):
            for _ in range(2):
                x,y,d = evaluate(np.arange(4).reshape(-1,1))
                assert np.isnan(y[1]) and np.isfinite(y[[0,2,3]]).all()
            return 'complete'
        stopped = False
        with mock.patch('training.audit.audit_optima', side_effect=audit):
            try:
                audit_network(cfg, likelihood, comm.rank == 0, run_audit=comm.rank == 0)
            except RuntimeError as error:
                assert 'injected fatal failure' in str(error)
                stopped = True
        assert comm.allgather(stopped) == [fatal, fatal]
        if comm.rank == 0:
            print(f'PASS {schedule} fatal={fatal}', flush=True)

with mock.patch('training.audit_failures.load_penalties', side_effect=ValueError('bad persisted penalty')):
    stopped = False
    try:
        audit_network(cfg, likelihood, comm.rank == 0, run_audit=comm.rank == 0,
                      penalty_source=('unused',1,['x']))
    except (ValueError, RuntimeError) as error:
        assert 'bad persisted penalty' in str(error)
        stopped = True
    assert all(comm.allgather(stopped))
if comm.rank == 0:
    print('PASS penalty load failure releases all ranks', flush=True)

# Longer than the 32 KiB the workers' receive holds: a whole one stopped the job in MPI itself.
with mock.patch('training.audit.audit_optima', side_effect=RuntimeError('long ' * 20_000)):
    stopped = False
    try:
        audit_network(cfg, likelihood, comm.rank == 0, run_audit=comm.rank == 0)
    except RuntimeError as error:
        assert 'long' in str(error)
        stopped = True
    assert all(comm.allgather(stopped))
if comm.rank == 0:
    print('PASS a long audit error reaches every rank', flush=True)
