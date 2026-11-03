"""Run with mpiexec -n 2: error propagation and subsequent batch reuse for every schedule."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpi4py import MPI
from utils.mpi_utils import broadcast_and_evaluate, EvaluationBatchError
comm=MPI.COMM_WORLD
for schedule in ('static','chunked'):
    os.environ['CLIENT_MPI_SCHEDULE']=schedule
    for failed_rank in (0,1):
        def evaluate(x):
            if comm.rank==failed_rank:raise ValueError('deliberate failure')
            return np.array([x[0],2*x[0]])
        failed=False
        try:
            broadcast_and_evaluate(np.arange(8.)[:,None] if comm.rank==0 else None,evaluate)
        except EvaluationBatchError:failed=True
        assert failed,(schedule,failed_rank,comm.rank,failed)
        x,v=broadcast_and_evaluate(np.arange(5.)[:,None] if comm.rank==0 else None,
                                   lambda x:np.array([x[0],2*x[0]]))
        if comm.rank==0:np.testing.assert_equal(v,np.c_[x[:,0],2*x[:,0]])
    x,v=broadcast_and_evaluate(np.empty((0,1)) if comm.rank==0 else None,lambda x:0.)
    assert len(v)==0
    def bad_shape(x):return [0.] if x[0]<2 else [0.,1.]
    try:broadcast_and_evaluate(np.arange(5.)[:,None] if comm.rank==0 else None,bad_shape)
    except EvaluationBatchError:pass
    else:raise AssertionError('inconsistent shapes accepted')
    # Many failures with long messages: more than the 32 KiB a waiting rank can receive.
    def refuse(x):raise ValueError('refused: '+'x'*3000)
    try:broadcast_and_evaluate(np.ones((60,49)) if comm.rank==0 else None,refuse)
    except EvaluationBatchError as error:
        assert error.total==60 and len(error.failures)==(60 if comm.rank==0 else 5),(schedule,comm.rank,error.total,len(error.failures))
    else:raise AssertionError('a batch of failures was accepted')
if comm.rank==0:print('MPI evaluation failure smoke passed',flush=True)
