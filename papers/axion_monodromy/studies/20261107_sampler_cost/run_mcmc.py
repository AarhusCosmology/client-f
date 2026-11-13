"""Cobaya's Metropolis-Hastings MCMC on the PDP+ combined emulator (emulator.py), counting likelihood calls.

    mpirun -n 4 python run_mcmc.py zone1b      # omega in [38.25, 60.5] (zone 1b's range between its bands)
    mpirun -n 4 python run_mcmc.py full        # omega over the whole prior [1, 200]

One chain per MPI process; Cobaya's own convergence test (R-1 of the means below 0.01, of the 95% bounds below 0.2,
between chains). The proposal starts from the covariance of the emulator's own posterior samples in that omega range
(as a run would start from an earlier run's covariance) and is learnt by Cobaya; starting points are drawn from a
Gaussian of that mean and width. Every other prior is the training box. Writes chains/<case>/ and counts_<case>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

import os

import emulator as em

case = sys.argv[1]
# SAMPLER_TAG (8 November: "wc" for the H0WC emulator, with EMULATOR_RUN) prefixes every output; none for PDP+
TAG = os.environ.get("SAMPLER_TAG", "")
OUT = f"{TAG}_{case}" if TAG else case
OMEGA = {"zone1b": (38.25, 60.5), "full": (1.0, 200.0)}[case]
MAX = {"zone1b": 2_000_000, "full": 150_000}[case]          # accepted samples per chain at most (a cap)
comm = MPI.COMM_WORLD
X = em.posterior_samples()
io = em.NAMES.index("omega")
S = X[(X[:, io] >= OMEGA[0]) & (X[:, io] <= OMEGA[1])]
box = em.BOX.copy()
box[io] = OMEGA
here = Path(__file__).resolve().parent
cov = here / f"covmat_{OUT}.txt"
if comm.Get_rank() == 0:
    np.savetxt(cov, np.cov(S.T), header=" ".join(em.NAMES))
comm.Barrier()
params = {n: dict(prior=dict(min=float(lo), max=float(hi)),
                  ref=dict(dist="norm", loc=float(S[:, k].mean()), scale=float(S[:, k].std())),
                  proposal=float(S[:, k].std()))
          for k, (n, (lo, hi)) in enumerate(zip(em.NAMES, box))}
MAX_TRIES = None
if case == "full":
    # The covariance of the whole posterior proposes jumps of ~55 in omega, nearly all rejected (the first attempt stopped
    # after Cobaya's 40 * d = 760 rejections in a row). As a careful user would: start from the local proposal Cobaya
    # learnt in the zone run, start each chain at a random omega over the prior, and allow 10 000 rejections in a row.
    cov = here / "chains" / (f"{TAG}_zone1b" if TAG else "zone1b") / "mcmc.covmat"
    params["omega"]["ref"] = dict(dist="uniform", min=1.0, max=200.0)
    MAX_TRIES = 10000


def emulator_like(**kw):
    return em.logp([kw[n] for n in em.NAMES])


info = dict(params=params,
            likelihood=dict(emulator=dict(external=emulator_like, input_params=list(em.NAMES))),
            sampler=dict(mcmc=dict(covmat=str(cov), max_samples=MAX, Rminus1_stop=0.01, Rminus1_cl_stop=0.2,
                                   measure_speeds=False, **({} if MAX_TRIES is None else dict(max_tries=MAX_TRIES)))),
            output=str(here / "chains" / OUT / "mcmc"), force=True)
t0 = time.time()
from cobaya.run import run  # noqa: E402
updated, sampler = run(info)
calls = comm.gather(em.CALLS, root=0)
products = sampler.products()
if comm.Get_rank() == 0:
    out = dict(case=case, omega=OMEGA, chains=comm.Get_size(), calls_per_chain=calls, calls_total=int(sum(calls)),
               seconds=time.time() - t0, seconds_per_call=(time.time() - t0) / max(calls[0], 1),
               converged=bool(getattr(sampler, "converged", False)),
               Rminus1_last=float(getattr(sampler, "Rminus1_last", np.nan)), max_samples=MAX)
    (here / f"counts_{OUT}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
