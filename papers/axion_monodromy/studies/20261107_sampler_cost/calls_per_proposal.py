"""Likelihood calls per Metropolis-Hastings proposal (8 November 2026).

    python calls_per_proposal.py zone1b|full          # PDP+; with SAMPLER_TAG=wc EMULATOR_RUN=<run> for H0WC

The progress files of the long runs (chains/<case>/mcmc.progress) give the accepted samples N of all chains and the
acceptance rate, so N / acceptance is the number of proposals. A proposal outside the prior box is rejected by Cobaya
without a call of the likelihood (A_osc >= 0 and the box in omega are often hit), so the calls are fewer. This runs one
chain of the same sampler with the proposal the long run learnt (chains/<case>/mcmc.covmat), started from that run's
last points, and counts both: the calls (emulator.CALLS) and the proposals (the sum of the chain's weights, which count
the rejections). Writes calls_per_proposal_<case>.json and chains/calib_<case>/.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

import emulator as em

case = sys.argv[1]
TAG = os.environ.get("SAMPLER_TAG", "")
OUT = f"{TAG}_{case}" if TAG else case
OMEGA = {"zone1b": (38.25, 60.5), "full": (1.0, 200.0)}[case]
here = Path(__file__).resolve().parent
src = here / "chains" / OUT
io = em.NAMES.index("omega")
box = em.BOX.copy()
box[io] = OMEGA
names = open(src / "mcmc.1.txt").readline().split()[1:]
last = np.vstack([np.loadtxt(f)[-2000:] for f in sorted(glob.glob(str(src / "mcmc.[0-9].txt")))])
X = last[:, [names.index(n) for n in em.NAMES]]
params = {n: dict(prior=dict(min=float(lo), max=float(hi)),
                  ref=dict(dist="norm", loc=float(X[:, k].mean()), scale=float(X[:, k].std())))
          for k, (n, (lo, hi)) in enumerate(zip(em.NAMES, box))}


def emulator_like(**kw):
    return em.logp([kw[n] for n in em.NAMES])


from cobaya.run import run  # noqa: E402
info = dict(params=params, likelihood=dict(emulator=dict(external=emulator_like, input_params=list(em.NAMES))),
            sampler=dict(mcmc=dict(covmat=str(src / "mcmc.covmat"), learn_proposal=False, burn_in=0, max_samples=6000,
                                   Rminus1_stop=0.0, measure_speeds=False, max_tries=100000)),
            output=str(here / "chains" / f"calib_{OUT}" / "mcmc"), force=True)
updated, sampler = run(info)
w = np.loadtxt(here / "chains" / f"calib_{OUT}" / "mcmc.1.txt")[:, 0]
out = dict(case=OUT, calls=int(em.CALLS), proposals=float(w.sum()), accepted=int(len(w)),
           calls_per_proposal=em.CALLS / float(w.sum()), acceptance=len(w) / float(w.sum()))
(here / f"calls_per_proposal_{OUT}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
