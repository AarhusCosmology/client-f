"""dynesty nested sampling (a standard nested sampler; PolyChord is not on PyPI) on the PDP+ combined emulator over the
whole prior (the training box, omega in [1, 200]), counting likelihood calls.

    PYTHONPATH=<dynesty> python run_nested.py NLIVE WORKERS

Random-direction slice sampling as PolyChord does it, with PolyChord's default of 5 * ndim slices per new point
(sample="rslice", slices=95), multi-ellipsoid bounds; stopped when the live points hold 1e-3 of the evidence
(dlogz = ln(1 + 1e-3)), BEST's tolerance. Writes nested_<NLIVE>.json and a checkpoint.
"""
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
NLIVE, WORKERS = int(sys.argv[1]), int(sys.argv[2])
import os  # noqa: E402
TAG = os.environ.get("SAMPLER_TAG", "")     # "wc" for the H0WC emulator (with EMULATOR_RUN); none for PDP+
NAME = f"nested_{TAG}_{NLIVE}" if TAG else f"nested_{NLIVE}"


def init():
    global em, LO, WIDTH
    sys.path.insert(0, str(here))
    import emulator as _em
    em = _em
    LO, WIDTH = em.BOX[:, 0], em.BOX[:, 1] - em.BOX[:, 0]


def prior_transform(u):
    return LO + u * WIDTH


def loglike(x):
    return em.logp(x)


def calls(_):
    return em.CALLS


def main():
    init()
    ndim = len(em.NAMES)
    import dynesty
    t0 = time.time()
    with mp.get_context("spawn").Pool(WORKERS, initializer=init) as pool:
        sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=NLIVE, bound="multi",
                                        sample="rslice", slices=5 * ndim, pool=pool, queue_size=WORKERS,
                                        rstate=np.random.default_rng(42))
        sampler.run_nested(dlogz=math.log1p(1e-3), print_progress=True,
                           checkpoint_file=str(here / f"{NAME}.save"), checkpoint_every=600)
    r = sampler.results
    out = dict(nlive=NLIVE, workers=WORKERS, ncall=int(np.sum(r.ncall)), niter=int(r.niter), logz=float(r.logz[-1]),
               logzerr=float(r.logzerr[-1]), information=float(r.information[-1]), seconds=time.time() - t0,
               calls_per_iteration=float(np.sum(r.ncall) / r.niter), sample="rslice", slices=5 * ndim,
               dlogz=math.log1p(1e-3), emulator=str(em.RUN))
    (here / f"{NAME}.json").write_text(json.dumps(out, indent=1))
    np.savez(here / f"{NAME}_samples.npz", samples=r.samples, logwt=r.logwt, logl=r.logl, names=np.array(em.NAMES))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
