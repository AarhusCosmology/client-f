"""One evaluation of a run's Cobaya input, before it has a client.yaml or a training box:

    python install/check_likelihoods.py RUN        # RUN = a folder under runs/ with a cobaya.yaml

after install/06_external_likelihoods.sh (or step 5). One process, one thread. Loading the
likelihoods is the slow and memory-hungry part (HiLLiPoP reads an inverse covariance of several
GB in one piece and makes temporary copies of it), so for a run with large likelihood data
use a compute node:  sbatch smoke/submit_likelihood_check.sh RUN.  Every stage prints when it
starts, with the seconds and the peak memory so far. It checks, and stops at the first that fails:

  1. every likelihood of runs/RUN/cobaya.yaml loads (its code is importable, its data is there)
  2. the Boltzmann code is the environment's, not a copy under $COBAYA_PACKAGES_PATH/code
  3. the parameters CLiENT will vary (those with a prior in the file) are exactly the ones
     Cobaya samples: a likelihood's default nuisance parameter that is not written into the
     file would be missing from every point CLiENT evaluates
  4. the log-posterior at the file's reference point is finite; it prints every likelihood's
     chi2 and the seconds of the first and of a second, slightly moved, evaluation
  5. which sampled parameters have no finite prior range (the training box must give one)

It trains nothing and writes nothing.
"""
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
# One thread unless told otherwise: on a login node CAMB would take every core it sees.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def fail(msg):
    print(f"FATAL: {msg}", flush=True)
    sys.exit(1)


def ok(msg):
    print(f"  ok  {msg}", flush=True)


START = time.perf_counter()


def stage(msg):
    """Say what starts now: a stage that hangs or swaps is then visible as the last line."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    print(f"[{time.perf_counter() - START:6.0f} s, peak memory {peak:5.1f} GB] {msg} ...", flush=True)


if len(sys.argv) != 2:
    fail("usage: python install/check_likelihoods.py RUN")
run = HERE / "runs" / sys.argv[1]
cobaya_yaml = run / "cobaya.yaml"
if not cobaya_yaml.exists():
    fail(f"no {cobaya_yaml}")
for name in ("CLIENT_REPO", "COBAYA_PACKAGES_PATH"):
    if not os.environ.get(name):
        fail(f"${name} is not set -- source env.sh")
packages = os.environ["COBAYA_PACKAGES_PATH"]
for code in ("classy", "CAMB", "camb"):
    if Path(packages, "code", code).exists():
        fail(f"{packages}/code/{code} exists: Cobaya would evaluate with it instead of the env's build")
sys.path.insert(0, os.environ["CLIENT_REPO"])
from likelihood.cobaya import CobayaLikelihood  # noqa: E402

stage(f"loading the likelihoods of {run.name} (their data are read now)")
t = time.perf_counter()
lik = CobayaLikelihood(str(cobaya_yaml))
model = lik.cobaya_model
ok(f"{run.name}: {len(model.likelihood)} likelihoods loaded in {time.perf_counter() - t:.0f} s: {', '.join(model.likelihood)}")

for name in model.theory:
    root = name.split(".")[0]
    module = sys.modules.get(root)
    if root in ("classy", "camb") and module is not None:
        inside = os.path.realpath(module.__file__).startswith(os.path.realpath(sys.prefix) + os.sep)
        line = f"{root} {getattr(module, '__version__', '?')} from {module.__file__}"
        if not inside:
            print(f"  NOTE  {line} is not inside this environment ({sys.prefix})", flush=True)
        else:
            ok(line)
        pinned = os.environ.get("CAMB_VERSION")
        if root == "camb" and pinned and module.__version__ != pinned:
            fail(f"camb is {module.__version__}, env.sh pins {pinned}")
        break

sampled = list(model.parameterization.sampled_params())
varied = list(lik.param_names)
if set(sampled) != set(varied):
    fail(f"Cobaya samples {sorted(set(sampled) - set(varied))} that the file does not list with a prior, "
         f"and the file lists {sorted(set(varied) - set(sampled))} that Cobaya does not sample")
ok(f"{len(varied)} sampled parameters, the same in the file and in Cobaya")

# The reference point of the file: ref (its loc), else the centre of the prior.
point, unbounded = [], []
for p in lik._params:
    info = lik.cobaya_info["params"][p.name]
    prior = info["prior"]
    if p.lower is None or p.upper is None:
        unbounded.append(p.name)
    if p.center is not None:
        point.append(float(p.center))
    elif "loc" in prior:
        point.append(float(prior["loc"]))
    else:
        point.append(0.5 * (float(prior["min"]) + float(prior["max"])))
point = np.array(point)

stage("first evaluation, at the reference point (the Boltzmann code, then every likelihood)")
t = time.perf_counter()
result = model.logposterior(dict(zip(varied, point)))
first = time.perf_counter() - t
if not np.isfinite(result.logpost):
    fail(f"log-posterior at the reference point is {result.logpost} "
         f"(log-priors {list(result.logpriors)}, log-likelihoods {list(result.loglikes)})")
ok(f"log-posterior at the reference point {result.logpost:.2f} (log-prior {float(np.sum(result.logpriors)):.2f}) in {first:.1f} s")
for name, value in zip(model.likelihood, result.loglikes):
    print(f"        {name:42s} chi2 = {-2 * value:12.2f}")

stage("second evaluation, 1e-4 away, through CLiENT's wrapper")
# Towards the middle of a prior range, so that a reference value on its edge (HiLLik's
# SPT3G_beta_pol_* start at their upper limit) does not leave it.
moved = point * (1 + 1e-4) + 1e-6
for k, p in enumerate(lik._params):
    if p.lower is not None and p.upper is not None:
        middle = 0.5 * (float(p.lower) + float(p.upper))
        moved[k] = point[k] + 1e-4 * (middle - point[k]) if point[k] != middle else point[k] + 1e-5 * (float(p.upper) - middle)
t = time.perf_counter()
again = lik.loglkl(moved)
second = time.perf_counter() - t
if not np.isfinite(again):
    fail(f"log-posterior at a point 1e-4 away is {again}")
ok(f"a second point, 1e-4 away, through CLiENT's wrapper: {again:.2f} in {second:.1f} s "
   f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")

print(f"\n{len(unbounded)} sampled parameters have no finite prior range; the training box must set them:\n  "
      + ", ".join(unbounded))
stage("done")
print("LIKELIHOOD CHECK PASSED", flush=True)
