"""What CLiENT's profile workers need on the cluster, in every backend process.

`python client.py profiles` starts `workers` processes, each with its own Cobaya model
(profiles/evaluation.py in CLiENT). install() replaces that module's two entry points by the
ones here, which add:

  * CLIENT_LOAD_GROUP=G (a run's job.env): the processes build their likelihood G at a time.
    HiLLiPoP's inverse covariance takes a process to 13 GB while it is read and leaves it at
    about a third; forty-six reading it together do not fit in the node. One lock per turn, in
    the node's own temporary directory. Unset or 0: all at once.
  * a logged error of Cobaya at a point (DES Y1 asking for P(k) beyond k = 3000 / Mpc, see
    refused.py) is a refused point, -inf, instead of the end of the study. What CAMB itself
    refuses already arrives as -inf. A refused point never gets a finite value here, whatever
    CLIENT_REFUSED_VALUE says: a profile is of the true objective.
  * the Boltzmann module Cobaya uses must be the environment's (CLIENT_ENV), or the patched classy a
    run's job.env asks for with CLASSPP_GUARD_DIR, or the CLASS++ build of CLASSPP_BUILD_DIR, as in
    launch.py (boltzmann_build.py).
  * CLIENT_CAMB_ACCURACY_TARGET=0 sets CAMB's global AccuracyTarget before the likelihood is
    built. CAMB 2.0.4 has "targeted accuracy improvements" (AccuracyTarget = 1, its default)
    that switch on at fixed thresholds; one is a neutrino mass of 600 k T_nu (0.1009 eV per
    state, results.f90 and equations.f90), above which the neutrino hierarchy is evolved
    deeper, its approximations switch later and four momenta are used instead of three. The
    spectra step by 1.6e-4 there. With 0 the likelihood is continuous in the neutrino mass,
    and is not the likelihood the run was trained on: it differs from it by about as much
    everywhere (README, "True profiles"). Unset: CAMB as the run had it.

The two functions are module-level so that a spawned process finds them by name; this file's
directory is on sys.path of every process the launchers start.

The region of a network trained on boosts is not applied here: it is the profile
setting `region` of CLiENT itself (profiles/region.py, docs/create_profiles.md; `run` by
default), written into the YAML by make_config.py. A study made when this module wrapped the
network for the profile code is recognised by the launcher.
"""
import fcntl
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import profiles.evaluation as evaluation

_initialize, _evaluate = evaluation.initialize_backend, evaluation.evaluate_backend
_kinds = {}


def _memory_gb():
    """(resident now, peak) of this process in GB; now is None where /proc is missing."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    try:
        with open("/proc/self/statm") as handle:
            now = int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except OSError:
        now = None
    return now, peak


def _turn(group):
    """An open file whose lock is this process's turn to build; closing it ends the turn.
    At most `group` processes of one study hold one at a time."""
    directory = Path(tempfile.gettempdir()) / f"client_profiles_{os.getuid()}_{os.getppid()}"
    directory.mkdir(parents=True, exist_ok=True)
    handles = [open(directory / f"turn_{k}", "a") for k in range(group)]
    while True:
        for handle in handles:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            for other in handles:
                if other is not handle:
                    other.close()
            return handle
        time.sleep(0.5 + random.random())


def _check_boltzmann(backend):
    prefix = os.environ.get("CLIENT_ENV")
    model = getattr(backend, "cobaya_model", None)
    if not prefix or model is None:
        return
    hpc_dir = str(Path(__file__).resolve().parent.parent)
    if hpc_dir not in sys.path:
        sys.path.insert(0, hpc_dir)
    from boltzmann_build import location_problem   # the env's build, or the CLASS++ build a run asks for

    for name in model.theory:
        root = name.split(".")[0]
        file = getattr(sys.modules.get(root), "__file__", None)
        if root in ("classy", "camb") and file:
            problem = location_problem(root, file, env_prefix=prefix)
            if problem:
                raise RuntimeError(problem)


def camb_accuracy_target():
    """CLIENT_CAMB_ACCURACY_TARGET as an integer, or None when it is unset."""
    text = (os.environ.get("CLIENT_CAMB_ACCURACY_TARGET") or "").strip()
    return int(text) if text else None


def initialize(*args):
    """profiles.evaluation.initialize_backend, in turn."""
    group = int(os.environ.get("CLIENT_LOAD_GROUP") or 0)
    target = camb_accuracy_target()
    if target is not None:
        import camb

        camb.config.AccuracyTarget = target
    started = time.time()
    turn = _turn(group) if group > 0 else None
    waited = time.time() - started
    try:
        _initialize(*args)
    finally:
        if turn is not None:
            turn.close()
    _check_boltzmann(evaluation._BACKEND)
    now, peak = _memory_gb()
    print(f"[profiles] backend process {os.getpid()}: likelihood built in {time.time() - started - waited:.0f} s"
          + (f" after waiting {waited:.0f} s for its turn" if turn is not None else "")
          + f"; memory {'%.1f' % now if now is not None else '?'} GB, peak {peak:.1f} GB"
          + (f"; CAMB AccuracyTarget {target}" if target is not None else ""), flush=True)


def evaluate(x):
    """profiles.evaluation.evaluate_backend; a logged error of Cobaya at the point is -inf."""
    try:
        from cobaya.log import LoggedError
    except ImportError:                                   # a backend without Cobaya
        return _evaluate(x)
    try:
        return _evaluate(x)
    except LoggedError as error:
        kind = re.sub(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", "#", str(error))[:160]
        _kinds[kind] = _kinds.get(kind, 0) + 1
        if _kinds[kind] == 1:
            print(f"[profiles] backend process {os.getpid()}: the likelihood refuses a point with a logged error; it and "
                  f"later ones of this kind count as -inf: {error}", flush=True)
        return float("-inf")


def install():
    """Make CLiENT's profile code start its backend processes through this module."""
    evaluation.initialize_backend, evaluation.evaluate_backend = initialize, evaluate


def newton_settings(text=None):
    """CLIENT_PROFILE_NEWTON="step=0.5,noise=0.01": fields of the residual Newton stage that a
    profile YAML does not carry (profiles/residual_newton.py::NewtonConfig; the YAML has only
    corrected_tol and maxfun). For a likelihood whose value noise is not the 1e-3 to 2.5e-3 the
    defaults were set for; Returns the dict it applied.

    These are outside what the method was validated with. They are printed, and the launcher
    writes them beside the study and refuses to resume it with others."""
    import dataclasses

    import profiles.residual_newton as newton

    text = os.environ.get("CLIENT_PROFILE_NEWTON", "") if text is None else text
    fields = {f.name: f.type for f in dataclasses.fields(newton.NewtonConfig)}
    settings = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, equals, value = item.partition("=")
        key = key.strip()
        if not equals or key not in fields or key in ("tol", "maxcalls"):
            raise ValueError(f"CLIENT_PROFILE_NEWTON: {item!r} is not a field of NewtonConfig that may be set here "
                             f"(tol and maxcalls are corrected_tol and maxfun of the profile YAML)")
        kind = fields[key] if isinstance(fields[key], type) else {"float": float, "int": int, "bool": bool}[str(fields[key])]
        settings[key] = (value.strip().lower() in ("1", "true", "yes")) if kind is bool else kind(value)
    if settings:
        base = newton.NewtonConfig

        def configured(**given):
            return base(**{**settings, **given})

        newton.NewtonConfig = configured
    return settings
