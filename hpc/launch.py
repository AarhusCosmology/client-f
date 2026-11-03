"""CLiENT with an explicit numerical domain, under MPI.

    mpirun -n N python launch.py runs/RUN/domain_bounds.json <client.py arguments>

    ... launch.py runs/RUN/domain_bounds.json runs/RUN/client.yaml -n RUN -o results -i 16    (new run)
    ... launch.py runs/RUN/domain_bounds.json results/<timestamp>_RUN -i 2                     (continue)

What it adds to client.py:

  * the training box is the explicit domain of the json file (domain.py), on every rank;
    Cobaya's prior densities are unchanged;
  * batch points are rounded to single precision before they are evaluated, so the
    coordinates stored in the training files are exactly the ones the likelihood saw;
  * a continued run must be given the domain it was started with (compared with the run's
    metadata.json), and the Boltzmann module Cobaya uses must be the one of the environment, or the
    patched classy a run's job.env asks for with CLASSPP_GUARD_DIR, or the CLASS++ build of another
    commit it asks for with CLASSPP_BUILD_DIR (boltzmann_build.py);
  * the run's settings must leave prior.n_sigma null: client.py would apply it after the domain,
    and the run would train in another box than the json file's (n_sigma_problem);
  * with CLIENT_LOAD_GROUP=G in the environment the ranks build their likelihood G at a time,
    in rank order, instead of all at once. For likelihoods whose data need far more memory
    while they are read than afterwards (HiLLiPoP's inverse covariance), where every rank
    loading together would exceed the node (loading.py). Unset or 0: all at once.

likelihood.loglkl is wrapped in one respect only: a logged error of Cobaya at a point (which
stop_at_error: false does not cover) gives -inf instead of stopping the run (refused.py).
Otherwise every true evaluation goes through CLiENT's own code. With CLIENT_REFUSED_VALUE set
a refused point gets a finite value and is trained on: a number, or `design` to take it from
the design's own values (refused.py). The value is written to the run (refused_value.json),
and a continued run must be given the same one.
Rank 0 writes logs/backend_<tag>.json: module, version and file of the Boltzmann code, the
commits of CLiENT and CLASS++ (boltzmann_build.commits: CLASS++'s only when the job's settings
name it), the parameters, the effective bounds, the likelihoods and the derived parameters. It also appends
one line per batch of true evaluations to logs/evaluations_<tag>.jsonl: when, how many points,
how many came back not finite, the seconds the batch took and the range of the values. The
audit's batches take their own transport (client.audit_network) and are wrapped the same way,
with kind "audit" in their lines. A failure to write that journal never stops the run.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
CLIENT_REPO = os.environ.get("CLIENT_REPO", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, CLIENT_REPO)
sys.path.insert(0, str(HERE))
# runs/ and logs/ of the jobs: this folder, or the folder of a site that keeps its own beside CLiENT (README.md)
LOGS = Path(os.environ.get("CLIENT_HPC_DIR") or HERE) / "logs"
comm = MPI.COMM_WORLD
rank = comm.rank


def n_sigma_problem(target, domain_file):
    """Why the launcher does not start the run `target` with its settings, or None. The settings
    are the YAML given for a new run, or the one in the run directory of a continued run.

    client.py applies a prior.n_sigma that is not null (likelihood.restrict_prior_bounds) after
    build() below has applied the box of the domain file. The run would then train in the box of
    ref +- n_sigma * scale and not in the domain file's, while logs/backend_<job>.json, written in
    build(), records the domain file's."""
    import yaml

    target = Path(target)
    for file in [target] if target.is_file() else sorted(target.glob("*.yaml")) + sorted(target.glob("*.yml")):
        try:
            settings = yaml.safe_load(file.read_text())
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue                                          # CLiENT says what is wrong with a file it cannot read
        prior = settings.get("prior") if isinstance(settings, dict) else None
        n_sigma = prior.get("n_sigma") if isinstance(prior, dict) else None
        if n_sigma is not None:
            return (f"[launch] {file}: prior.n_sigma is {n_sigma!r}; under the launcher it must be null. client.py applies it "
                    f"after the launcher has applied the box of {domain_file}: the run would train in the box of "
                    "ref +- n_sigma * scale, not in the domain file's, and logs/backend_<job>.json would record a box the run "
                    "does not have. "
                    + ("Set it to null: the training box is the domain file's." if target.is_file() else
                       "The run was started with it, so its box is not this domain file's (compare its metadata.json)."))
    return None


if len(sys.argv) < 3:
    raise SystemExit(__doc__)
DOMAIN_FILE = Path(sys.argv.pop(1))
TARGET = Path(sys.argv[1])
# Before anything is imported or built. Every rank reads the same file and stops; rank 0 says why.
problem = n_sigma_problem(TARGET, DOMAIN_FILE)
if problem:
    raise SystemExit(problem if rank == 0 else 1)

import client  # noqa: E402
from domain import apply_domain, read_domain  # noqa: E402
from loading import build_in_turn  # noqa: E402
from refused import design_value, read_record, refuse_logged_errors, write_record  # noqa: E402
from boltzmann_build import commits, location_problem  # noqa: E402
from run.run import Run  # noqa: E402
from utils import mpi_utils  # noqa: E402

BOUNDS = read_domain(DOMAIN_FILE)
original_build = client.build_likelihood
original_evaluate = client.broadcast_and_evaluate
original_evaluate_checked = mpi_utils.broadcast_and_evaluate_checked
original_from_args = Run.from_args.__func__
state = {}
# One tag for every rank: the ranks find the design's refused value in a file named after it.
TAG = comm.bcast(os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S")) if rank == 0 else None, root=0)
VALUE_FILE = LOGS / f"refused_value_{TAG}.json"


def from_args(cls, args):
    """CLiENT's own run set-up (rank 0 only), noting where the run is."""
    run = original_from_args(cls, args)
    state["run_dir"] = Path(run.run_dir)
    return run


def continued_refused_value(truth):
    """A continued run keeps the refused value it was started with (its refused_value.json).
    Runs started before that file existed have none and are taken at the given number."""
    record = read_record(TARGET / "refused_value.json") if TARGET.is_dir() else None
    setting = truth.refused_setting
    if record is None:
        if TARGET.is_dir() and isinstance(setting, tuple):
            raise RuntimeError(f"CLIENT_REFUSED_VALUE=design, but {TARGET} has no refused_value.json to continue with: "
                               "give the number the run was started with")
        return
    if isinstance(setting, tuple):
        truth.set_refused_value(record["value"])
    elif setting is None or not np.isclose(setting, record["value"], rtol=1e-9, atol=0):
        raise RuntimeError(f"{TARGET} was started with the refused value {record['value']:g}; "
                           f"CLIENT_REFUSED_VALUE is {os.environ.get('CLIENT_REFUSED_VALUE') or 'unset'}")


def boltzmann(truth):
    """(name, module file, version) of the Boltzmann code the Cobaya model evaluates with."""
    found = []
    for name, theory in truth.cobaya_model.theory.items():
        root = name.split(".")[0]
        module = sys.modules.get(root)
        if root in ("classy", "camb") and module is not None:
            entry = (root, getattr(module, "__file__", "?"), getattr(module, "__version__", "?"))
            if entry not in found:
                found.append(entry)
    return found


def build(*args, **kwargs):
    truth = build_in_turn(comm, lambda: original_build(*args, **kwargs), label="[launch]")   # CLIENT_LOAD_GROUP
    apply_domain(truth, BOUNDS)
    refuse_logged_errors(truth, rank, value_file=VALUE_FILE)   # a Cobaya logged error at a point: -inf (refused.py)
    continued_refused_value(truth)
    codes = boltzmann(truth)
    for name, file, _ in codes:             # the env's build, or the CLASS++ build a run asks for
        problem = location_problem(name, file)
        if problem:
            raise RuntimeError(problem)
    if TARGET.is_dir() and (TARGET / "metadata.json").exists():
        saved = {p["name"]: (p["lower"], p["upper"]) for p in json.loads((TARGET / "metadata.json").read_text())["parameters"]}
        for name, (lo, hi) in truth.prior_bounds.items():
            if name not in saved or not np.allclose(saved[name], (lo, hi), rtol=1e-6, atol=0):
                raise RuntimeError(f"{TARGET} was started with another domain for {name}: {saved.get(name)} against {(lo, hi)}")
    state["truth"] = truth
    if rank == 0:
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"backend_{TAG}.json").write_text(json.dumps(dict(
            target=str(TARGET), domain=str(DOMAIN_FILE), ranks=comm.size, host=MPI.Get_processor_name(),
            boltzmann=[dict(name=n, file=f, version=v) for n, f, v in codes], **commits(CLIENT_REPO),
            likelihoods=list(truth.cobaya_model.likelihood), parameter_names=truth.param_names,
            effective_bounds=truth.prior_bounds, derived_names=list(truth.derived_names),
            schedule=os.environ.get("CLIENT_MPI_SCHEDULE", "(default)"),
            refused_value=truth.refused_value, refused_setting=os.environ.get("CLIENT_REFUSED_VALUE") or None), indent=2))
        print(f"[launch] domain {DOMAIN_FILE}: {truth.ndim} parameters; Boltzmann code: "
              + ", ".join(f"{n} {v}" for n, _, v in codes), flush=True)
        if truth.refused_value is not None:
            print(f"[launch] a point the likelihood refuses gets the value {truth.refused_value:g} and is trained on "
                  "(CLIENT_REFUSED_VALUE)", flush=True)
        elif truth.refused_setting is not None:
            VALUE_FILE.unlink(missing_ok=True)
            print(f"[launch] a point the likelihood refuses gets a value taken from the design (the {truth.refused_setting[1]:g} "
                  "quantile of its finite values) and is trained on (CLIENT_REFUSED_VALUE=design)", flush=True)
    return truth


def single_precision_in_box(samples):
    bounds = np.array(list(state["truth"].prior_bounds.values()), float)
    lo = bounds[:, 0].astype(np.float32)
    hi = bounds[:, 1].astype(np.float32)
    lo = np.where(lo.astype(float) < bounds[:, 0], np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(float) > bounds[:, 1], np.nextafter(hi, np.float32(-np.inf)), hi)
    return np.clip(np.asarray(samples, dtype=np.float32), lo, hi).astype(float)


def evaluate(samples, evaluator):
    if rank == 0:
        samples = single_precision_in_box(samples)
    started = time.time()
    result = original_evaluate(samples, evaluator)
    if rank == 0:
        result = refused_value_from_design(result)
        journal(started, result)
    return result


def evaluate_checked(samples, evaluator, width):
    """The audit's batches. The design has been evaluated before any of them, so the refused
    value is settled (refused_value_from_design)."""
    if rank == 0:
        samples = single_precision_in_box(samples)
    started = time.time()
    points, values, errors = original_evaluate_checked(samples, evaluator, width)
    if rank == 0:
        journal(started, (points, values), kind="audit")
    return points, values, errors


def refused_value_from_design(result):
    """With CLIENT_REFUSED_VALUE=design the first evaluation of a new run, its design, sets the
    value: here, on rank 0, before CLiENT sees the result. The design's own refused points
    came back as -inf and get the value now; the other ranks read it from VALUE_FILE at their
    next refused point. Either way the value is written to the run once, for a continuation."""
    truth = state["truth"]
    record = None
    if isinstance(truth.refused_setting, tuple) and truth.refused_value is None:
        values = np.array(result[1], dtype=float)                 # a copy, in the shape CLiENT expects
        targets = values.reshape(len(values), -1)[:, 0]           # as split_evaluations reads them
        value, raw = design_value(targets, truth.refused_setting[1])
        refused = ~np.isfinite(targets)
        record = dict(value=value, setting=os.environ["CLIENT_REFUSED_VALUE"], quantile=truth.refused_setting[1],
                      quantile_value=raw, design_points=int(len(targets)), design_refused=int(refused.sum()),
                      design_best=float(targets[~refused].max()), design_worst=float(targets[~refused].min()), job=TAG)
        targets[refused] = value
        truth.set_refused_value(value)
        write_record(VALUE_FILE, record)
        print(f"[launch] refused value set from the design: {value:g} (the {record['quantile']:g} quantile of its "
              f"{len(targets) - record['design_refused']} finite values is {raw:.6g}; best {record['design_best']:.6g}, worst "
              f"{record['design_worst']:.6g}); {record['design_refused']} refused design points carry it", flush=True)
        result = (result[0], values)
    elif truth.refused_value is not None and "recorded" not in state:
        record = dict(value=truth.refused_value, setting=os.environ.get("CLIENT_REFUSED_VALUE"), job=TAG)
    run_dir = state.get("run_dir")
    if record is not None and run_dir is not None and run_dir.is_dir() and not (run_dir / "refused_value.json").exists():
        write_record(run_dir / "refused_value.json", record)
    state["recorded"] = True
    return result


def memory_regions():
    """Memory mappings of this process (rank 0, which trains and samples), or None where
    /proc is not there. Every function TensorFlow compiles with XLA adds some and none is
    given back; Linux allows a process 65 530 (vm.max_map_count), and two runs of 21
    iterations ended at that limit with "LLVM compilation error: Cannot allocate memory"."""
    try:
        with open("/proc/self/maps") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def journal(started, result, kind=None):
    """One line per batch of true evaluations (see the module text); never raises."""
    try:
        values = np.asarray(state["truth"].split_evaluations(result[1])[0], float)   # as the loop reads them
        finite = values[np.isfinite(values)]
        not_finite = int(len(values) - len(finite))
        floor = state["truth"].refused_value           # refused points carry this value when it is set
        refused = int(np.count_nonzero(finite == floor)) if floor is not None else 0
        finite = finite[finite != floor] if floor is not None else finite
        state["batch"] = state.get("batch", 0) + 1
        line = dict(batch=state["batch"], kind=kind, started=started, seconds=time.time() - started, points=int(len(values)),
                    not_finite=not_finite, refused=refused, refused_value=floor, ranks=comm.size,
                    memory_regions=memory_regions(),
                    best=float(finite.max()) if len(finite) else None,       # best, median, worst: of the values the likelihood gave
                    median=float(np.median(finite)) if len(finite) else None,
                    worst=float(finite.min()) if len(finite) else None)
        with open(LOGS / f"evaluations_{TAG}.jsonl", "a") as handle:
            handle.write(json.dumps(line) + "\n")
    except Exception as error:                                   # diagnostics must not cost a run
        print(f"[launch] evaluation journal not written: {error!r}", flush=True)


client.build_likelihood = build
client.broadcast_and_evaluate = evaluate
mpi_utils.broadcast_and_evaluate_checked = evaluate_checked       # client.audit_network reads it when called
Run.from_args = classmethod(from_args)
client.main()
