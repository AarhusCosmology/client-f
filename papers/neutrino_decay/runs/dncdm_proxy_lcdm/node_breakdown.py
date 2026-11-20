"""chi^2 of every likelihood along the true log10Gamma profiles of the DNCDM proxy (9 November 2026): which data make the
step from the Gamma -> 0 plateau to the minimum at log10Gamma ~ 6, where the proxy falls short of massless LambdaCDM
(minimize_lcdm.py: by 0.59 in chi^2 on PDP+, 0.77 on H0WC), and which data make the H0WC peak at log10Gamma 12.8.

    python runs/dncdm_proxy_lcdm/node_breakdown.py [--workers N] [--only pp wc]       # submit_breakdown.sh

The profile studies store only the total log posterior of a node. This recomputes every node of the two finished 1D
studies of each data set (results_profiles/dncdm_proxy_<ds>_gamma outside the peak grid's [11.5, 13.75], and all of
dncdm_proxy_<ds>_peak_gamma) with the run's own cobaya.yaml at the node's true optimum (xopt), and keeps Cobaya's
chi^2 of every likelihood and the log prior. The recomputed log posterior against the study's own (logl) checks that
this is the objective CLiENT optimised; a difference above 0.05 is printed with the node.

One Cobaya model per process (each pool serves one data set; PDP+ first), two threads each; the most expensive nodes
(large Gamma) go first. Output: logs/node_breakdown_<job>.json (every node) and a table per data set of Delta chi^2 by
likelihood against the plateau node log10Gamma = -2. The shell must have sourced env.sh and runs/dncdm_proxy_pp/job.env.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[2]
RUN_YAML = {ds: KIT / "runs" / f"dncdm_proxy_{ds}" / "cobaya.yaml" for ds in ("pp", "wc")}
PEAK = (11.5, 13.75)
CHECK = 0.05            # |recomputed - study's log posterior| printed above this

_model = {}


def _init(yaml_path):
    """Build the data set's model once per process (the likelihoods take a minute to load). A failure is kept and
    returned by the first task: an initializer that raises makes the pool start new workers for ever."""
    os.chdir(KIT)                                  # the yamls' python_path (likelihoods/) is relative to this folder (papers/neutrino_decay/)
    try:
        from cobaya.model import get_model
        from cobaya.yaml import yaml_load_file

        info = yaml_load_file(str(yaml_path))
        info.pop("sampler", None)
        info.pop("output", None)
        _model["m"] = get_model(info)
    except Exception as error:                     # noqa: BLE001 - reported by task(), which stops the pool
        _model["error"] = f"{type(error).__name__}: {error}"


def nodes(ds, root=None):
    """(study, index, log10Gamma, study logl, names, x) of every node: the full grid outside the peak grid, the peak grid."""
    root = Path(root or KIT / "results_profiles")
    out = []
    for study in (f"dncdm_proxy_{ds}_gamma", f"dncdm_proxy_{ds}_peak_gamma"):
        tp = json.loads((root / study / "true_profiles.json").read_text())
        names = tp["full_parameter_names"]
        prof = tp["profiles"]["log10Gamma"]
        for j, (g, logl, x) in enumerate(zip(prof["grid"], prof["logl"], prof["xopt"])):
            if study.endswith("_gamma") and not study.endswith("peak_gamma") and PEAK[0] - 1e-9 <= g <= PEAK[1] + 1e-9:
                continue
            out.append((study, j, float(g), float(logl), names, [float(v) for v in x[:len(names)]]))
    return out


def task(node):
    study, j, g, logl, names, x = node
    t = time.time()
    if "error" in _model:
        return dict(study=study, node=j, log10Gamma=g, study_logl=logl, error=f"model: {_model['error']}", fatal=True,
                    seconds=0.0)
    try:
        model = _model["m"]
        sampled = list(model.parameterization.sampled_params())
        if sorted(sampled) != sorted(names):
            raise ValueError(f"sampled parameters {sampled} are not the study's {names}")
        r = model.logposterior(dict(zip(names, x)), make_finite=False)
        return dict(study=study, node=j, log10Gamma=g, study_logl=logl, logpost=float(r.logpost),
                    chi2={lk: -2 * float(v) for lk, v in zip(model.likelihood, r.loglikes)},
                    minus2_logprior=-2 * float(sum(r.logpriors)), params=dict(zip(names, x)), seconds=time.time() - t)
    except Exception as error:                     # noqa: BLE001 - recorded with the node
        return dict(study=study, node=j, log10Gamma=g, study_logl=logl, error=f"{type(error).__name__}: {error}",
                    seconds=time.time() - t)


def table(rows):
    """Delta chi^2 by likelihood against the plateau node (log10Gamma = -2), one line a node."""
    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["log10Gamma"])
    if not ok:
        return []
    ref = min(ok, key=lambda r: abs(r["log10Gamma"] + 2))
    lks = list(ref["chi2"])
    short = [lk.split(".")[-1][:10] for lk in lks]
    lines = [f"   Delta chi^2 against log10Gamma {ref['log10Gamma']:+.2f} (negative: better than the plateau)",
             "   log10G   total  " + " ".join(f"{s:>10s}" for s in short) + "   prior  recomputed-study"]
    for r in ok:
        tot = -2 * (r["logpost"] - ref["logpost"])
        lines.append(f"   {r['log10Gamma']:6.3f} {tot:7.3f}  "
                     + " ".join(f"{r['chi2'][lk] - ref['chi2'][lk]:10.3f}" for lk in lks)
                     + f" {r['minus2_logprior'] - ref['minus2_logprior']:7.3f}  {r['logpost'] - r['study_logl']:+8.3f}"
                     + ("  <-- CHECK" if abs(r["logpost"] - r["study_logl"]) > CHECK else ""))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--only", nargs="+", choices=list(RUN_YAML), default=list(RUN_YAML))
    a = p.parse_args()
    import classy
    sys.path.insert(0, str(KIT))
    import client_paths  # noqa: F401  CLiENT's hpc/ tools
    from boltzmann_build import location_problem

    problem = location_problem("classy", classy.__file__)
    if problem:
        raise SystemExit(f"FATAL: {problem} (source runs/dncdm_proxy_pp/job.env first)")
    rows = {}
    for ds in a.only:
        todo = sorted(nodes(ds), key=lambda n: -n[2])           # large Gamma (the slow nodes) first
        print(f"== {ds}: {len(todo)} nodes on {a.workers} workers, OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}; "
              f"classy from {classy.__file__}", flush=True)
        rows[ds] = []
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(RUN_YAML[ds],)) as pool:
            for r in pool.imap_unordered(task, todo, chunksize=1):
                if r.get("fatal"):
                    pool.terminate()
                    raise SystemExit(f"FATAL: {ds}: the Cobaya model could not be built: {r['error']}")
                rows[ds].append(r)
                if "error" in r:
                    print(f"   {r['study']} node {r['node']} ({r['log10Gamma']:.4f}): ERROR {r['error']}", flush=True)
        rows[ds].sort(key=lambda r: r["log10Gamma"])
        bad = sum("error" in r for r in rows[ds])
        print(f"   {ds}: {len(rows[ds]) - bad} nodes evaluated, {bad} errors; median "
              f"{sorted(r['seconds'] for r in rows[ds])[len(rows[ds]) // 2]:.0f} s a node", flush=True)
        print("\n".join(table(rows[ds])), flush=True)
    out = KIT / "logs" / f"node_breakdown_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.write_text(json.dumps(dict(rows=rows, classy=classy.__file__), indent=1))
    print(f"\n   written {out.relative_to(KIT)}", flush=True)


if __name__ == "__main__":
    main()
