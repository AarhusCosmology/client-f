"""The cost of one evaluation of the true likelihood against the number of threads (section 5.3 of the paper).

    source runs/dncdm_proxy_wc/job.env                   # from this folder: the run's CLASS++ build
    python smoke/eval_threads.py [RUN] [--threads 1 3] [--repeat 1]

RUN is dncdm_proxy_wc by default. The point is the best fit of the run's true profile of log10Gamma around the peak
(results_profiles/dncdm_proxy_<ds>_peak_gamma/true_profiles.json, best_x). Each thread count runs in a fresh process,
since OpenMP reads OMP_NUM_THREADS once. The script also times the background of CLASS++ alone (one thread, the run's
settings and the same point), which is the serial part that more threads cannot shorten. The paper's numbers were
measured on a laptop (Apple M3 Pro) and converted to a q64 core with the factor of paper/cost_table.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def point(run):
    ds = run.rsplit("_", 1)[-1]
    best = json.loads((HERE / "results_profiles" / f"dncdm_proxy_{ds}_peak_gamma" / "true_profiles.json").read_text())
    return dict(zip(best["full_parameter_names"], best["best_x"])), best["best_logl"]


def child(run, what):
    import yaml
    from cobaya.model import get_model

    os.chdir(HERE)                                     # the yaml's python_path (likelihoods/) is relative to this folder
    info = yaml.safe_load((HERE / "runs" / run / "cobaya.yaml").read_text())
    x, _ = point(run)
    if what == "background":
        from classy import Class
        args = dict(info["theory"]["classy"].get("extra_args") or {})
        cosmo = {k: v for k, v in x.items() if k in ("omega_b", "omega_cdm", "H0")}   # what the background reads
        cosmo["dncdm1.log10Gamma"] = x["log10Gamma"]                                     # as cobaya.yaml passes it
        c = Class()
        c.set({**args, **cosmo, "output": ""})
        t0 = time.time()
        c.compute(["background"])
        print(json.dumps({"what": what, "seconds": time.time() - t0}))
        return
    model = get_model(info)
    t0 = time.time()
    logpost = model.logpost(x)
    print(json.dumps({"what": what, "seconds": time.time() - t0, "logpost": logpost}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="dncdm_proxy_wc")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(a.run, a.child)
    _, best = point(a.run)
    print(f"{a.run}: best fit of the true profile, logpost {best:.3f}")
    jobs = [(n, "logpost") for n in a.threads] * a.repeat + [(1, "background")]
    for n, what in jobs:
        env = dict(os.environ, OMP_NUM_THREADS=str(n))
        out = subprocess.run([sys.executable, "-I", __file__, a.run, "--child", what], env=env, capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()[-1]
        r = json.loads(out)
        extra = f", logpost {r['logpost']:.3f}" if "logpost" in r else ""
        print(f"   {what:10s} {n} thread{'s' if n > 1 else ' '}: {r['seconds']:7.1f} s{extra}")


if __name__ == "__main__":
    main()
