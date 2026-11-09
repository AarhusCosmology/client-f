"""chi^2 of every likelihood at given points: where does a model's improvement come from?

    python smoke/chi2_breakdown.py POINTS DATA [--reference LCDM] [--only LIKELIHOOD ...] [--limit N]

POINTS is a JSON file (smoke/chi2_points_s5.json: the S+5 network best fits of the SIDR + DRMD combined
networks, made on the laptop by results/20261022_s5_aic_preview/make_points.py) with, for each data set
DATA, the Cobaya file and the points (label, the sampled parameters, the network's log-posterior). The
run's Cobaya model is evaluated at each point (CLASS++ as the shell's PYTHONPATH gives it: source the
run's job.env for the patched build). Printed per point: chi^2 = -2 log L of every likelihood, their sum,
-2 log prior, and the true minus the network log-posterior; then every likelihood's chi^2 minus that at
the --reference point. --only keeps the named likelihoods (a test on the laptop, which has no PR4
lensing or SN data); keep planck_NPIPE_highl_CamSpec.TTTEEE among them, or its nuisance parameters go to
CLASS, which refuses them. Writes logs/chi2_breakdown_<DATA>_<job>.json. Made 22 October 2026; tested on
the laptop with low-l TT, CamSpec and DESI at two points.
"""
import argparse
import json
import os
import time
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("points", type=Path)
    p.add_argument("data")
    p.add_argument("--reference", default="LCDM", help="label of the point the differences are taken from")
    p.add_argument("--only", nargs="+")
    p.add_argument("--limit", type=int, help="the first N points only (a test)")
    args = p.parse_args()

    spec = json.loads(args.points.read_text())["data"][args.data]
    info = yaml.safe_load(Path(spec["cobaya"]).read_text())
    info.pop("sampler", None)
    info.pop("output", None)
    if args.only:
        missing = set(args.only) - set(info["likelihood"])
        if missing:
            raise SystemExit(f"not likelihoods of {spec['cobaya']}: {sorted(missing)}")
        info["likelihood"] = {k: v for k, v in info["likelihood"].items() if k in args.only}

    import classy
    from cobaya.model import get_model

    print(f"== {args.data}: {spec['cobaya']}, network {spec['network']}; classy from {classy.__file__}", flush=True)
    model = get_model(info)
    names = list(model.parameterization.sampled_params())
    likelihoods = list(model.likelihood.keys())
    points = spec["points"][:args.limit] if args.limit else spec["points"]
    results = []
    for point in points:
        missing = [n for n in names if n not in point["params"]]
        if missing:
            raise SystemExit(f"point {point['label']} has no value for {missing}")
        start = time.time()
        r = model.logposterior({n: point["params"][n] for n in names}, make_finite=False)
        if r.loglikes is None or len(r.loglikes) == 0:
            print(f"   {point['label']}: outside the prior of {spec['cobaya']}; not evaluated", flush=True)
            continue
        chi2 = {name: -2 * float(v) for name, v in zip(likelihoods, r.loglikes)}
        entry = dict(label=point["label"], chi2=chi2, chi2_total=sum(chi2.values()),
                     minus2_logprior=-2 * float(sum(r.logpriors)), logpost=float(r.logpost),
                     network_logpost=point.get("network_logpost"), seconds=time.time() - start,
                     params=point["params"])
        results.append(entry)
        net = entry["network_logpost"]
        print(f"   {point['label']:12s} chi2 {entry['chi2_total']:10.2f}   -2 ln prior {entry['minus2_logprior']:8.2f}   "
              f"log-posterior {entry['logpost']:10.3f}"
              + ("" if net is None or args.only else f"   true - network {entry['logpost'] - net:+.3f}")
              + f"   ({entry['seconds']:.0f} s)", flush=True)

    Path("logs").mkdir(exist_ok=True)
    out = Path(f"logs/chi2_breakdown_{args.data}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    out.write_text(json.dumps(dict(data=args.data, cobaya=spec["cobaya"], network=spec["network"],
                                   classy=classy.__file__, likelihoods=likelihoods, results=results), indent=1))

    short = [name.split(".")[-1][:14] for name in likelihoods]
    print(f"\n== {args.data}: chi^2 per likelihood")
    print(f"   {'point':12s}" + "".join(f"{s:>16s}" for s in short) + f"{'total':>12s}")
    for e in results:
        print(f"   {e['label']:12s}" + "".join(f"{e['chi2'][n]:16.2f}" for n in likelihoods) + f"{e['chi2_total']:12.2f}")
    reference = next((e for e in results if e["label"] == args.reference), None)
    if reference is not None:
        print(f"\n== {args.data}: chi^2 minus that at {args.reference}")
        print(f"   {'point':12s}" + "".join(f"{s:>16s}" for s in short) + f"{'total':>12s}")
        for e in results:
            d = [e["chi2"][n] - reference["chi2"][n] for n in likelihoods]
            print(f"   {e['label']:12s}" + "".join(f"{x:+16.2f}" for x in d) + f"{sum(d):+12.2f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
