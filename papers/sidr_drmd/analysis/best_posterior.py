"""The posterior of a network from BEST's weighted samples, pooled over seeds, beside the report's chain.

    python analysis/best_posterior.py RUN_DIR [--network 0] [--parameters 12]
                                      [--split logGeff:-2.5 --split delta_Neff_drmd:0.1] [--out DIR]

`client.py evidence` leaves BEST's dead points with their posterior weights beside its JSON
(RUN_DIR/evidence/evidence_it_N_nliveM[_TAG].txt). They are a posterior sample of the network
inside the same region as the report's chain, but without its cells: nested sampling shrinks
from the whole region, so its live points populate every mode in proportion to its mass, and
nothing needs to cross between modes. The report's chain of a region with several ellipsoids
is mixed from per-cell chains by bridge-sampling evidences (CLiENT's sampling/mixture.py),
which can be marked as not reliable. This program reads every BEST run of the same target
(seeds: `submit_evidence.sh ... --seed 43 --tag seed43`; another target, such as `--region E`,
is left out) and writes

  best_posterior_it_N.md    the runs (seed, ln Z, effective samples), the shares of the posterior
                            above or below each --split and in each region cell, run by run with
                            the scatter over seeds, and the parameter table, beside the report's
                            chain (posterior_it_N.npz) where that samples the same network
  best_posterior_it_N.json  the same numbers
  best_posterior_it_N.npz   the pooled weighted samples (x, weights, names, run)
  best_corner_it_N.png      the first --parameters parameters, BEST against the report's chain

into RUN_DIR/analysis (or --out). BEST's share of a mode carries a log error of roughly
sqrt(H / n_k), n_k the live points in that mode, so a minor mode is uncertain by tens of percent
in one run; the scatter over seeds measures it. Each run enters the pool with weight n_live.
Nothing of CLiENT is imported and no network is loaded: it runs in seconds, on the laptop after
sync_from_hpc.sh or on the cluster (analysis/queue_best_seeds.sh submits it after the seeds).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def runs_of(run, network):
    """BEST runs of network `network` with the target of the untagged one (or of the first)."""
    found = []
    for path in sorted((run / "evidence").glob(f"evidence_it_{network}_nlive*.json")):
        match = re.fullmatch(rf"evidence_it_{network}_nlive(\d+)(?:_(.+))?\.json", path.name)
        if not match or not path.with_suffix(".txt").exists():
            continue
        saved = json.loads(path.read_text())
        found.append(dict(path=path, tag=match.group(2) or "", saved=saved,
                          target=json.dumps(saved.get("target"), sort_keys=True)))
    if not found:
        raise SystemExit(f"no BEST run of network {network} with its samples in {run / 'evidence'}")
    found.sort(key=lambda r: (r["tag"] != "", r["tag"]))
    reference = found[0]["target"]
    kept, notes = [], []
    for r in found:
        if r["target"] != reference:
            notes.append(f"{r['path'].name}: another target than {found[0]['path'].name} (left out)")
        elif r["saved"].get("iteration_cap_reached"):
            notes.append(f"{r['path'].name}: stopped at BEST's iteration cap (left out)")
        else:
            kept.append(r)
    seeds = [r["saved"].get("seed") for r in kept]
    if len(set(seeds)) != len(seeds):
        notes.append(f"runs share a seed ({seeds}): their scatter is not a measurement")
    return kept, notes


def load_samples(r):
    """(weights normalised to 1, x) of one run; the columns are checked against its names."""
    names = list(r["saved"]["param_names"])
    listed = [line.split("\t")[0] for line in r["path"].with_suffix(".paramnames").read_text().splitlines() if line.strip()]
    if listed[:len(names)] != names:
        raise SystemExit(f"{r['path'].with_suffix('.paramnames')} does not list {names}")
    table = np.loadtxt(r["path"].with_suffix(".txt"))
    if table.shape[1] != 2 + len(names):
        raise SystemExit(f"{r['path'].with_suffix('.txt')}: {table.shape[1]} columns, expected {2 + len(names)}")
    w, x = table[:, 0], table[:, 2:]
    if not (np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0):
        raise SystemExit(f"{r['path'].with_suffix('.txt')}: weights not finite and non-negative")
    keep = w > w.max() * 1e-12
    return w[keep] / w[keep].sum(), x[keep]


def region_labels(run, saved):
    """(labels function, file) of the region recorded with the run, or (None, reason)."""
    region = saved.get("region")
    if not isinstance(region, dict) or not region.get("file"):
        return None, "the network was integrated without a region"
    file = run / region["file"]
    if not file.exists():
        return None, f"{file} is not here"
    if region.get("sha256") and hashlib.sha256(file.read_bytes()).hexdigest() != region["sha256"]:
        return None, f"{file} is not the file the evidence recorded (sha256)"
    with np.load(file) as f:
        means, reach, lower = f["means"], f["reach"].ravel(), f["lower"]
        whiten = f["whiten"].reshape(len(means), means.shape[1], -1)
        width = f["upper"] - lower

    def labels(x):
        # CLiENT's sampling/region.py, Region.radii and Region.labels (nearest ellipsoid)
        u = (np.asarray(x, float) - lower) / width
        z = np.einsum("knd,kde->kne", u[None] - means[:, None], whiten)
        return (np.sqrt((z ** 2).sum(axis=2)) / reach[:, None]).argmin(axis=0)
    return labels, str(file.relative_to(run)), len(means)


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    c = np.cumsum(w[order]) - 0.5 * w[order]
    return np.interp(q, c / w.sum(), x[order])


def report_chain(run, network, target):
    """The report's chain of the same network (x, sampling info) or (None, None, reason)."""
    file = run / "analysis" / f"posterior_it_{network}.npz"
    if not file.exists():
        return None, None, f"no {file.name}"
    stored = np.load(file, allow_pickle=True)
    recorded = json.loads(str(stored["target_json"])) if "target_json" in stored else None
    if not recorded or not target or recorded.get("model_sha256") != target.get("model_sha256") \
            or (recorded.get("region") or {}).get("sha256") != (target.get("region") or {}).get("sha256"):
        return None, None, f"{file.name} samples another network or region than the evidence"
    names = [str(n) for n in stored["names"]]
    if names != list(target["names"]):
        return None, None, f"{file.name} has other parameters"
    sampling = json.loads(str(stored["sampling_json"])) if "sampling_json" in stored else None
    return stored["x"].astype(float), sampling, None


def fmt(v, spec):
    return "-" if v is None or not np.isfinite(v) else format(v, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--network", type=int, default=0)
    parser.add_argument("--parameters", type=int, default=12, help="the first N parameters in the corner plot")
    parser.add_argument("--split", action="append", default=[], metavar="NAME:VALUE",
                        help="report the posterior's share above VALUE of NAME (repeatable)")
    parser.add_argument("--contours", type=float, nargs="*", default=[0.68, 0.95])
    parser.add_argument("--smooth", type=float, default=0.4, help="getdist's smoothing scale, in bins")
    parser.add_argument("--no-corner", action="store_true")
    parser.add_argument("--out", type=Path, help="directory for the outputs (default RUN_DIR/analysis)")
    args = parser.parse_args()

    run, n = args.run, args.network
    kept, notes = runs_of(run, n)
    names = list(kept[0]["saved"]["param_names"])
    target = kept[0]["saved"].get("target")
    splits = []
    for s in args.split:
        name, _, value = s.rpartition(":")
        if name not in names:
            raise SystemExit(f"--split {s}: no parameter {name}")
        splits.append((name, float(value)))

    labelled = region_labels(run, kept[0]["saved"])
    labels, region_file = (labelled[0], labelled[1]) if labelled[0] else (None, None)
    n_cells = labelled[2] if labels else 0
    if not labels:
        notes.append(f"no region cells: {labelled[1]}")

    per_run, pooled_x, pooled_w, pooled_run = [], [], [], []
    a = np.array([r["saved"]["n_live"] for r in kept], float)
    a /= a.sum()
    for k, r in enumerate(kept):
        w, x = load_samples(r)
        entry = dict(file=r["path"].name, tag=r["tag"], seed=r["saved"].get("seed"), n_live=r["saved"]["n_live"],
                     lnZ=r["saved"]["lnZ"], sigma_lnZ=r["saved"]["sigma_lnZ"], H=r["saved"].get("information_H"),
                     samples=int(len(w)), ess=float(1 / np.sum(w ** 2)),
                     splits={f"{name}>{value:g}": float(w[x[:, names.index(name)] > value].sum()) for name, value in splits},
                     mean=(w @ x).tolist())
        if labels:
            lab = labels(x)
            entry["cells"] = [float(w[lab == c].sum()) for c in range(n_cells)]
        per_run.append(entry)
        pooled_x.append(x), pooled_w.append(a[k] * w), pooled_run.append(np.full(len(w), k))
    x, w, which = np.concatenate(pooled_x), np.concatenate(pooled_w), np.concatenate(pooled_run)

    def combine(values):
        """Pooled value (n_live weights) and the scatter over runs: (pooled, sd of one run, error of the pool)."""
        v = np.asarray(values, float)
        if len(v) < 2:
            return float(a @ v), None, None
        sd = float(v.std(ddof=1))
        return float(a @ v), sd, float(sd * np.sqrt(np.sum(a ** 2)))

    chain, sampling, chain_note = report_chain(run, n, target)
    if chain_note:
        notes.append(f"no comparison with the report's chain: {chain_note}")
    bridge = None
    if sampling and sampling.get("cells"):
        bridge = {c["cell"]: (c.get("weight"), c.get("efficiency")) for c in sampling["cells"]}

    summary = dict(run=run.resolve().name, network=n, runs=per_run, notes=notes, region=region_file, splits={}, cells=[],
                   parameters=[])
    for name, value in splits:
        key = f"{name}>{value:g}"
        pooled, sd, err = combine([e["splits"][key] for e in per_run])
        summary["splits"][key] = dict(best=pooled, seed_sd=sd, error=err,
                                      chain=None if chain is None else float(np.mean(chain[:, names.index(name)] > value)))
    if labels:
        chain_lab = labels(chain) if chain is not None else None
        for c in range(n_cells):
            pooled, sd, err = combine([e["cells"][c] for e in per_run])
            summary["cells"].append(dict(cell=c, best=pooled, seed_sd=sd, error=err,
                                         bridge=None if not bridge or c not in bridge else bridge[c][0],
                                         efficiency=None if not bridge or c not in bridge else bridge[c][1],
                                         chain=None if chain_lab is None else float(np.mean(chain_lab == c))))
    for j, name in enumerate(names):
        mean = float(w @ x[:, j])
        sd = float(np.sqrt(w @ (x[:, j] - mean) ** 2))
        lo, hi = weighted_quantile(x[:, j], w, [0.16, 0.84])
        _, mean_sd, mean_err = combine([e["mean"][j] for e in per_run])
        row = dict(name=name, mean=mean, sd=sd, low68=float(lo), high68=float(hi), seed_error_of_mean=mean_err)
        if chain is not None:
            row.update(chain_mean=float(chain[:, j].mean()), chain_sd=float(chain[:, j].std()),
                       shift_sigma=float((mean - chain[:, j].mean()) / sd) if sd > 0 else None)
        summary["parameters"].append(row)

    out = args.out or run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"best_posterior_it_{n}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / f"{stem}.npz", x=x.astype(np.float32), weights=w, names=np.array(names), run=which,
                        files=np.array([e["file"] for e in per_run]), target_json=json.dumps(target))

    lines = [f"# {summary['run']}: posterior of network {n} from BEST's samples", "",
             "Weighted dead points of every BEST run of this network and target, pooled with weight n_live; "
             "the report's chain (analysis/posterior_it_N.npz, cells mixed by bridge-sampling evidences) beside it. "
             "Errors \"over seeds\" are the scatter of the runs: of one run, and of the pool.", "",
             "## BEST runs", "", "| file | seed | n_live | ln Z | error | H | weighted samples | effective samples |",
             "|---|---|---|---|---|---|---|---|"]
    for e in per_run:
        lines.append(f"| {e['file']} | {e['seed']} | {e['n_live']} | {e['lnZ']:.2f} | {e['sigma_lnZ']:.2f} | "
                     f"{fmt(e['H'], '.1f')} | {e['samples']} | {e['ess']:.0f} |")
    if len(per_run) > 1:
        z = [e["lnZ"] for e in per_run]
        lines += ["", f"ln Z over the runs: mean {np.mean(z):.2f}, scatter {np.std(z, ddof=1):.2f} "
                      f"(BEST's own error {np.mean([e['sigma_lnZ'] for e in per_run]):.2f})."]
    if splits:
        lines += ["", "## Shares of the posterior", "",
                  "| share | " + " | ".join(e["tag"] or "untagged" for e in per_run) + " | BEST pooled | scatter of one run | "
                  "error of the pool | report's chain |", "|---" * (len(per_run) + 5) + "|"]
        for key, s in summary["splits"].items():
            lines.append(f"| {key} | " + " | ".join(f"{e['splits'][key]:.3f}" for e in per_run)
                         + f" | {s['best']:.3f} | {fmt(s['seed_sd'], '.3f')} | {fmt(s['error'], '.3f')} | {fmt(s['chain'], '.3f')} |")
    if summary["cells"]:
        lines += ["", f"## Region cells ({region_file}: the points nearest each ellipsoid)", "",
                  "| cell | " + " | ".join(e["tag"] or "untagged" for e in per_run) + " | BEST pooled | scatter of one run | "
                  "report's bridge weight | its efficiency |", "|---" * (len(per_run) + 5) + "|"]
        for c in summary["cells"]:
            if c["best"] < 1e-4 and (c["bridge"] or 0) < 1e-4:
                continue
            lines.append(f"| {c['cell']} | " + " | ".join(f"{e['cells'][c['cell']]:.4f}" for e in per_run)
                         + f" | {c['best']:.4f} | {fmt(c['seed_sd'], '.4f')} | {fmt(c['bridge'], '.4f')} | {fmt(c['efficiency'], '.3f')} |")
        lines.append("")
        lines.append("Cells holding less than 1e-4 by both estimates are not listed.")
    lines += ["", "## Parameters", "",
              "| parameter | mean | sd | 68% interval | error of the mean over seeds | report's chain: mean | sd | shift (BEST sd) |",
              "|---|---|---|---|---|---|---|---|"]
    for p in summary["parameters"]:
        lines.append(f"| {p['name']} | {p['mean']:.6g} | {p['sd']:.3g} | {p['low68']:.6g} to {p['high68']:.6g} | "
                     f"{fmt(p['seed_error_of_mean'], '.2g')} | {fmt(p.get('chain_mean'), '.6g')} | {fmt(p.get('chain_sd'), '.3g')} | "
                     f"{fmt(p.get('shift_sigma'), '+.2f')} |")
    if notes:
        lines += ["", "## Notes", ""] + [f"- {t}" for t in notes]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_corner:
        import matplotlib
        matplotlib.use("Agg")
        from getdist import MCSamples, plots
        listed = dict(line.split("\t")[:2] for line in kept[0]["path"].with_suffix(".paramnames").read_text().splitlines()
                      if "\t" in line)
        shown = names[:args.parameters]
        run_name = re.sub(r"^\d{8}_\d{6}_", "", run.resolve().name)
        box_file = HERE / "runs" / run_name / "domain_bounds.json"
        box = json.loads(box_file.read_text()) if box_file.exists() else {}
        settings = dict(contours=sorted(args.contours), smooth_scale_2D=args.smooth, smooth_scale_1D=args.smooth)
        common = dict(names=shown, labels=[listed.get(s, s) for s in shown],
                      ranges={s: tuple(box[s]) for s in shown if s in box}, settings=settings)
        columns = [names.index(s) for s in shown]
        samples = [MCSamples(samples=x[:, columns], weights=w, label=f"BEST ({len(per_run)} run{'s' * (len(per_run) > 1)})", **common)]
        if chain is not None:
            samples.append(MCSamples(samples=chain[:, columns], label="report's chain (bridge-weighted cells)", **common))
        g = plots.get_subplot_plotter(width_inch=max(9, 1.5 * len(shown)))
        g.settings.num_plot_contours = len(args.contours)
        g.settings.legend_fontsize = 14
        g.triangle_plot(samples, filled=[True, False][:len(samples)], contour_colors=["#1f5fa8", "#c0392b"][:len(samples)])
        g.export(str(out / f"best_corner_it_{n}.png"))
        print(f"written: {out / f'best_corner_it_{n}.png'}")
    print(f"written: {out / (stem + '.md')}, .json, .npz")


if __name__ == "__main__":
    main()
