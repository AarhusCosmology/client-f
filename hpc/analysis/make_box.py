"""The training box of a run from a centre and widths: centre +- N widths, cut at the prior ranges.

    python analysis/make_box.py RUN --centre results/RUN_centre/centre.json [--half 6] [--widen NAME=FACTOR ...] [--force]

centre.json is what analysis/find_centre.py wrote: for every parameter its value (the best fit
with the cosmology fixed, or the fixed value itself), the width from a covariance of an
earlier chain, and the width of the log-posterior along that parameter alone at the best fit.
The box is, for every parameter,

    [ max(prior lower, centre - N width),  min(prior upper, centre + N width) ],   N = --half

with these widths:
  a parameter in the covariance      its width there (marginal: what a chain's spread was)
  a parameter the covariance lacks   its conditional width times --lacking (default 2): a
                                     conditional width is never above the marginal one. At a
                                     prior edge, where there is no such width: the distance
                                     inside at which the log-posterior is 1 lower, times --lacking
  a fixed parameter (the cosmology)  its width in the covariance; without one the run cannot
                                     be boxed and this stops
  a parameter whose conditional width is more than 1.1 times the covariance's   the
                                     conditional one: the covariance is too narrow there
Writes runs/RUN/domain_bounds.json (refused if it exists, without --force) and
runs/RUN/domain_notes.md with the table. Changes nothing else.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "analysis"))
from find_centre import prior_of, read_covmat  # noqa: E402


def box(centre, covariance_widths, half=6.0, lacking=2.0, widen=None):
    """[(name, lower, upper, row of the notes)] for every parameter of a centre.json."""
    widen = widen or {}
    rows = []
    for name, p in centre["parameters"].items():
        prior = p.get("prior") or {}
        if p.get("inert"):
            raise SystemExit(f"{name}: the log-posterior does not depend on it (centre.json): fix it in the run's Cobaya "
                             "file and find the centre again, rather than give a parameter that does nothing a box")
        width, source = covariance_widths.get(name), "covariance"
        conditional, one_sided = p.get("conditional_width"), p.get("one_sided_width")
        if width is None:
            if conditional is not None:
                width, source = lacking * conditional, f"{lacking:g} x conditional"
            elif one_sided is not None:
                width, source = lacking * one_sided, f"{lacking:g} x distance of a drop of 1 from the edge"
            else:
                raise SystemExit(f"{name}: no width in the covariance and none measured at the best fit: give it one (--width {name}=W)")
        elif conditional is not None and conditional > 1.1 * width:
            width, source = conditional, "conditional (above the covariance's)"
        if name in widen:
            width, source = width * widen[name], source + f", x {widen[name]:g}"
        low, high = p["value"] - half * width, p["value"] + half * width
        cut = []
        if prior.get("lower") is not None and low < float(prior["lower"]):
            low, cut = float(prior["lower"]), cut + ["lower"]
        if prior.get("upper") is not None and high > float(prior["upper"]):
            high, cut = float(prior["upper"]), cut + ["upper"]
        rows.append(dict(name=name, centre=p["value"], width=width, width_from=source, lower=low, upper=high,
                         cut_at_prior=cut, fixed_in_fit=bool(p.get("fixed")), at_prior_edge=p.get("at_prior_edge"),
                         prior=prior, widths_in_box=(high - low) / width))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--centre", type=Path, required=True)
    parser.add_argument("--covmat", type=Path, help="default: the one find_centre.py used")
    parser.add_argument("--half", type=float, default=6.0)
    parser.add_argument("--lacking", type=float, default=2.0)
    parser.add_argument("--widen", nargs="*", default=[], help="NAME=FACTOR on a parameter's width")
    parser.add_argument("--width", nargs="*", default=[], help="NAME=WIDTH for a parameter without one")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    folder = HERE / "runs" / args.run
    target = folder / "domain_bounds.json"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; --force writes a new one")
    centre = json.loads(args.centre.read_text())
    covmat = args.covmat or (Path(centre["covmat"]) if centre.get("covmat") else None)
    widths = {}
    if covmat is not None:
        names, matrix = read_covmat(covmat)
        widths = dict(zip(names, np.sqrt(np.diag(matrix)).tolist()))
    widths.update({item.partition("=")[0]: float(item.partition("=")[2]) for item in args.width})
    # a fixed parameter has no prior entry in centre.json: take it from the run's Cobaya file
    import yaml

    info = yaml.safe_load((folder / "cobaya.yaml").read_text())["params"]
    for name, p in centre["parameters"].items():
        if "prior" not in p:
            p["prior"] = dict(zip(("kind", "lower", "upper", "loc", "scale"), prior_of(info[name])))
    rows = box(centre, widths, args.half, args.lacking, {item.partition("=")[0]: float(item.partition("=")[2]) for item in args.widen})

    order = list(info)                                   # the order of the Cobaya file, which is CLiENT's
    rows.sort(key=lambda r: order.index(r["name"]))
    target.write_text(json.dumps({r["name"]: [r["lower"], r["upper"]] for r in rows}, indent=1) + "\n")

    lines = [f"# Training box of {args.run}", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} by analysis/make_box.py from {args.centre} "
             f"(best fit with {sum(r['fixed_in_fit'] for r in rows)} parameters fixed; log-posterior {centre.get('logpost', float('nan')):.2f}) "
             f"and the widths of {covmat}.", "",
             f"Every parameter: centre +- {args.half:g} widths, cut at its prior range where that is nearer. "
             f"{sum(bool(r['cut_at_prior']) for r in rows)} of {len(rows)} ranges are cut by a prior edge.", "",
             "| parameter | centre | width | width from | lower | upper | box, in widths | cut at prior | prior |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        prior = r["prior"]
        shown = (f"N({prior['loc']:g}, {prior['scale']:g})" if prior.get("kind") == "gauss"
                 else f"[{prior.get('lower')}, {prior.get('upper')}]")
        lines.append(f"| {r['name']} | {r['centre']:.6g}{' (fixed in the fit)' if r['fixed_in_fit'] else ''} | {r['width']:.4g} | {r['width_from']} | "
                     f"{r['lower']:.6g} | {r['upper']:.6g} | {r['widths_in_box']:.1f} | {', '.join(r['cut_at_prior']) or '-'} | {shown} |")
    (folder / "domain_notes.md").write_text("\n".join(lines) + "\n")
    print(f"written: {target} ({len(rows)} parameters) and {folder / 'domain_notes.md'}")
    cut = [r["name"] for r in rows if r["cut_at_prior"]]
    print(f"cut by a prior edge: {', '.join(cut) or 'none'}")
    other = [f"{r['name']} ({r['width_from']})" for r in rows if r["width_from"] != "covariance"]
    print(f"width not from the covariance: {', '.join(other) or 'none'}")


if __name__ == "__main__":
    main()
