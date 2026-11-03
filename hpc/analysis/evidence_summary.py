"""The evidences of a run's networks side by side: does ln Z settle as the run goes on?

    python analysis/evidence_summary.py RUN_DIR [--out DIR]

Reads only what is saved: RUN_DIR/evidence/evidence_it_N_nliveM[_TAG].json (`client.py
evidence`, the BEST nested sampler on the network), RUN_DIR/boost/boost_it_N.json and
RUN_DIR/analysis/true_check_it_N.npz. Writes evidence.md, evidence.json and evidence.png to
RUN_DIR/analysis (or --out). Three numbers per network, all the integral of exp(log-posterior)
over the parameters in the run's own prior convention (docs/evidence.md of CLiENT):

  ln Z (BEST)       nested sampling of the network; its error is BEST's sqrt(H / n_live), which is
                    not a test: a second seed (--seed 43 --tag seed43) is, and is listed beside it
  ln Z (boost)      what the boost drawn from this network measured itself, by bridge sampling of
                    its T = 1 chain inside the region: an independent estimate of the same integral
  ln Z (corrected)  ln Z (BEST) + ln <exp(truth - network)> over posterior samples of the
                    network with true values (analysis/true_check.py): the evidence of the TRUE
                    likelihood, as far as those samples carry it. An unbiased network with a
                    scatter s under-rates the evidence by about s^2 / 2, so the networks' own
                    evidences rise as the scatter falls even when nothing else changes; the
                    corrected ones should agree. The error of the correction is a bootstrap of
                    the samples, and it is only as good as their effective number. The mean
                    is over every sample drawn: one the likelihood refused weighs nothing and
                    is still counted (see `correction`), and the notes under the table say
                    how many there were.

Which true check corrects which evidence. Since 11 October 2026 an evidence and a true check
record the target they were made on (the network's file, the domain, the box and the region:
sampling/provenance.py). Where both do, they must be the same, or there is no correction. A
result made before that records none: it is joined to the other by the network's number, as it
was then, and the notes name those networks. Withholding them instead emptied the table of
every existing run, whose evidences take hours each. A file that records something that
cannot be used (damaged, or of a layout this script does not know) is neither: it gets no
correction, and the note says which file.

How the samples of a true check were drawn. Since 13 October 2026 a true check records the
sampler of its chain, and whether the chain can carry a mean over the posterior
(`sampling_correction_unavailable`: the weight of a cell of a region with several ellipsoids
was not reliable). Where it cannot there is no correction, and the note gives the reason. A
true check from before that records neither and is used as it was, with one note: when the
network's region has several ellipsoids, its chain may have kept the shares of its start
between them (chain_sampling.py), and the correction is then over those shares.

Where the correction does not reach. The true check's samples are drawn from the network's
posterior by its chain; the evidence integrates the network with its own sampler. When the
evidence's posterior (its weighted chain of nested samples, beside its JSON) lies largely where
the check has no sample, at network values above the highest of them or farther from their mean
than the farthest, the correction says nothing about that mass and there is none (`unsampled` in
evidence.json has both shares). Were the samples draws of the posterior the evidence integrated,
each share would be about 1 / (n + 1). Network 21 of the SINU branch of 13 October had 99.9%: a
maximum in a corner of the box, 148 above the truth there, that its chain never reached.

What is compared. The networks are compared with the last one, and their corrected evidences
with each other, whatever each was integrated over: they are estimates of one evidence, as far
as each region holds the posterior, and whether they agree is what the table is for. A note
says when they were not integrated over the same part of the box. Two evidences in different
prior conventions are not compared; the convention is the priors as they were defined, not
the box or the mass measured at its faces, which the same table of an evidence carries.
Runs of one network count as a test of the sampler's scatter only on the same target, in the
same convention, with the same stopping rule and tolerance, and when none stopped at its cap.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from chain_sampling import chain_sampling


def log_mean_exp(d):
    """ln of the mean of exp(d). An entry of -inf weighs nothing and is counted in the mean."""
    return float(logsumexp(d) - np.log(len(d)))


def refused_value_of(run):
    """The value a run's refused points carry (launch.py writes RUN/refused_value.json), or None."""
    try:
        return float(json.loads((Path(run) / "refused_value.json").read_text())["value"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def correction(file, rng, repeats=2000, refused_value=None):
    """ln <exp(truth - network)> at the posterior samples of a true check, its bootstrap error,
    the effective number of samples, and the scatter and mean of truth - network.

    The mean is over every point drawn. A point the likelihood refused is a sample of the
    network's posterior at which the true likelihood is zero: it has weight zero and stays in
    the denominator. Leaving it out would normalise over the points that succeeded and put the
    corrected evidence too high by -ln(1 - f) for a refused share f. Refused are the points the
    true check marked (`refused` in its file); in a file from before it marked them, a truth of
    -inf or equal to `refused_value`, the run's. A truth that is NaN or +inf is an evaluation
    that failed, not a refusal: what the point weighs is not known, and there is no correction.

    Returns a dict. `points` is every point, `refused` and `failed` are the counts, `scatter`
    and `mean` are of the finite differences at points that were not refused. Without a
    correction (the true check says its samples cannot carry one: the weight of a cell of the
    network's region was not reliable; a failed evaluation; fewer than 10 points with a
    difference) `value` is None and `unavailable` says why, every reason there is."""
    saved = np.load(file)
    truth, network = np.asarray(saved["truth"], float), np.asarray(saved["network"], float)
    if "refused" in saved.files:
        refused = np.asarray(saved["refused"], bool)
    else:
        refused = np.isneginf(truth) | (truth == refused_value if refused_value is not None else False)
    d = truth - network
    d[refused] = -np.inf                                  # weight zero, and still one of the points
    failed = ~refused & ~np.isfinite(d)
    used = ~refused & ~failed
    counts = dict(points=int(len(d)), refused=int(refused.sum()), failed=int(failed.sum()))
    # The producer records this independently of the target: evidence identity must not
    # change when a sampler changes. Legacy true checks keep their documented policy.
    reasons = []
    if "sampling_correction_unavailable" in saved:
        reasons.append(str(saved["sampling_correction_unavailable"].item()))
    if failed.any():
        reasons.append(f"{counts['failed']} of the {len(d)} true evaluations failed (NaN or +inf), "
                       "and a mean over every point drawn cannot leave them out")
    if any(reasons):
        return dict(value=None, **counts, unavailable="; ".join(filter(None, reasons)))
    if used.sum() < 10:
        return dict(value=None, **counts, unavailable=f"{int(used.sum())} of the {len(d)} points have a true value: fewer than 10")
    w = np.exp(d[used] - d[used].max())
    draws = rng.integers(0, len(d), (repeats, len(d)))
    resampled = np.array([log_mean_exp(d[row]) for row in draws])      # a draw of refused points alone is -inf
    return dict(value=log_mean_exp(d), error=float(np.std(resampled[np.isfinite(resampled)])),
                effective=float(w.sum() ** 2 / np.sum(w ** 2)), points=counts["points"], scatter=float(d[used].std()),
                mean=float(d[used].mean()), refused=counts["refused"], failed=counts["failed"])


# The share of an evidence's own posterior where its true check has no sample above which the
# check does not correct it. Were the check's n samples draws of the posterior the evidence
# integrated, the share above the highest of them, or farther out than the farthest, would be
# about 1 / (n + 1), 0.2% for 500: on the twelve good networks of the runs of 13 October (16, 31
# and 49 parameters) it was at most 0.5% and 0.75%, and 99.9% for the one network whose
# evidence lived on a maximum its chain never reached.
UNSAMPLED = 0.05


def unsampled(evidence_file, check_file):
    """What of an evidence's own posterior lies where its true check has no sample: its weighted
    chain of nested samples (beside its JSON) against the check's samples. `higher` is the share
    at network values above the highest of theirs, `farther` the share farther from their mean,
    in their covariance, than the farthest of them (None when their covariance is singular).
    Each sample's own distance is taken from the mean and covariance of the others: an evidence
    point is measured from a mean and covariance it did not enter, and so must a sample be.
    None when the chain or the samples' coordinates are not there, or name other parameters."""
    chain = Path(str(evidence_file)[:-len(".json")] + ".txt")
    if not chain.exists():
        return None
    with np.load(check_file, allow_pickle=False) as saved:
        if "x" not in saved.files or "names" not in saved.files:
            return None
        points, network = np.asarray(saved["x"], float), np.asarray(saved["network"], float)
        names = [str(name) for name in saved["names"]]
    order = json.loads(Path(evidence_file).read_text()).get("param_names")
    if not order or sorted(order) != sorted(names) or len(points) < 10 or not np.isfinite(network).any():
        return None
    rows = np.loadtxt(chain, ndmin=2)
    weight, value = rows[:, 0] / rows[:, 0].sum(), -rows[:, 1]
    samples = rows[:, 2:][:, [order.index(name) for name in names]]
    highest = float(network[np.isfinite(network)].max())
    found = dict(higher=float(weight[value > highest].sum()), farther=None, samples=len(points),
                 highest=highest, top=float(value.max()))
    covariance = np.atleast_2d(np.cov(points, rowvar=False))
    scales = np.linalg.eigvalsh(covariance)
    if scales.min() > 1e-12 * scales.max():
        inverse, centre = np.linalg.inv(covariance), points.mean(axis=0)

        def distance(x):
            return np.sqrt(np.einsum("ni,ij,nj->n", x - centre, inverse, x - centre))

        # Leave-one-out, in closed form. A sample's distance from a mean and covariance it is
        # part of is smaller, by a share that grows with the dimension: with 500 samples in 49
        # parameters the farthest of them left 1.9 to 5.1% of good networks' posteriors outside,
        # where 0.2% is expected, and withheld a correction that was sound.
        n = len(points)
        own = np.einsum("ni,ij,nj->n", points - centre, inverse, points - centre)
        with np.errstate(divide="ignore"):
            left_out = (n - 2) / (n - 1) * (n / (n - 1)) ** 2 * own / np.maximum(1 - n * own / (n - 1) ** 2, 0.0)
        found["farther"] = float(weight[distance(samples) > np.sqrt(left_out.max())].sum())
    return found


def unreached(found):
    """Why there is no correction, for an evidence whose posterior lies where its check has no sample."""
    where = [f"{100 * found['higher']:.1f}% at network values above the highest of the check's {found['samples']} samples "
             f"({found['highest']:.2f}; the evidence's chain reaches {found['top']:.2f})"]
    if found["farther"] is not None:
        where.append(f"{100 * found['farther']:.1f}% farther from their mean than the farthest of them")
    return (f"its own posterior lies where its true check has no sample ({'; '.join(where)}; about "
            f"{100 / (found['samples'] + 1):.1f}% each if the check sampled that posterior). The correction does not "
            "reach that mass, where the network was never checked against the truth: a maximum the truth lacks, "
            "which the network's posterior chain missed, looks like this")


TARGET_KEYS = {"schema", "model_sha256", "metadata_sha256", "names", "domain", "lower", "upper", "region",
               "temperature", "objective"}          # as CLiENT's sampling/provenance.py writes and reads them


def usable_target(target):
    """(target, None) for a recorded target that can be compared, (None, why not) for one that
    cannot. Something is recorded: this is not a result from before targets were."""
    if not isinstance(target, dict):
        return None, "recorded target is not a table of entries"
    if target.get("schema") != 1:
        return None, f"recorded target has schema {target.get('schema')!r}, which this summary does not know"
    if not TARGET_KEYS <= target.keys():
        return None, f"recorded target lacks {', '.join(sorted(TARGET_KEYS - target.keys()))}"
    return target, None


def recorded_target(file):
    """(target, problem) of a true check's file. (None, None): it records none, as every true
    check before 11 October 2026. (target, None): the target CLiENT's sampling/provenance.py
    wrote. (None, text): something is recorded that cannot be used, which is neither of the
    two and must not pass for the first. Read here, without CLiENT: this script reads only what
    is saved, wherever this folder lies."""
    with np.load(file, allow_pickle=False) as saved:
        if "target_json" not in saved:
            return None, None
        try:
            target = json.loads(str(saved["target_json"].item()))
        except (ValueError, TypeError, AttributeError):
            return None, "recorded target cannot be read"
    return usable_target(target)


def convention_of(saved):
    """What decides the prior convention an evidence is reported in: the convention's name, every
    parameter's prior as it was defined (its range and where that came from) and the constant
    that takes ln Z to that convention, which follows from them. The rows of `priors` also
    carry the integration box and the posterior mass measured at its faces. Those are not
    part of it: the mass is a sampled number, different in any two runs."""
    priors, correction = saved.get("priors"), saved.get("ln_prior_correction")
    if isinstance(priors, list):
        priors = [[row.get(key) for key in ("name", "lower", "upper", "source")] if isinstance(row, dict) else row
                  for row in priors]
    return dict(convention=saved.get("convention"), priors=priors,
                ln_prior_correction=correction if not isinstance(correction, (int, float)) else round(float(correction), 9))


def stopping_rule(saved):
    """What ended an evidence, as far as it is recorded: the kind of rule and its tolerance.
    Since 11 October 2026 the tolerance is the fraction of the evidence left in the live points
    (the output then has `live_fraction`); before, it was BEST's own, on ln(Z_live / Z_dead)."""
    return ["fraction of the evidence" if "live_fraction" in saved else "BEST's own", saved.get("tolerance")]


def same_convention(a, b):
    """Whether two evidences are reported in the same prior convention. Only then is their
    difference a difference of evidence."""
    return a.get("normalisation") == b.get("normalisation")


def integrated_over(entry):
    """The file of the region an evidence was integrated in, or None for the whole box. Its
    output named the file until 11 October 2026 and holds the resolved entry since."""
    region = entry.get("region")
    if isinstance(region, dict):
        region = region.get("file")
    return Path(region).name if region else None


def same_run_of_the_sampler(a, b):
    """Whether two evidences of one network differ by the sampler's scatter alone: the same
    target, prior convention and stopping rule (its kind and its tolerance). The recorded
    targets decide when both have one; otherwise what they were integrated over. A target
    that cannot be used is the same as nothing."""
    if not same_convention(a, b) or a.get("stop") != b.get("stop") or a.get("target_problem") or b.get("target_problem"):
        return False
    if a.get("target") and b.get("target"):
        return a["target"] == b["target"]
    return integrated_over(a) == integrated_over(b)


def collect(run):
    rng = np.random.default_rng(1)
    refused_value = refused_value_of(run)
    rows = {}
    for file in sorted(glob.glob(str(run / "evidence" / "evidence_it_*_nlive*.json"))):
        match = re.search(r"evidence_it_(\d+)_nlive(\d+)(?:_(.+))?\.json$", file)
        saved = json.loads(Path(file).read_text())
        # no target, or null: an evidence from before targets were recorded
        target, target_problem = (None, None) if saved.get("target") is None else usable_target(saved["target"])
        entry = dict(tag=match.group(3) or "", n_live=int(match.group(2)), lnZ=saved["lnZ"], sigma=saved["sigma_lnZ"],
                     lnZ_box=saved["lnZ_box"], H=saved["information_H"], dead=saved["n_dead"], minutes=saved["seconds"] / 60,
                     target=target, target_problem=target_problem,
                     normalisation=convention_of(saved), stop=stopping_rule(saved),
                     region=saved.get("region"), seed=saved.get("seed"), capped=bool(saved.get("iteration_cap_reached")),
                     faces=[(f["parameter"], f["side"], f["mass"]) for f in saved.get("box_faces_reached", [])])
        check = run / "analysis" / f"true_check_it_{int(match.group(1))}.npz"
        entry["unsampled"] = unsampled(file, check) if check.exists() else None
        rows.setdefault(int(match.group(1)), dict(best=[]))["best"].append(entry)
    for iteration, row in rows.items():
        row["best"].sort(key=lambda e: (e["tag"] != "", e["tag"], -e["n_live"]))
        boost = run / "boost" / f"boost_it_{iteration}.json"
        if boost.exists():
            modes = json.loads(boost.read_text())["modes"]
            values = np.array([m["log_evidence"] for m in modes if "log_evidence" in m], float)
            if len(values) == len(modes):
                row["boost"] = dict(value=float(logsumexp(values)),
                                    spread=float(max(m.get("log_evidence_spread", 0.0) for m in modes)), modes=len(modes))
        check = run / "analysis" / f"true_check_it_{iteration}.npz"
        if check.exists():
            found = correction(check, rng, refused_value=refused_value)
            checked_target, check_problem = recorded_target(check)
            row["true_check"] = found
            row["unrecorded_sampler"] = chain_sampling(check, run, iteration)[0]
            for entry in row["best"]:
                # Both record their target: it must be the same one. One of them was made before
                # targets were recorded: joined by the network's number, as then, and said so.
                # One records something that cannot be used: that is not the second case.
                recorded = checked_target is not None and entry["target"] is not None
                problem = (check_problem and f"the true check's {check_problem}") or (
                    entry["target_problem"] and f"the evidence's {entry['target_problem']}")
                if problem:
                    entry["correction_unavailable"] = dict(unavailable=f"{problem}, so that the two were made on the same target cannot be checked")
                elif recorded and checked_target != entry["target"]:
                    entry["correction_unavailable"] = dict(unavailable="true check and evidence sampled different targets (model/domain/region/box)")
                elif found["value"] is not None and entry["unsampled"] and max(
                        entry["unsampled"]["higher"], entry["unsampled"]["farther"] or 0.0) > UNSAMPLED:
                    entry["correction_unavailable"] = dict(unavailable=unreached(entry["unsampled"]))
                    entry["target_recorded"] = recorded
                else:
                    entry["correction" if found["value"] is not None else "correction_unavailable"] = found
                    entry["target_recorded"] = recorded
        data = run / "training_data" / f"data_it_{iteration}.csv"
        if data.exists():
            with open(data) as handle:
                row["training_points"] = sum(1 for _ in handle) - 1
    return dict(sorted(rows.items()))


def number(value, spec="+.2f"):
    return "-" if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = collect(args.run)
    if not rows:
        raise SystemExit(f"no evidence_it_*.json in {args.run / 'evidence'}")
    out = args.out or args.run / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    last = max(rows)
    reference = rows[last]["best"][0]
    lines = [f"# {args.run.name}: the evidence, network by network", "",
             "ln Z in the run's own prior convention. BEST: nested sampling of the network (its error is sqrt(H / n_live)). "
             "Boost: the bridge-sampling estimate of the boost drawn from that network. Corrected: BEST plus "
             "ln <exp(truth - network)> at posterior samples with true values, the evidence of the true likelihood; "
             "its error adds the bootstrap error of that mean.", "",
             f"| network | training points | ln Z (BEST) | error | against network {last} | H | dead points | min | integrated over | "
             "ln Z (boost) | scatter of truth - network | effective samples | correction | ln Z (corrected) | error |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = []
    for iteration, row in rows.items():
        for entry in row["best"]:
            corrected = entry.get("correction")
            difference = entry["lnZ"] - reference["lnZ"]
            together = float(np.hypot(entry["sigma"], reference["sigma"]))
            fixed = None if corrected is None else entry["lnZ"] + corrected["value"]
            fixed_error = None if corrected is None else float(np.hypot(entry["sigma"], corrected["error"]))
            boost = row.get("boost")
            label = f"{iteration}" + (f" ({entry['tag']})" if entry["tag"] else "")
            lines.append(
                f"| {label} | {row.get('training_points', '-')} | {entry['lnZ']:.2f} | {entry['sigma']:.2f} | "
                + ("-" if entry is reference or not same_convention(entry, reference) else f"{difference:+.2f} ({difference / together:+.1f} sigma)")
                + f" | {entry['H']:.1f} | {entry['dead']} | {entry['minutes']:.0f} | {'region' if entry['region'] else 'box'} | "
                + ("-" if boost is None else f"{boost['value']:.2f} ± {boost['spread']:.2f}")
                + " | " + ("-" if corrected is None else f"{corrected['scatter']:.2f}")
                + " | " + ("-" if corrected is None else f"{corrected['effective']:.0f} of {corrected['points']}")
                + " | " + ("-" if corrected is None else f"{corrected['value']:+.2f} ± {corrected['error']:.2f}")
                + " | " + number(fixed, ".2f") + " | " + number(fixed_error, ".2f") + " |")
            table.append(dict(entry, network=iteration, boost=boost, correction=corrected, lnZ_corrected=fixed,
                              lnZ_corrected_error=fixed_error, training_points=row.get("training_points"),
                              correction_unavailable=entry.get("correction_unavailable"),
                              unrecorded_sampler=row.get("unrecorded_sampler") if corrected is not None else None))
    notes = []
    for iteration, row in rows.items():
        check = row.get("true_check")
        if check and "unavailable" in check:
            notes.append(f"- network {iteration}: no correction to the true likelihood: {check['unavailable']}"
                         + (f" ({check['refused']} refused)" if check["refused"] else "") + ".")
        elif check and check["refused"]:
            notes.append(f"- network {iteration}: the likelihood refused {check['refused']} of the {check['points']} posterior samples "
                         f"of its true check. They weigh nothing and are counted in the mean over all {check['points']}, which puts "
                         f"the correction {-np.log1p(-check['refused'] / check['points']):.3f} below a mean over the others alone.")
        if row.get("unrecorded_sampler") and any(entry.get("correction") for entry in row["best"]):
            notes.append(f"- network {iteration}: its true check {row['unrecorded_sampler']}. The correction is over those samples "
                         "as they are, which may not be the network's posterior: make the true check again (it samples such a "
                         "chain anew).")
        for entry in row["best"]:
            unavailable = entry.get("correction_unavailable", {}).get("unavailable")
            if unavailable and unavailable != (check or {}).get("unavailable"):
                notes.append(f"- network {iteration} ({entry['tag'] or 'default'}): no correction: {unavailable}.")
            for parameter, side, mass in entry["faces"]:
                notes.append(f"- network {iteration}{' (' + entry['tag'] + ')' if entry['tag'] else ''}: {mass:.1e} of the posterior "
                             f"within 2% of the box's {side} face in {parameter}, inside its prior range: ln Z misses the mass beyond it.")
            if entry["capped"]:
                notes.append(f"- network {iteration}: BEST stopped at its iteration cap, not at its tolerance: ln Z is incomplete.")
    for iteration, row in rows.items():
        groups = []
        for entry in row["best"]:
            if entry["capped"]:                 # incomplete (noted above): no measure of the sampler's scatter
                continue
            group = next((g for g in groups if same_run_of_the_sampler(entry, g[0])), None)
            if group is None:
                groups.append([entry])
            else:
                group.append(entry)
        for entries in groups:
            seeds = [e["seed"] for e in entries]
            if len(entries) > 1 and None not in seeds and len(set(seeds)) == len(seeds):
                values = [e["lnZ"] for e in entries]
                notes.append(f"- network {iteration}: {len(entries)} BEST runs span {max(values) - min(values):.2f} in ln Z "
                             f"(errors {', '.join(format(e['sigma'], '.2f') for e in entries)}): same target, distinct seeds.")
        if len(groups) > 1:
            notes.append(f"- network {iteration}: its BEST runs differ in target, prior convention or stopping rule "
                         "(its tolerance; or at a fraction of the evidence since 11 October 2026, at one half before): their "
                         "differences are not the sampler's scatter alone.")
    name = lambda t: f"{t['network']}" + (f" ({t['tag']})" if t["tag"] else "")
    unrecorded = [name(t) for t in table if t["correction"] is not None and not t.get("target_recorded")]
    if unrecorded:
        notes.append(f"- networks {', '.join(unrecorded)}: the evidence, the true check or both were made before they recorded "
                     "their target (11 October 2026). The correction is joined by the network's number, as it was then: that "
                     "both used the same network file, box and region is not checked.")
    where = {}
    for t in table:
        where.setdefault(integrated_over(t) or "the whole box", []).append(name(t))
    if len(where) > 1:
        notes.append("- the networks were not integrated over the same part of the box ("
                     + "; ".join(f"{place}: {', '.join(names)}" for place, names in where.items())
                     + "). They are compared as estimates of one evidence, which they are as far as each region holds the posterior.")
    other_convention = [name(t) for t in table if not same_convention(t, reference)]
    if other_convention:
        notes.append(f"- networks {', '.join(other_convention)}: another prior convention than network {last}'s; not compared with it.")
    fixed = [(t["network"], t["lnZ_corrected"], t["lnZ_corrected_error"]) for t in table
             if t["lnZ_corrected"] is not None and not t["tag"] and same_convention(t, reference)]
    if len(fixed) > 1:
        values = np.array([v for _, v, _ in fixed])
        notes.append(f"- corrected ln Z of networks {', '.join(str(n) for n, _, _ in fixed)}: spread {values.max() - values.min():.2f} "
                     f"(largest error {max(e for _, _, e in fixed):.2f}).")
    lines += ["", "## Notes", ""] + (notes or ["- nothing flagged."])
    (out / "evidence.md").write_text("\n".join(lines) + "\n")
    (out / "evidence.json").write_text(json.dumps(dict(run=args.run.name, rows=table), indent=1, default=float) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    main_rows = [t for t in table if not t["tag"]]
    ax.errorbar([t["network"] for t in main_rows], [t["lnZ"] for t in main_rows], [t["sigma"] for t in main_rows],
                fmt="o-", color="C0", capsize=3, label="network (BEST)")
    other = [t for t in table if t["tag"]]
    if other:
        ax.errorbar([t["network"] + 0.15 for t in other], [t["lnZ"] for t in other], [t["sigma"] for t in other],
                    fmt="o", color="C0", mfc="white", capsize=3, label="network (BEST, tagged runs)")
    boosts = [t for t in main_rows if t["boost"]]
    if boosts:
        ax.errorbar([t["network"] - 0.15 for t in boosts], [t["boost"]["value"] for t in boosts], [t["boost"]["spread"] for t in boosts],
                    fmt="s", color="C2", capsize=3, label="network (its boost's bridge sampling)")
    fixed_rows = [t for t in main_rows if t["lnZ_corrected"] is not None]
    if fixed_rows:
        ax.errorbar([t["network"] for t in fixed_rows], [t["lnZ_corrected"] for t in fixed_rows], [t["lnZ_corrected_error"] for t in fixed_rows],
                    fmt="D--", color="C3", capsize=3, label="corrected to the true likelihood")
    ax.set(xlabel="network", ylabel="ln Z", title=args.run.name[:60])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "evidence.png", dpi=140)
    print("\n".join(lines[4:]))
    print(f"\nwritten: {out / 'evidence.md'}, evidence.json, evidence.png")


if __name__ == "__main__":
    main()
