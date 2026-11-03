"""python client.py augment RUN --profiles STUDY [--points FILE.csv] [-n 300] [-i N]

Add true evaluations made elsewhere to a run's training set, through CLiENT's acquisition
filter. The sources are profile studies (every true call in STUDY/calls/attempt_*.jsonl) and
CSV files with the run's parameters and `loglkl`.

Why. The acquisition aims for a training density c L^(1/T), with L estimated by the current
network. Where the network is too pessimistic, the region gets too few points and is never
corrected. On SINU (Planck + DESI) this hid the strong-coupling peak for six iterations; the
true 1D profile in logGeff found it, and its evaluations, added to the training set, taught
the network the peak (error there +83 -> +0.5 log-units with all 2845 of them, +8.6 with 300).
This needs a direction already known to be a problem, and a profile along it.

The filter. A training set distributed as c L^(1/T) has the same density wherever L is the
same. The density law rho(L) is read off the training set itself: the k-NN number density at
every training point (k = acquisition.n_neighbors, the selector's whitening), its median in
bins of the points' true log L, made monotone. A candidate with true value L is short of
points where the k-NN density of the training set around it is below rho(L), and is kept with
the selector's retention factor a = [1 - rho/rho(L)]_+ (dataset/acquisition.py; the paper's
eq. 13 with the true likelihood). Both densities come from the same estimator, so its bias in
high dimension cancels as it does there. The candidates are visited once, in random order, and
each is kept with probability a as it stands when its turn comes: every point kept raises the
density its neighbours see, so a tight cluster of profile probes contributes about k points and
no more, a real gap (a near 1) is filled, and the noise-level shortfalls that the k-NN estimate
shows at about half the candidates of a region already at its target (a of 0.1 - 0.2) are
mostly passed over. It stops at -n points.

Coordinates are compared at the training set's single precision, so points injected earlier
are recognised as already present. Candidates below the range the law covers (more than 0.5 T (n + 3 sqrt(2n)) log-units under
the best training value; if fewer than 120 training points lie in that range, the range of the
best 120, with a warning) are left out, as are points outside the run's domain and exact
copies of training points. A candidate above the best training value is judged against the
density of the best training points (no extrapolation).

Output. RUN/augmentation/augment_it_N.csv (the parameters, then loglkl) and augment_it_N.json
(sources, counts, the density law, where each point came from). No existing file is changed.
When RUN is continued from iteration N the loop adds these points on loading data_it_N.csv
(client.py); from then on they are part of data_it_<N+1>.csv. Derived parameters of the added
points are unknown (NaN), as for any point recorded without them. Delete the two files to undo.

The first training that includes the points runs a longer schedule
(training.augment_schedule_factor, default 3, multiplies lr_patience, lr_grace and patience) and
leaves augment_it_N.trained.json; later iterations use the ordinary schedule.
--output DIR writes them elsewhere instead, for a trial the run does not see.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dataset.acquisition import _log_unit_ball_volume
from run.storage import augmentation_training_iteration, write_json

_TINY = 1e-300


def training_log_density(dataset):
    """k-NN log number density at every training point (the point itself excluded)."""
    k, ndim = dataset.n_neighbors, dataset.knn_index.ndim
    distances, _ = dataset.knn_index.query(np.asarray(dataset.inputs, float), n_neighbors=k + 1)
    return np.log(k) - _log_unit_ball_volume(ndim) - ndim * np.log(np.maximum(distances[:, -1], _TINY))


class DensityLaw:
    """log rho as a function of true log L, from the training set: medians in equal-count
    bins of log L over the range the acquired cloud spans, made non-decreasing."""

    def __init__(self, logl, log_rho, temperature, ndim, bins=25, min_count=40):
        from sklearn.isotonic import IsotonicRegression

        logl, log_rho = np.asarray(logl, float), np.asarray(log_rho, float)
        self.best = float(logl.max())
        self.window = float(0.5 * temperature * (ndim + 3.0 * np.sqrt(2.0 * ndim)))
        finite = np.flatnonzero(np.isfinite(logl))
        inside = finite[logl[finite] >= self.best - self.window]
        # A young training set can have too few points that close to its best one (iteration 5
        # of the SINU run: 347 of 12596; iteration 3: 6). The law is then read from its best
        # 3 * min_count points, whatever range they span.
        self.extended = len(inside) < 3 * min_count
        if self.extended:
            if len(finite) < 3 * min_count:
                raise ValueError(f"only {len(finite)} training points; too few to read a density law from")
            inside = finite[np.argsort(logl[finite])[-3 * min_count:]]
        n_bins = min(bins, len(inside) // min_count)
        groups = np.array_split(inside[np.argsort(logl[inside])], n_bins)
        self.centres = np.array([np.median(logl[g]) for g in groups])
        self.counts = np.array([len(g) for g in groups])
        raw = np.array([np.median(log_rho[g]) for g in groups])
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            self.centres, raw, sample_weight=self.counts)
        self.values = self._iso.predict(self.centres)
        self.floor = float(self.centres[0])

    def __call__(self, logl):
        return self._iso.predict(np.asarray(logl, float))

    def summary(self):
        return dict(best=self.best, window=self.window, floor=self.floor, extended=bool(self.extended),
                    logl=self.centres.tolist(), log_density=self.values.tolist(),
                    counts=self.counts.tolist())


def select_known_points(dataset, x, y, n_max, rng=None):
    """Indices (in order of selection) of the candidates (x, true log L y) that the training
    set is short of, by the density-deficit rule described in the module docstring, and a
    dictionary of diagnostics."""
    rng = rng or np.random.default_rng()
    index, k, ndim = dataset.knn_index, dataset.n_neighbors, dataset.knn_index.ndim
    # The training set holds single-precision coordinates, and these points would be stored
    # the same way: compare, and measure densities, at that precision.
    x = np.asarray(x, np.float32).reshape(-1, ndim)
    y = np.asarray(y, float).ravel()
    stored = {row.tobytes() for row in np.ascontiguousarray(dataset.inputs, np.float32)}
    copies = np.array([row.tobytes() in stored for row in np.ascontiguousarray(x)], bool).reshape(-1)
    _, twin = np.unique(x, axis=0, return_inverse=True)
    twin = np.asarray(twin).ravel()
    x = x.astype(float)
    law = DensityLaw(dataset.targets[:, 0], training_log_density(dataset),
                     dataset.target_temperature, ndim)
    distances, _ = index.query(x)
    nearest_sq = distances ** 2
    whitened = index.whiten(x)
    finite = np.isfinite(y)
    below = finite & (y < law.floor)
    eligible = finite & ~copies & ~below
    log_target = np.full(len(x), -np.inf)
    log_target[finite] = law(y[finite])
    log_constant = np.log(k) - _log_unit_ball_volume(ndim)

    def retention(rows):
        # The law is the density of the training set as it is; the few points added here
        # change its size by a per cent or so, which is not rescaled.
        log_rho = log_constant - 0.5 * ndim * np.log(np.maximum(nearest_sq[rows].max(axis=-1), _TINY))
        log_deficit = log_target[rows] - log_rho
        return np.where(eligible[rows] & (log_deficit > 0.0), -np.expm1(-np.maximum(log_deficit, 0.0)), 0.0), log_deficit

    everyone = np.arange(len(x))
    first_retention, first_deficit = retention(everyone)
    selected = []
    for i in rng.permutation(len(x)):
        if len(selected) >= n_max:
            break
        a, _ = retention(i)
        if not rng.random() < a:
            continue
        selected.append(int(i))
        eligible[i] = False
        new_sq = np.sum((whitened - whitened[i]) ** 2, axis=1)
        eligible[twin == twin[i]] = False  # candidates identical to the point just added
        worst = nearest_sq.argmax(axis=1)
        closer = new_sq < nearest_sq[everyone, worst]
        nearest_sq[closer, worst[closer]] = new_sq[closer]
    last_retention, _ = retention(everyone)
    info = dict(candidates=len(x), not_finite=int(np.count_nonzero(~finite)),
                copies_of_training_points=int(np.count_nonzero(finite & copies)),
                below_law=int(np.count_nonzero(below)),
                short_at_start=int(np.count_nonzero(first_retention)),
                expected_at_start=float(first_retention.sum()),
                short_at_end=int(np.count_nonzero(last_retention)),
                expected_at_end=float(last_retention.sum()),
                selected=len(selected), density_law=law.summary())
    return np.array(selected, int), info, first_deficit


def load_profile_calls(study, names):
    """Every finished true evaluation of a profile study, in the run's parameter order:
    (x, y, labels). The study must have used the run's training objective."""
    study = Path(study)
    manifest = json.loads((study / "manifest.json").read_text())
    if "native" not in str(manifest.get("objective", "")):
        raise ValueError(f"{study}: objective {manifest.get('objective')!r} is not the training target")
    order = list(manifest["bounds"])
    if set(order) != set(names):
        raise ValueError(f"{study}: its parameters differ from the run's")
    columns = [order.index(n) for n in names]
    xs, ys, labels = [], [], []
    journals = sorted((study / "calls").glob("attempt_*.jsonl"))
    if not journals:
        raise FileNotFoundError(f"no calls/attempt_*.jsonl in {study}")
    for journal in journals:
        started = {}
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "start":
                started[event["call"]] = (event["x"], event.get("stage", ""))
            elif event["event"] == "return" and event.get("value") is not None and event["call"] in started:
                point, stage = started[event["call"]]
                xs.append([point[c] for c in columns])
                ys.append(float(event["value"]))
                labels.append(f"{study.name}/{journal.stem}#{event['call']} {stage}")
    return np.array(xs, float).reshape(-1, len(names)), np.array(ys, float), labels


def load_points_csv(path, names):
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [c for c in list(names) + ["loglkl"] if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return (frame[list(names)].to_numpy(float), frame["loglkl"].to_numpy(float),
            [f"{Path(path).name}#{i}" for i in range(len(frame))])


def read_augmentation(path, names):
    """(inputs, targets) of an augment_it_N.csv, checked against the run's parameters."""
    import pandas as pd

    frame = pd.read_csv(path)
    if list(frame.columns) != list(names) + ["loglkl"]:
        raise ValueError(f"{path}: columns are not the run's parameters followed by loglkl")
    return frame[list(names)].to_numpy(float), frame["loglkl"].to_numpy(float)


def _marker(path):
    return path.with_name(path.stem + ".trained.json")


def pending_augmentations(run_dir, inputs):
    """Paths of the augment_it_N.csv files whose points are in the training set `inputs`
    and have not yet been trained on (no augment_it_N.trained.json beside them)."""
    import re

    inputs = np.asarray(inputs, np.float32)
    pending = []
    for path in sorted(Path(run_dir).glob("augmentation/augment_it_*.csv")):
        if (not re.fullmatch(r"augment_it_\d+", path.stem)
                or augmentation_training_iteration(_marker(path)) is not None):
            continue
        import pandas as pd

        rows = pd.read_csv(path).iloc[:, :inputs.shape[1]].to_numpy(np.float32)
        if len(rows) and all(np.any(np.all(np.isclose(inputs, row, rtol=1e-6, atol=0.0), axis=1))
                             for row in (rows[0], rows[-1])):
            pending.append(path)
    return pending


def mark_trained(paths, iteration, epochs, schedule_factor):
    """Record that the points of these augmentation files have had their first training."""
    for path in paths:
        write_json(_marker(path), dict(
            iteration=int(iteration), epochs=int(epochs), schedule_factor=float(schedule_factor),
            time=time.strftime("%Y-%m-%d %H:%M:%S")))


def _node_table(study, names, x, chosen, bounds):
    """Candidates and selected points per node of each profiled parameter of a 1D study, and
    a warning where the profile stops short of the run's domain."""
    path = Path(study) / "true_profiles.json"
    if not path.exists():
        return []
    profiles = json.loads(path.read_text()).get("profiles", {})
    lines = []
    for name, record in profiles.items():
        if name not in names or "grid" not in record:
            continue
        grid = np.asarray(record["grid"], float)
        lo, hi = bounds[names.index(name)]
        step = float(np.median(np.diff(np.sort(grid)))) if len(grid) > 1 else 0.0
        gaps = [f"[{a:g}, {b:g}]" for a, b in ((lo, grid.min()), (grid.max(), hi)) if b - a > 1.5 * step]
        if gaps:
            lines.append(f"   WARNING: the {name} profile covers [{grid.min():g}, {grid.max():g}] of the domain "
                         f"[{lo:g}, {hi:g}]. Nothing added here constrains the network in {' and '.join(gaps)}: "
                         "it may extrapolate the profile's trend there, and the next batch then goes "
                         "there first (docs/augment_training.md).")
        node = np.abs(x[:, names.index(name)][:, None] - grid[None, :]).argmin(axis=1)
        lines.append(f"   {name:>12} " + " ".join(f"{g:6.2f}" for g in grid))
        lines.append(f"   {'candidates':>12} " + " ".join(f"{np.count_nonzero(node == j):6d}" for j in range(len(grid))))
        lines.append(f"   {'selected':>12} " + " ".join(f"{np.count_nonzero(node[chosen] == j):6d}" for j in range(len(grid))))
    return lines


def main(argv=None):
    import pandas as pd

    from config.config import Config
    from dataset.dataset import TrainingDataset
    from likelihood.surrogate import SurrogateMetadata
    from run.run import _start_iteration, settings_file

    p = argparse.ArgumentParser(prog="client.py augment", description=__doc__.split("\n\n")[1])
    p.add_argument("run", type=Path, help="run directory")
    p.add_argument("--profiles", action="append", type=Path, default=[], metavar="STUDY",
                   help="profile study directory; repeat for several")
    p.add_argument("--points", action="append", type=Path, default=[], metavar="CSV",
                   help="CSV with the run's parameters and loglkl; repeat for several")
    p.add_argument("-n", "--n-max", type=int, default=300, help="at most this many points (300)")
    p.add_argument("-i", "--iteration", type=int,
                   help="training set data_it_N.csv to augment (default: the one a continued run loads)")
    p.add_argument("--seed", type=int, help="default: the run's seed plus the iteration")
    p.add_argument("--output", type=Path, help="write here instead of RUN/augmentation (a trial)")
    p.add_argument("--force", action="store_true", help="replace an existing augment_it_N.csv")
    a = p.parse_args(argv)
    if not a.profiles and not a.points:
        p.error("give --profiles and/or --points")
    if a.n_max < 1:
        p.error("--n-max must be positive")

    run = a.run.resolve()
    cfg = Config.from_yaml(settings_file(run))
    md = SurrogateMetadata.load(run / "metadata.json")
    names = md.param_names
    bounds = np.array([md.bounds[n] for n in names], float)
    iteration = a.iteration if a.iteration is not None else _start_iteration(run)
    data_path = run / "training_data" / f"data_it_{iteration}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")
    out_dir = (a.output or run / "augmentation").resolve()
    out_csv = out_dir / f"augment_it_{iteration}.csv"
    if out_csv.exists() and not a.force:
        raise FileExistsError(f"{out_csv} exists; --force replaces it")

    data = pd.read_csv(data_path)
    if list(data.columns) != list(names) + ["loglkl"]:
        raise ValueError(f"{data_path}: columns are not the run's parameters followed by loglkl")
    dataset = TrainingDataset(data[names].to_numpy(float), data["loglkl"].to_numpy(float),
                              SimpleNamespace(param_names=names, derived_names=[]),
                              n_neighbors=cfg.acquisition.n_neighbors,
                              target_temperature=cfg.acquisition.target_temperature)

    xs, ys, labels, sources = [], [], [], []
    for study in a.profiles:
        x, y, lab = load_profile_calls(study, names)
        other = json.loads((Path(study) / "manifest.json").read_text()).get("run")
        note = "" if other is None or Path(other).resolve() == run else f" (made for {other})"
        print(f"{Path(study).name}: {len(y)} true evaluations{note}")
        xs.append(x), ys.append(y), labels.extend(lab)
        sources.append(dict(kind="profiles", path=str(Path(study).resolve()), evaluations=len(y)))
    for path in a.points:
        x, y, lab = load_points_csv(path, names)
        print(f"{Path(path).name}: {len(y)} points")
        xs.append(x), ys.append(y), labels.extend(lab)
        sources.append(dict(kind="points", path=str(Path(path).resolve()), evaluations=len(y)))
    x, y = np.vstack(xs), np.concatenate(ys)
    inside = np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1]), axis=1)
    x, y, labels = x[inside], y[inside], [l for l, keep in zip(labels, inside) if keep]

    seed = a.seed if a.seed is not None else (0 if cfg.seed is None else cfg.seed) + iteration
    t0 = time.monotonic()
    chosen, info, deficit = select_known_points(dataset, x, y, a.n_max, np.random.default_rng(seed))
    info.update(outside_domain=int(np.count_nonzero(~inside)), seconds=time.monotonic() - t0)

    law = info["density_law"]
    print(f"Training set data_it_{iteration}.csv: {len(dataset.inputs)} points, best log L {law['best']:.2f}; "
          f"T = {dataset.target_temperature:g}, k = {dataset.n_neighbors}")
    if law["extended"]:
        print(f"   WARNING: fewer than 120 training points lie within {law['window']:.0f} log-units of the best; the density "
              f"law is read from the best 120 instead, which span {law['best'] - law['floor']:.0f} log-units. A training "
              "set this young need not follow L^(1/T) yet; the filter then finds little to add.")
    print(f"Candidates: {info['candidates']} in the domain ({info['outside_domain']} outside), "
          f"{info['not_finite']} not finite, {info['copies_of_training_points']} already training points, "
          f"{info['below_law']} more than {law['best'] - law['floor']:.0f} log-units below the best")
    print(f"Short of training density: {info['short_at_start']} candidates (summed retention "
          f"{info['expected_at_start']:.0f}); selected {info['selected']} (limit {a.n_max}); "
          f"summed retention left: {info['expected_at_end']:.0f}")
    if len(chosen):
        gain = y[chosen] - law["best"]
        print(f"Selected points: log L from {y[chosen].min():.2f} to {y[chosen].max():.2f}; "
              f"{int(np.count_nonzero(gain > 0))} above the best training value (by up to {max(gain.max(), 0):.2f})")
    for study in a.profiles:
        for line in _node_table(study, names, x, chosen, bounds):
            print(line)
    if not len(chosen):
        print("Nothing to add: the training set is not short of density at the candidates.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(x[chosen].astype(np.float32), columns=names)
    frame["loglkl"] = y[chosen].astype(np.float32)
    frame.to_csv(out_csv, index=False)
    info.update(run=str(run), iteration=iteration, training_points=len(dataset.inputs), seed=seed,
                n_max=a.n_max, sources=sources, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                points=[dict(source=labels[i], loglkl=float(y[i]), log_deficit=float(deficit[i])) for i in chosen])
    (out_dir / f"augment_it_{iteration}.json").write_text(json.dumps(info, indent=1))
    where = (f"a continued run adds them when it loads data_it_{iteration}.csv" if a.output is None
             else "a trial: the run does not see them")
    print(f"Wrote {len(chosen)} points to {out_csv} ({where})")


if __name__ == "__main__":
    main()
