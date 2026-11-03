"""Derived-coordinate views of a CLiENT run, for profiles in derived parameters.

A profile in a derived parameter d(x) maximises over the level set d(x) = d0: a nonlinear
constraint, which the profile machinery -- built on holding coordinates fixed -- has no
notion of. So change variables instead. Pick a sampled "pivot" parameter p on which d
depends monotonically (θ_s on H0, σ8 or S8 on logA, Ω_m on ω_cdm) and use the coordinates
z = x with x_p replaced by d. For given z the pivot is recovered by Newton's method on the
derived-parameter emulator, so a profile in d is an ordinary coordinate profile in z.

A *view* is a directory laid out like a CLiENT run in the z coordinates: metadata,
training data, a network z -> log L (pivot solve, then the original network) and a
backend that maps z -> x and calls the original likelihood. `client.py profiles` runs on
it unchanged -- 1D and 2D, hybrid scan, residual Newton, resume and call accounting --
and nothing of the original run is modified. A network trained on boosts brings the
region of its boosts along, and profiles of the view search it there, as they search the
run's own network (ViewRegion, restrict_view).

The constraint is imposed through the emulator, so a node's exact derived value differs
from its grid value by the emulator's error at that point; `client.py derived-view
--nodes` evaluates the exact value at every final node.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from model.activations import Alsing

# Newton steps in the pivot solve. From a secant start (one pivot) it converges to machine
# precision in well under this on the emulator's smooth, monotone maps; the fixed count
# keeps the map a smooth, twice-differentiable function of z.
NEWTON_STEPS = 12

# Largest change of a training label, in log L, that float32 rounding of a row's view
# coordinates may cause for the row to be written to the view (half the profile audit's
# Delta chi^2 = 0.01 tolerance, in log L).
ROUNDTRIP_TOLERANCE = 2e-3

# Rows further than this below the best training point are not written to a view: see
# make_view.
DEFICIT_LIMIT = 500.0

# Penalty on the relative residual (d_emu(x) - d0) / spread when the pivot would have to
# leave its box to reach d0. Converged points have residuals ~1e-12 and are unaffected;
# a point one percent of the training spread short pays 100 log-units.
PENALTY = 1e6


def _activation_of(layer):
    if isinstance(layer, Alsing):
        return "alsing"
    name = layer.get_config().get("activation")
    if name in ("tanh", "linear"):
        return name
    raise ValueError(f"unsupported emulator activation {name!r}")


def emulator_parameters(emulator, outputs):
    """The weights of a derived-parameter emulator (derived/emulator.py), restricted to
    the output columns `outputs`, as plain arrays."""
    from derived.emulator import DerivedDenormalization, SelectInputs

    layers = emulator.layers
    select = next(l for l in layers if isinstance(l, SelectInputs))
    norm = next(l for l in layers if isinstance(l, tf.keras.layers.Normalization))
    denorm = next(l for l in layers if isinstance(l, DerivedDenormalization))
    dense = [l for l in layers if isinstance(l, tf.keras.layers.Dense)]
    acts = [l for l in layers[layers.index(dense[0]) + 1:]
            if isinstance(l, (Alsing, tf.keras.layers.Activation))]
    if len(acts) != len(dense) - 1:
        raise ValueError("emulator is not a Dense/activation chain")

    hidden = []
    for d, a in zip(dense[:-1], acts):
        kernel, bias = d.get_weights()
        kind = _activation_of(a)
        if kind == "alsing":
            beta, gamma = a.get_weights()
        else:
            beta = gamma = np.zeros_like(bias)
        hidden.append(dict(kernel=kernel, bias=bias, beta=beta, gamma=gamma, kind=kind))
    out_kernel, out_bias = dense[-1].get_weights()
    mean = np.asarray(denorm.derived_mean.numpy())
    std = np.asarray(denorm.derived_std.numpy())
    return dict(
        select=list(select.indices),
        norm_mean=np.asarray(norm.mean, float).ravel(),
        norm_variance=np.asarray(norm.variance, float).ravel(),
        hidden=hidden,
        out_kernel=out_kernel[:, outputs],
        out_bias=out_bias[outputs],
        out_mean=mean[outputs],
        out_std=std[outputs],
    )


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class PivotSolve(tf.keras.layers.Layer):
    """z -> [x | r]: the pivots recovered from their derived targets.

    Column pivot_columns[j] of z holds the target value of emulator output j; the same
    column of x holds the pivot parameter that reaches it, found by Newton's method on
    the emulator, clipped to the pivot's box. r is the remaining residual in units of
    the output's training spread: ~0 wherever the target is reachable.

    The emulator's forward pass and its derivative along the pivots are written out
    here rather than calling the emulator model, so that the layer's weights are its own:
    copy_model_dtype can then clone it to float64 like any other layer, and the whole map
    is twice differentiable, as the profile optimisers need.
    """

    def __init__(self, n_inputs, pivot_columns, select, hidden_sizes, activations,
                 pivot_lower, pivot_upper, steps=NEWTON_STEPS, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = int(n_inputs)
        self.pivot_columns = [int(c) for c in pivot_columns]
        self.select = [int(i) for i in select]
        self.hidden_sizes = [int(n) for n in hidden_sizes]
        self.activations = list(activations)
        self.pivot_lower = [float(v) for v in pivot_lower]
        self.pivot_upper = [float(v) for v in pivot_upper]
        self.steps = int(steps)
        missing = [c for c in self.pivot_columns if c not in self.select]
        if missing:
            raise ValueError(f"pivot columns {missing} are not inputs of the emulator")

    def get_config(self):
        config = super().get_config()
        config.update(n_inputs=self.n_inputs, pivot_columns=self.pivot_columns,
                      select=self.select, hidden_sizes=self.hidden_sizes,
                      activations=self.activations, pivot_lower=self.pivot_lower,
                      pivot_upper=self.pivot_upper, steps=self.steps)
        return config

    def build(self, input_shape):
        k, m = len(self.pivot_columns), len(self.select)
        fixed = dict(trainable=False, initializer="zeros")
        self.norm_mean = self.add_weight(name="norm_mean", shape=(m,), **fixed)
        self.norm_variance = self.add_weight(name="norm_variance", shape=(m,), **fixed)
        self.kernels, self.biases, self.betas, self.gammas = [], [], [], []
        width = m
        for i, n in enumerate(self.hidden_sizes):
            self.kernels.append(self.add_weight(name=f"kernel_{i}", shape=(width, n), **fixed))
            self.biases.append(self.add_weight(name=f"bias_{i}", shape=(n,), **fixed))
            self.betas.append(self.add_weight(name=f"beta_{i}", shape=(n,), **fixed))
            self.gammas.append(self.add_weight(name=f"gamma_{i}", shape=(n,), **fixed))
            width = n
        self.out_kernel = self.add_weight(name="out_kernel", shape=(width, k), **fixed)
        self.out_bias = self.add_weight(name="out_bias", shape=(k,), **fixed)
        self.out_mean = self.add_weight(name="out_mean", shape=(k,), **fixed)
        self.out_std = self.add_weight(name="out_std", shape=(k,), **fixed)
        super().build(input_shape)

    def set_emulator(self, parameters):
        """Load the arrays emulator_parameters() returns."""
        self.norm_mean.assign(parameters["norm_mean"])
        self.norm_variance.assign(parameters["norm_variance"])
        for i, layer in enumerate(parameters["hidden"]):
            self.kernels[i].assign(layer["kernel"])
            self.biases[i].assign(layer["bias"])
            self.betas[i].assign(layer["beta"])
            self.gammas[i].assign(layer["gamma"])
        self.out_kernel.assign(parameters["out_kernel"])
        self.out_bias.assign(parameters["out_bias"])
        self.out_mean.assign(parameters["out_mean"])
        self.out_std.assign(parameters["out_std"])

    def emulate(self, x):
        """d(x) and its derivatives along the pivots: shapes (B, k) and (B, k, k)."""
        dtype = x.dtype
        k = len(self.pivot_columns)
        u = tf.gather(x, self.select, axis=1)
        # tf.keras.layers.Normalization divides by max(sqrt(variance), epsilon).
        denom = tf.maximum(tf.sqrt(self.norm_variance), tf.cast(1e-7, dtype))
        a = (u - self.norm_mean) / denom
        positions = [self.select.index(c) for c in self.pivot_columns]
        tangent = tf.one_hot(positions, len(self.select), dtype=dtype) / denom  # (k, m)
        da = tf.tile(tangent[None], [tf.shape(x)[0], 1, 1])                   # (B, k, m)
        for kernel, bias, beta, gamma, kind in zip(
            self.kernels, self.biases, self.betas, self.gammas, self.activations
        ):
            h = a @ kernel + bias
            dh = da @ kernel
            if kind == "alsing":
                s = tf.sigmoid(beta * h)
                a = (gamma + s * (1 - gamma)) * h
                slope = gamma + s * (1 - gamma) + h * (1 - gamma) * s * (1 - s) * beta
            elif kind == "tanh":
                a = tf.tanh(h)
                slope = 1 - a * a
            else:
                a, slope = h, tf.ones_like(h)
            da = dh * slope[:, None, :]
        y = a @ self.out_kernel + self.out_bias
        dy = da @ self.out_kernel                                                  # (B, k_tangent, k_out)
        return self.out_mean + self.out_std * y, tf.transpose(dy, [0, 2, 1]) * self.out_std[None, :, None]

    def call(self, z):
        dtype = z.dtype
        k = len(self.pivot_columns)
        lo = tf.constant(self.pivot_lower, dtype)
        hi = tf.constant(self.pivot_upper, dtype)
        target = tf.gather(z, self.pivot_columns, axis=1)
        E = tf.one_hot(self.pivot_columns, self.n_inputs, dtype=dtype)  # (k, N)

        def place(pivots):
            return z + (pivots - target) @ E

        if k == 1:
            # Secant start between the ends of the pivot's range: exact for a linear map.
            d_lo, _ = self.emulate(place(tf.broadcast_to(lo, tf.shape(target))))
            d_hi, _ = self.emulate(place(tf.broadcast_to(hi, tf.shape(target))))
            slope = d_hi - d_lo
            safe = tf.where(tf.abs(slope) > 0, slope, tf.ones_like(slope))
            pivots = tf.clip_by_value(lo + (target - d_lo) / safe * (hi - lo), lo, hi)
        else:
            pivots = tf.broadcast_to((lo + hi) / 2, tf.shape(target))
        for _ in range(self.steps):
            d, J = self.emulate(place(pivots))
            step = tf.linalg.solve(J, (target - d)[..., None])[..., 0]
            step = tf.where(tf.math.is_finite(step), step, tf.zeros_like(step))
            pivots = tf.clip_by_value(pivots + step, lo, hi)
        x = place(pivots)
        d, _ = self.emulate(x)
        return tf.concat([x, (d - target) / self.out_std], axis=1)


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class SliceColumns(tf.keras.layers.Layer):
    def __init__(self, start, stop, **kwargs):
        super().__init__(**kwargs)
        self.start, self.stop = int(start), int(stop)

    def get_config(self):
        config = super().get_config()
        config.update(start=self.start, stop=self.stop)
        return config

    def call(self, x):
        return x[:, self.start:self.stop]


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class FeasibilityPenalty(tf.keras.layers.Layer):
    """log L - penalty * |r|^2: zero wherever the pivot solve reached its targets."""

    def __init__(self, penalty=PENALTY, **kwargs):
        super().__init__(**kwargs)
        self.penalty = float(penalty)

    def get_config(self):
        config = super().get_config()
        config.update(penalty=self.penalty)
        return config

    def call(self, inputs):
        logl, residual = inputs
        return logl - self.penalty * tf.reduce_sum(residual * residual, axis=1, keepdims=True)


def pivot_map_model(n_inputs, pivot_columns, pivot_bounds, parameters, steps=NEWTON_STEPS):
    """The model z -> [x | r] on its own; the backend uses it to map points to x."""
    z = tf.keras.Input(shape=(n_inputs,))
    layer = PivotSolve(
        n_inputs, pivot_columns, parameters["select"],
        [len(h["bias"]) for h in parameters["hidden"]],
        [h["kind"] for h in parameters["hidden"]],
        [lo for lo, _ in pivot_bounds], [hi for _, hi in pivot_bounds], steps=steps,
        name="pivot_solve",
    )
    out = layer(z)
    layer.set_emulator(parameters)
    return tf.keras.Model(z, out)


def view_model(network, pivot_map, penalty=PENALTY):
    """z -> log L: the pivot solve, then a copy of the original network, then the
    feasibility penalty. One flat graph, so copy_model_dtype clones it like any other."""
    n, k = pivot_map.input_shape[1], pivot_map.output_shape[1] - pivot_map.input_shape[1]
    z = tf.keras.Input(shape=(n,))
    xr = pivot_map.layers[-1](z)
    x = SliceColumns(0, n, name="physical")(xr)
    r = SliceColumns(n, n + k, name="residual")(xr)
    # Re-apply the network's own layers to x, so the view is one flat graph that shares
    # the network's weights. (clone_model(network, input_tensors=x) leaves a dangling
    # InputLayer behind, which copy_model_dtype then cannot clone.)
    logl = network._run_through_graph(x, operation_fn=lambda layer: layer)
    logl = logl[0] if isinstance(logl, (list, tuple)) else logl
    logl = FeasibilityPenalty(penalty, name="feasibility")([logl, r])
    return tf.keras.Model(z, logl)


def to_physical(pivot_map, z):
    """Map points z (n, N) to x and the residuals r, in float64."""
    from model.network import copy_model_dtype

    model = pivot_map if pivot_map.compute_dtype == "float64" else copy_model_dtype(pivot_map, "float64")
    out = model(tf.constant(np.atleast_2d(np.asarray(z, float))), training=False).numpy()
    n = model.input_shape[1]
    return out[:, :n], out[:, n:]


class ViewRegion:
    """The region of the source run's boosts, seen from a view (profiles/region.py).

    The region (sampling/region.py) is a set of ellipsoids in the run's coordinates x, and a
    point z of the view is inside it when x(z), the point the pivot solve maps z to, is: rho(z)
    is the region's rho at x(z). The edge, the reach and the boost points are the region's own.
    restrict_view gives the searches the network inside it.
    """

    def __init__(self, region, pivot_map):
        from model.network import copy_model_dtype

        self.region = region
        self.pivot_map = pivot_map if pivot_map.compute_dtype == "float64" else copy_model_dtype(pivot_map, "float64")

    def __len__(self):
        return len(self.region)

    @property
    def inflate(self):
        return self.region.inflate

    @property
    def reach(self):
        return self.region.reach

    @property
    def rows(self):
        return self.region.rows

    def rho(self, z):
        x, _ = to_physical(self.pivot_map, z)
        return self.region.rho(x)


def restrict_view(model, region, dtype="float64"):
    """A view's network at `dtype`, seen inside `region`, the region of its source run.

    model.network.restrict_model for a view: the region is in the run's coordinates, so its
    wall (RegionWall) takes x(z), the output of the pivot solve, and it stands between the
    original network and the feasibility penalty, which keeps z to reachable targets inside
    the region and outside alike. It can be evaluated, differentiated twice and copied with
    copy_model_dtype; it is not for training or saving.
    """
    from model.network import RegionWall, copy_model_dtype
    from sampling.region import STRENGTH

    inner = copy_model_dtype(model, dtype)
    try:
        feasibility, x = inner.get_layer("feasibility"), inner.get_layer("physical").output
    except ValueError:
        raise ValueError("restrict_view needs the network of a derived-coordinate view (view_model)") from None
    logl, residual = feasibility.input
    wall = RegionWall(region.means, region.whiten, region.reach, region.lower, region.width,
                      region.inflate, region.floor, STRENGTH, dtype=dtype, name="region_wall")
    value = FeasibilityPenalty(feasibility.penalty, dtype=dtype, name="feasibility_in_region")(
        [wall([logl, x]), residual])
    out = tf.keras.Model(inner.input, value)
    out.dtype_policy = tf.keras.mixed_precision.Policy(dtype)
    return out


def choose_pivots(emulator, param_names, bounds, derived_names, targets, anchor, width,
                  exclude=(), grid=65):
    """One pivot per derived target, by sensitivity at the anchor, monotone over the box.

    The score of a candidate is |d d/d x_i| times the parameter's width near the peak,
    i.e. how far d moves over a typical posterior excursion of x_i. A candidate must be
    an emulator input, not excluded and not already a pivot, and d must be strictly
    monotone in it along its whole range with the other parameters at the anchor, so
    that the pivot solve has exactly one solution there. Returns ([pivot names], report).
    """
    from derived.emulator import SelectInputs

    select = next(l for l in emulator.layers if isinstance(l, SelectInputs)).indices
    x0 = tf.constant(anchor[None, :], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x0)
        d0 = emulator(x0)
    jac = tape.batch_jacobian(d0, x0).numpy()[0]  # (k_all, N)
    chosen, report = [], {}
    for target in targets:
        j = derived_names.index(target)
        scores = {}
        for i in select:
            name = param_names[i]
            if name in exclude or name in chosen:
                continue
            xs = np.repeat(anchor[None, :], grid, axis=0)
            xs[:, i] = np.linspace(bounds[i][0], bounds[i][1], grid)
            values = emulator(tf.constant(xs, tf.float32)).numpy()[:, j]
            steps = np.diff(values)
            monotone = bool(np.all(steps > 0) or np.all(steps < 0))
            scores[name] = dict(score=float(abs(jac[j, i]) * width[i]), monotone=monotone)
        ok = {n: s for n, s in scores.items() if s["monotone"]}
        if not ok:
            raise ValueError(f"no parameter moves {target} monotonically over its range; "
                             "give a pivot explicitly")
        pivot = max(ok, key=lambda n: ok[n]["score"])
        chosen.append(pivot)
        report[target] = dict(pivot=pivot, candidates=scores)
    return chosen, report


def make_view(run, output, targets, pivots=None, exclude=(), iteration=None):
    """Write the derived-coordinate view of `run` for the derived parameters `targets`.

    Returns the view's summary (also written to view.json).
    """
    from derived.emulator import load_derived_model, model_path
    from derived.metadata import DerivedMetadata
    from likelihood.base import ParameterInfo
    from likelihood.surrogate import SurrogateMetadata

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} exists and is not empty")
    derived = DerivedMetadata.load(run / "derived_data")
    if iteration is None:
        iteration = max(
            int(p.stem.rsplit("_", 1)[1]) for p in (run / "derived_data").glob("derived_model_it_*.keras")
            if (run / "trained_models" / f"model_it_{p.stem.rsplit('_', 1)[1]}.keras").exists()
        )
    unknown = sorted(set(targets) - set(derived.names))
    if unknown:
        raise ValueError(f"not derived parameters of this run: {unknown}")

    # The network, the run's metadata and the region of its boosts, if it has one, from one
    # verified private copy, with the identity of those files for view.json. The network is
    # the bare one, as the run saved it; the region is copied into the view (below). The
    # derived emulator and the training set are read from the run and their hashes recorded:
    # one replaced while it is read stops the view.
    from run.storage import digest, write_json
    from sampling.targets import load_snapshot

    surrogate, source_target = load_snapshot(run, iteration, "run")
    metadata, network = surrogate.metadata, surrogate.model
    names = metadata.param_names
    bounds = [metadata.bounds[n] for n in names]
    read_here = [model_path(run / "derived_data", iteration), run / "training_data" / f"data_it_{iteration}.csv"]
    source_files = {str(path.relative_to(run)): digest(path) for path in read_here}
    emulator = load_derived_model(read_here[0])
    data = pd.read_csv(read_here[1])
    if {str(path.relative_to(run)): digest(path) for path in read_here} != source_files:
        raise RuntimeError(f"the derived emulator or the training set of iteration {iteration} changed while the "
                           "view was being made; make it again when the run is not training")
    x = data[names].to_numpy(np.float64)
    loglkl = data["loglkl"].to_numpy(np.float64)

    # The anchor is the best training point; the widths come from the best 5%.
    anchor = x[np.argmax(loglkl)]
    top = x[loglkl >= np.quantile(loglkl, 0.95)]
    width = top.std(axis=0)
    if pivots is None:
        pivots, pivot_report = choose_pivots(emulator, names, bounds, derived.names, targets,
                                             anchor, width, exclude=exclude)
    else:
        pivot_report = {t: dict(pivot=p, candidates=None) for t, p in zip(targets, pivots)}
    if len(set(pivots)) != len(pivots) or set(pivots) & set(exclude):
        raise ValueError("pivots must be distinct and not excluded")
    columns = [names.index(p) for p in pivots]
    outputs = [derived.names.index(t) for t in targets]

    parameters = emulator_parameters(emulator, outputs)
    pmap = pivot_map_model(len(names), columns, [bounds[c] for c in columns], parameters)
    model = view_model(network, pmap)

    # z of every training row: the emulated derived value in the pivot's column, so the
    # pivot solve maps it back onto the row itself. Rounded to float32 first, as the CSV
    # stores it and the profile workflow reads it, so the round trip below is the one the
    # workflow's label audit will see.
    d_rows = emulator.predict(x.astype(np.float32), batch_size=65536, verbose=0).astype(float)[:, outputs]
    z = x.copy()
    z[:, columns] = d_rows
    z = z.astype(np.float32).astype(float)
    d_rows = z[:, columns]
    back, residual = to_physical(pmap, z)
    rel = np.abs(back[:, columns] - x[:, columns]) / np.array([bounds[c][1] - bounds[c][0] for c in columns])

    # The profile workflow reads training coordinates as float32, and for a precisely
    # measured derived parameter that rounding alone moves the recovered pivot enough to
    # change log L: ~1e-3 at the Planck peak for 100 theta_s, far more where the
    # likelihood is steep. The workflow's label audit compares these rows with the
    # backend to Delta chi^2 = 0.01, so only rows whose coordinates survive the rounding
    # are written: those whose label moves by at most ROUNDTRIP_TOLERANCE, estimated
    # from the network's own gradient. The others are far from the peak, where the
    # workflow uses training rows for nothing but starting points.
    from model.network import copy_model_dtype

    net64 = copy_model_dtype(network, "float64")
    xt = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        value = net64(xt, training=False)[:, 0]
    shift = np.sum(tape.gradient(value, xt).numpy() * (back - x), axis=1)
    # The estimate needs the network's gradient to be close to the truth's, which holds
    # near the peak and not thousands of log-units down (on a 13D Planck TT network at
    # iteration 3 it was 7x low there, and the audit failed). Rows that far down serve a
    # profile as neither audit rows nor starting points, so they are left out.
    keep = (np.abs(shift) <= ROUNDTRIP_TOLERANCE) & (loglkl >= loglkl.max() - DEFICIT_LIMIT)
    if not keep.any():
        raise ValueError("no training row survives float32 rounding in these coordinates")
    roundtrip = dict(
        max_relative_pivot_error=float(rel.max()), median_relative_pivot_error=float(np.median(rel)),
        max_residual=float(np.abs(residual).max()),
        label_shift_tolerance=ROUNDTRIP_TOLERANCE, deficit_limit=DEFICIT_LIMIT,
        rows_kept=int(keep.sum()), rows_total=len(keep),
        best_dropped_deficit=(float(loglkl.max() - loglkl[~keep].max()) if (~keep).any() else None),
    )

    view_params = list(metadata.parameters)
    labels = dict(zip(derived.names, derived.labels))
    for target, column, j in zip(targets, columns, range(len(targets))):
        lo, hi = np.float32(d_rows[:, j].min()), np.float32(d_rows[:, j].max())
        lo, hi = float(np.nextafter(lo, np.float32(-np.inf))), float(np.nextafter(hi, np.float32(np.inf)))
        view_params[column] = ParameterInfo(
            name=target, label=labels[target], scale=1.0, lower=lo, upper=hi,
            center=float(d_rows[np.argmax(loglkl), j]), sigma=float(np.std(d_rows[loglkl >= np.quantile(loglkl, 0.95), j])),
        )
    view_names = [p.name for p in view_params]

    for sub in ("likelihood_input", "trained_models", "training_data"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    SurrogateMetadata(parameters=tuple(view_params)).save(output / "metadata.json")
    model.save(output / "trained_models" / f"model_it_{iteration}.keras")
    pmap.save(output / "likelihood_input" / "pivot_map.keras")
    table = pd.DataFrame(z[keep].astype(np.float32), columns=view_names)
    table["loglkl"] = loglkl[keep].astype(np.float32)
    table.to_csv(output / "training_data" / f"data_it_{iteration}.csv", index=False)

    from run.run import settings_file

    run_yaml = settings_file(run)
    cfg = yaml.safe_load(run_yaml.read_text())
    inner_input = run / "likelihood_input" / Path(cfg["likelihood"]["input"]).name
    spec = dict(
        wrapper=cfg["likelihood"]["wrapper"],
        input=str(inner_input),
        parameters=[dict(name=p.name, label=p.label, scale=p.scale, lower=p.lower,
                         upper=p.upper, center=p.center, sigma=p.sigma) for p in view_params],
        physical_names=names,
        pivots=dict(zip(targets, pivots)),
        pivot_map=str(output / "likelihood_input" / "pivot_map.keras"),
        penalty=PENALTY,
    )
    (output / "likelihood_input" / "derived_coordinates.json").write_text(json.dumps(spec, indent=2))
    cfg["likelihood"] = dict(wrapper="derived_coordinates", input="derived_coordinates.json")
    cfg.pop("derived", None)
    (output / run_yaml.name).write_text(yaml.safe_dump(cfg, sort_keys=False))

    # The region of a boost network (sampling/targets.py region_file) is copied into the view
    # as it is, in the run's coordinates: profiles of the view search the network inside it,
    # as they search a run's network (profiles/region.py, ViewRegion and restrict_view above).
    # Without it they would search the bare network over the whole box, where a network
    # trained on boosts has maxima the truth does not have.
    entry, region = source_target["region"], None
    if entry is not None:
        import shutil

        copy = output / "region" / Path(entry["file"]).name
        copy.parent.mkdir()
        shutil.copyfile(run / entry["file"], copy)
        if digest(copy) != entry["sha256"]:
            raise RuntimeError(f"{entry['file']} changed while the view was being made; make it again when the "
                               "run is not boosting")
        region = dict(file=f"region/{copy.name}", sha256=entry["sha256"], source=entry["file"])
    summary = dict(
        source_run=str(run), iteration=iteration, source_target=source_target, source_files=source_files,
        targets=list(targets),
        pivots=dict(zip(targets, pivots)), pivot_selection=pivot_report,
        coordinates=view_names, roundtrip=roundtrip, penalty=PENALTY,
        newton_steps=NEWTON_STEPS,
        source_region=None if entry is None else entry["file"], region=region,
        note="Constraint imposed through the derived emulator; evaluate exact derived "
             "values at the final nodes with `client.py derived-view --nodes`.",
    )
    write_json(output / "view.json", summary)
    return summary
