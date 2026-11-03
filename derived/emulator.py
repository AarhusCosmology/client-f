from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model.activations import build_activation
from run.storage import atomic_path, save_model
from training.training import save_history, train_model


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class SelectInputs(tf.keras.layers.Layer):
    """Keep only the columns the derived parameters depend on.

    Part of the model, so a caller passes the full parameter vector exactly as it does
    to the log-likelihood network, and a likelihood on derived parameters can call both
    on the same positions inside the sampler.
    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class DerivedDenormalization(tf.keras.layers.Layer):
    """Fixed per-output affine layer d = mean + std * z.

    The vector counterpart of model.network.TargetDenormalization: the network predicts
    standardised values, which puts derived parameters of very different magnitudes
    (A_s ~ 2e-9, H0 ~ 67) on one footing in the loss, and this returns physical units.
    """

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.initial_mean = [float(v) for v in mean]
        self.initial_std = [float(v) for v in std]

    def build(self, input_shape):
        shape = (len(self.initial_mean),)
        self.derived_mean = self.add_weight(
            name="derived_mean",
            shape=shape,
            initializer=tf.keras.initializers.Constant(self.initial_mean),
            trainable=False,
        )
        self.derived_std = self.add_weight(
            name="derived_std",
            shape=shape,
            initializer=tf.keras.initializers.Constant(self.initial_std),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.derived_mean + self.derived_std * inputs

    def get_config(self):
        config = super().get_config()
        config.update({"mean": self.initial_mean, "std": self.initial_std})
        return config


def build_derived_model(inputs, derived, input_indices, n_layers, n_neurons, activation):
    """Normalization -> MLP -> Dense(k) -> DerivedDenormalization.

    Returns (model, standardized_model, mean, std). The two models share every layer;
    the standardised one, whose output is (d - mean) / std, is the one trained, and the
    other is the one saved and used.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    derived = np.asarray(derived, dtype=np.float64)

    # A column that is all NaN, or constant, gets unit scaling; the network then
    # learns (or is masked to) a constant.
    mean = np.nanmean(derived, axis=0) if len(derived) else np.zeros(derived.shape[1])
    std = np.nanstd(derived, axis=0) if len(derived) else np.ones(derived.shape[1])
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)

    norm = tf.keras.layers.Normalization()
    norm.adapt(inputs[:, input_indices])

    x_in = tf.keras.Input(shape=(inputs.shape[1],))
    x = norm(SelectInputs(input_indices, name="derived_inputs")(x_in))
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_neurons)(x)
        x = build_activation(activation)(x)
    z = tf.keras.layers.Dense(derived.shape[1], name="standardized_derived")(x)
    d = DerivedDenormalization(mean, std, name="derived_denormalization")(z)

    return tf.keras.Model(x_in, d), tf.keras.Model(x_in, z), mean, std


def masked_mse(y_true, y_pred):
    """Mean squared error over the finite targets only.

    A point may lack some derived values (a parameter added to the likelihood partway
    through a run) and still carry information about the others, so NaN targets are
    masked per element rather than dropping the row.
    """
    finite = tf.math.is_finite(y_true)
    safe_true = tf.where(finite, y_true, tf.zeros_like(y_true))
    diff = tf.where(finite, y_pred - safe_true, tf.zeros_like(y_pred))
    count = tf.reduce_sum(tf.cast(finite, y_pred.dtype), axis=-1)
    return tf.reduce_sum(tf.square(diff), axis=-1) / tf.maximum(count, 1.0)


def resolve_input_indices(setting, param_names, auto_inputs):
    """Column indices, in parameter order, for DerivedConfig.inputs."""
    if setting == "auto":
        names = list(auto_inputs) or list(param_names)
    elif setting == "all":
        names = list(param_names)
    elif isinstance(setting, str):
        raise ValueError(f"derived.inputs must be 'auto', 'all' or a list, got {setting!r}")
    else:
        names = list(setting)
    unknown = sorted(set(names) - set(param_names))
    if unknown:
        raise ValueError(f"derived.inputs names unknown parameters: {unknown}")
    return [i for i, name in enumerate(param_names) if name in names]


def predict_derived(model, positions, batch_size=65536):
    positions = np.asarray(positions, dtype=np.float32)
    return model.predict(positions, batch_size=batch_size, verbose=0).astype(np.float64)


def evaluate_derived_model(model, inputs, derived, loglkl, names, core_fraction=0.2):
    """Held-out accuracy per derived parameter, overall and in the high-likelihood core.

    The core is the top `core_fraction` of the held-out points by log-likelihood: the
    training set follows L^(1/T), so its best points sit nearest the posterior, which
    is where the emulator has to be accurate. `rms_over_spread` compares the error with
    the parameter's own spread over the same points; the spread of the core is still
    wider than the posterior, so the definitive test is against exact values on the
    posterior chain.
    """
    predicted = predict_derived(model, inputs)
    core = loglkl >= np.quantile(loglkl, 1.0 - core_fraction)
    rows = []
    for j, name in enumerate(names):
        row = {"name": name}
        for tag, mask in (("", np.ones(len(inputs), bool)), ("core_", core)):
            ok = mask & np.isfinite(derived[:, j])
            error = predicted[ok, j] - derived[ok, j]
            spread = float(np.std(derived[ok, j])) if ok.any() else np.nan
            rms = float(np.sqrt(np.mean(error**2))) if ok.any() else np.nan
            row.update({
                f"{tag}n": int(ok.sum()),
                f"{tag}rms": rms,
                f"{tag}max_abs": float(np.max(np.abs(error))) if ok.any() else np.nan,
                f"{tag}spread": spread,
                f"{tag}rms_over_spread": rms / spread if spread > 0 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def train_derived_model(inputs, derived, loglkl, input_indices, names, cfg, rng=None):
    """Fit the emulator; returns (model, history, metrics, report).

    Rows with no finite derived value carry no information and are dropped. The
    validation points are held out of the scaling statistics as well as the fit, and
    the report is computed on them.
    """
    rng = np.random.default_rng() if rng is None else rng
    inputs = np.asarray(inputs, dtype=np.float32)
    derived = np.asarray(derived, dtype=np.float64)
    loglkl = np.asarray(loglkl, dtype=np.float64).ravel()

    keep = np.isfinite(derived).any(axis=1)
    inputs, derived, loglkl = inputs[keep], derived[keep], loglkl[keep]
    n_val = max(1, int(round(cfg.validation_split * len(inputs))))
    if len(inputs) - n_val < 10:
        raise ValueError(
            f"only {len(inputs)} points have derived values; too few to train on"
        )
    order = rng.permutation(len(inputs))
    val, train = order[:n_val], order[n_val:]

    model, standardized, mean, std = build_derived_model(
        inputs[train], derived[train], input_indices,
        cfg.n_layers, cfg.n_neurons, cfg.activation,
    )
    standardized_targets = ((derived - mean) / std).astype(np.float32)

    history, metrics = train_model(
        model=standardized,
        inputs=inputs[train],
        targets=standardized_targets[train],
        loss=masked_mse,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        validation_split=0.0,
        validation_data=(inputs[val], standardized_targets[val]),
        patience=cfg.patience,
        lr_schedule=cfg.lr_schedule,
        lr_factor=cfg.lr_factor,
        lr_patience=cfg.lr_patience,
        lr_min=cfg.lr_min,
        lr_grace=cfg.lr_grace,
    )
    metrics["n_train"] = len(train)
    metrics["n_validation"] = len(val)

    report = evaluate_derived_model(model, inputs[val], derived[val], loglkl[val], names)
    return model, history, metrics, report


def model_path(derived_data_dir, iteration):
    return Path(derived_data_dir) / f"derived_model_it_{iteration}.keras"


def load_derived_model(path):
    return tf.keras.models.load_model(path, compile=False)


def format_report(report):
    lines = [f"   {'parameter':<20}{'rms':>12}{'rms/spread':>12}{'core rms':>12}"
             f"{'core rms/spread':>17}"]
    for row in report.itertuples():
        lines.append(
            f"   {row.name:<20}{row.rms:>12.4g}{row.rms_over_spread:>12.3g}"
            f"{row.core_rms:>12.4g}{row.core_rms_over_spread:>17.3g}"
        )
    return "\n".join(lines)


def train_and_save(inputs, derived, loglkl, param_names, metadata, cfg,
                   derived_data_dir, iteration, rng=None, log=print):
    """Train the iteration's emulator and write model, history and report to
    derived_data/. Returns (model, report)."""
    input_indices = resolve_input_indices(cfg.inputs, param_names, metadata.inputs)
    n_known = int(np.isfinite(np.asarray(derived, dtype=float)).any(axis=1).sum())
    log(
        f"Training derived-parameter emulator on {n_known} points: "
        f"{len(metadata.names)} outputs from {len(input_indices)} inputs "
        f"({', '.join(param_names[i] for i in input_indices)})"
    )
    model, history, metrics, report = train_derived_model(
        inputs, derived, loglkl, input_indices, metadata.names, cfg, rng=rng
    )

    derived_data_dir = Path(derived_data_dir)
    derived_data_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path(derived_data_dir, iteration))
    save_history(history.history, derived_data_dir / f"derived_history_it_{iteration}.csv")
    with atomic_path(derived_data_dir / f"derived_report_it_{iteration}.csv") as temporary:
        report.to_csv(temporary, index=False)

    log(
        f"Derived emulator: best epoch {metrics['epoch']}, val_loss "
        f"{metrics['val_loss']:.3g} (standardised units), held-out accuracy on "
        f"{metrics['n_validation']} points:\n{format_report(report)}"
    )
    return model, report
