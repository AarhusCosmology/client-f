import numpy as np
import tensorflow as tf

from .activations import build_activation


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class TargetDenormalization(tf.keras.layers.Layer):
    """
    Fixed affine layer: y_raw = target_mean + target_std * z.

    The mean and standard deviation are stored as non-trainable weights so
    that ``model.get_weights()`` / ``set_weights()`` includes and correctly
    updates them.

    This is NOT sufficient on its own to refresh a model in place between
    iterations, though: the model's input ``Normalization`` layer caches its
    ``mean``/``variance`` as plain tensors derived from its underlying
    ``adapt_mean``/``adapt_variance`` weights, and only recomputes that cache
    inside ``finalize_state()``. ``set_weights()`` updates the underlying
    weights but does not call ``finalize_state()``, so the ``Normalization``
    layer silently keeps normalizing with the OLD statistics until
    ``finalize_state()`` is called explicitly on it afterward. Any in-place
    refresh of a full surrogate model must account for this separately.
    """

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.initial_mean = float(mean)
        self.initial_std = float(std)

    def build(self, input_shape):
        self.target_mean = self.add_weight(
            name="target_mean",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_mean),
            trainable=False,
        )
        self.target_std = self.add_weight(
            name="target_std",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_std),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.target_mean + self.target_std * inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mean": self.initial_mean,
                "std": self.initial_std,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class QuadraticBackground(tf.keras.layers.Layer):
    """Fixed concave quadratic  top - (x - mean)' precision (x - mean) / 2,  as an (n, 1) output.

    A model built with a background is this quadratic plus the network, which then learns
    only the difference between the log-likelihood and the quadratic
    (acquisition.boost.background, docs/boost.md). Mean, precision and top are non-trainable
    weights: saved with the model, read back by model_background, never refitted.
    """

    def __init__(self, mean, precision, top, **kwargs):
        super().__init__(**kwargs)
        self.initial_mean = np.asarray(mean, dtype=np.float64).ravel()
        self.initial_precision = np.asarray(precision, dtype=np.float64)
        self.initial_top = float(top)

    def build(self, input_shape):
        n = len(self.initial_mean)
        self.mean = self.add_weight(name="mean", shape=(n,), initializer="zeros", trainable=False)
        self.precision = self.add_weight(name="precision", shape=(n, n), initializer="zeros",
                                         trainable=False)
        self.top = self.add_weight(name="top", shape=(), initializer="zeros", trainable=False)
        self.mean.assign(self.initial_mean)
        self.precision.assign(self.initial_precision)
        self.top.assign(self.initial_top)
        super().build(input_shape)

    def call(self, inputs):
        z = inputs - self.mean
        return self.top - 0.5 * tf.reduce_sum(tf.matmul(z, self.precision) * z, axis=1,
                                              keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update(mean=self.initial_mean.tolist(),
                      precision=self.initial_precision.tolist(), top=self.initial_top)
        return config


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class RegionWall(tf.keras.layers.Layer):
    """A network's value inside the region of its boosts and the region's wall outside.

    The Keras form of sampling.region.Region.restrict, for consumers that take a Keras model
    and copy it (the profile code, restrict_model below). Inputs: [value (n, 1), x (n, ndim)].
    Where rho(x) <= edge the value passes through; elsewhere the output is
    floor - strength (rho - edge)^2, which does not depend on the network. The region's numbers
    are plain constants at double precision, cast to the layer's own: they are no weights, and
    a copy at another precision is rebuilt from the config.
    """

    def __init__(self, means, whiten, reach, lower, width, edge, floor, strength, **kwargs):
        super().__init__(**kwargs)
        self.means = np.atleast_2d(np.asarray(means, dtype=np.float64))
        self.whiten = np.asarray(whiten, dtype=np.float64).reshape(
            len(self.means), self.means.shape[1], -1)
        self.reach = np.asarray(reach, dtype=np.float64).ravel()
        self.lower = np.asarray(lower, dtype=np.float64).ravel()
        self.width = np.asarray(width, dtype=np.float64).ravel()
        self.edge, self.floor, self.strength = float(edge), float(floor), float(strength)

    def call(self, inputs):
        value, x = inputs

        def constant(array):
            return tf.constant(array, dtype=x.dtype)

        u = (x - constant(self.lower)) / constant(self.width)
        z = tf.einsum("knd,kde->kne", u[None] - constant(self.means)[:, None],
                      constant(self.whiten))
        rho2 = tf.reduce_min(
            tf.reduce_sum(z * z, axis=2) / constant(self.reach ** 2)[:, None], axis=0)
        edge2 = constant(self.edge ** 2)
        # The root only of what is outside: a root has no derivative at the centre of a mode.
        beyond = tf.sqrt(tf.maximum(rho2, edge2)) - constant(self.edge)
        wall = constant(self.floor) - constant(self.strength) * tf.square(beyond)
        return tf.where((rho2 <= edge2)[:, None], value, wall[:, None])

    def get_config(self):
        config = super().get_config()
        config.update(means=self.means.tolist(), whiten=self.whiten.tolist(),
                      reach=self.reach.tolist(), lower=self.lower.tolist(),
                      width=self.width.tolist(), edge=self.edge, floor=self.floor,
                      strength=self.strength)
        return config


@tf.keras.utils.register_keras_serializable(package="CLiENT")
class SmoothStep(tf.keras.layers.Layer):
    """The weight of one of two adjacent networks in a combined emulator (run/combine.py).

    Rising (the upper network's weight): 0 where input coordinate `index` is at most `lower`, 1
    where it is at least `upper`, and in between the quintic smootherstep 6t^5 - 15t^4 + 10t^3
    of t = (x - lower) / (upper - lower), whose first and second derivatives vanish at both
    ends: outside the band exactly one network has weight, and the blend can be differentiated
    twice (the profile code's curvature). `falling` gives the lower network's weight, the same
    curve of 1 - t; the two add to one, and each is exactly 0 or 1 outside the band, so the
    blend (1 - s) A + s B is exactly A or B there whatever the other network's value. Output
    (n, 1), at the input's precision.
    """

    def __init__(self, index, lower, upper, falling=False, **kwargs):
        super().__init__(**kwargs)
        self.index, self.lower, self.upper = int(index), float(lower), float(upper)
        self.falling = bool(falling)
        if not self.upper > self.lower:
            raise ValueError(f"SmoothStep needs upper > lower, not [{self.lower}, {self.upper}]")

    def call(self, x):
        lower = tf.constant(self.lower, dtype=x.dtype)
        width = tf.constant(self.upper - self.lower, dtype=x.dtype)
        one = tf.constant(1.0, x.dtype)
        t = tf.clip_by_value((x[:, self.index] - lower) / width, tf.constant(0.0, x.dtype), one)
        if self.falling:
            t = one - t
        return (t * t * t * (t * (t * 6.0 - 15.0) + 10.0))[:, None]

    def get_config(self):
        config = super().get_config()
        config.update(index=self.index, lower=self.lower, upper=self.upper, falling=self.falling)
        return config


def fit_background(inputs, targets, depth):
    """The quadratic background of a training set: dict(mean, precision, top, kappa, rows).

    Mean and covariance C are those of the rows within `depth` log-units of the best;
    precision = kappa C^-1, with kappa the one number that best matches those rows' depths
    (least squares through the origin); top is the best value. Deliberately crude: one
    covariance and one scale, read off the training set. Raises ValueError when the rows
    cannot carry it (too few, a singular covariance, a scale that is not positive).
    """
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    top = float(y.max())
    near = (top - y) < depth
    ndim = x.shape[1]
    if near.sum() < 5 * ndim:
        raise ValueError(f"quadratic background: {int(near.sum())} rows within {depth:g} of the "
                         f"best, {5 * ndim} needed")
    mean = x[near].mean(axis=0)
    try:
        precision = np.linalg.inv(np.cov(x[near], rowvar=False))
    except np.linalg.LinAlgError as error:
        raise ValueError(f"quadratic background: singular covariance ({error})") from None
    z = x[near] - mean
    half = 0.5 * np.einsum("ni,ij,nj->n", z, precision, z)
    kappa = float(np.sum((top - y[near]) * half) / np.sum(half * half))
    if not (np.isfinite(precision).all() and np.isfinite(kappa) and kappa > 0
            and np.all(np.linalg.eigvalsh(precision) > 0)):
        raise ValueError("quadratic background: the fitted quadratic is not concave")
    return dict(mean=mean, precision=kappa * precision, top=top, kappa=kappa,
                rows=int(near.sum()))


def background_values(background, points):
    """The background at `points`, at double precision."""
    z = np.asarray(points, dtype=np.float64) - np.asarray(background["mean"], dtype=np.float64)
    return float(background["top"]) - 0.5 * np.einsum(
        "ni,ij,nj->n", z, np.asarray(background["precision"], dtype=np.float64), z)


def model_background(model):
    """dict(mean, precision, top) of a model built with a background, or None."""
    for layer in model.layers:
        if isinstance(layer, QuadraticBackground):
            return dict(mean=layer.mean.numpy().astype(np.float64),
                        precision=layer.precision.numpy().astype(np.float64),
                        top=float(layer.top.numpy()))
    return None


def build_model(inputs, targets, n_layers, n_neurons, activation, background=None):
    """
    Build a fully-connected model with input Normalization and output
    TargetDenormalization baked in.

    With `background` (fit_background, or model_background of an earlier model) the public
    output is the background plus the network, and the network's own targets, the ones its
    TargetDenormalization is adapted to, are the training targets minus the background.

    The Normalization layer is adapted to x_train.  The trainable final
    Dense(1) layer predicts a standardized internal variable z; the
    non-trainable TargetDenormalization layer converts it back to raw
    log-likelihood units so the public model output is always in raw units.

    Parameters
    ----------
    inputs : array-like, shape (N, ndim)
        Training inputs used to adapt the input normalization layer.
    targets : array-like, shape (N,) or (N, 1)
        Training targets in raw log-likelihood units.  Used only to
        compute the target mean and standard deviation for the output
        denormalization layer.
    """
    target_array = np.asarray(targets, dtype=np.float64).ravel()
    if background is not None:
        target_array = target_array - background_values(background, inputs)

    fit_mean = float(np.mean(target_array))
    fit_std = float(np.std(target_array))

    norm = tf.keras.layers.Normalization()
    norm.adapt(inputs)

    n_features = inputs.shape[1]
    inputs = tf.keras.Input(shape=(n_features,))
    x = norm(inputs)
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_neurons)(x)
        x = build_activation(activation)(x)

    z_pred = tf.keras.layers.Dense(
        1, name="standardized_loglkl", bias_initializer="zeros")(x)
    outputs = head = TargetDenormalization(
        mean=fit_mean, std=fit_std, name="loglkl_denormalization"
    )(z_pred)

    if background is not None:
        quadratic = QuadraticBackground(
            background["mean"], background["precision"], background["top"],
            name="quadratic_background")(inputs)
        outputs = tf.keras.layers.Add(name="loglkl_with_background")([head, quadratic])

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def load_model(path):
    # A derived-coordinate view's network (derived/coordinates.py) contains layers that
    # are registered when that module is imported; the profile workflow loads views
    # through here without importing it.
    import derived.coordinates  # noqa: F401

    return tf.keras.models.load_model(path, compile=False)


def copy_model_dtype(model, dtype="float64"):
    """Evaluate the same learned coefficients at higher precision, without retraining.

    A float64 evaluation avoids quantized float32 objective values during profile
    line searches. This cannot recover precision already lost in learned weights.
    """
    def clone(layer):
        config = layer.get_config()
        config['dtype'] = dtype
        return type(layer).from_config(config)
    # Clone an uncompiled view: custom training losses need not be serializable
    # merely to evaluate a trained model, and optimizer state is irrelevant here.
    view = tf.keras.Model(model.input, model.output)
    out = tf.keras.models.clone_model(
        view, input_tensors=tf.keras.Input(shape=model.input_shape[1:], dtype=dtype),
        clone_function=clone)
    out.dtype_policy = tf.keras.mixed_precision.Policy(dtype)
    out.set_weights(model.get_weights())
    for layer in out.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.finalize_state()
    return out


def restrict_model(model, region, dtype="float64"):
    """`model` at `dtype`, seen inside `region` only: a Keras model like any other.

    For consumers of a finished network outside the loop (the profile code). A network trained
    on boosts is only good where the boosts' posterior points reach, and it has maxima beyond
    (docs/boost_background_region.md); the loop's own chains see it through
    sampling.region.Region.restrict. This is the same surrogate as a model: the network where
    rho <= region.inflate and the region's wall elsewhere (RegionWall). It can be evaluated,
    differentiated twice and copied with copy_model_dtype; it is not for training or saving.
    """
    from sampling.region import STRENGTH

    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"restrict_model needs the run's Keras model, not {type(model).__name__}: a launcher "
            "that wraps model.network.load_model itself must leave the region to the profile code")
    inner = copy_model_dtype(model, dtype)
    wall = RegionWall(region.means, region.whiten, region.reach, region.lower, region.width,
                      region.inflate, region.floor, STRENGTH, dtype=dtype, name="region_wall")
    out = tf.keras.Model(inner.input, wall([inner.output, inner.input]))
    out.dtype_policy = tf.keras.mixed_precision.Policy(dtype)
    return out


def transfer_warm_start(previous, model):
    """Transfer weights, compensating for newly adapted affine scalers.

    Optimizer state is deliberately not transferred. Both models must have the same
    CLiENT architecture. All compatibility checks precede any assignment.
    """
    old_layers, new_layers = previous.layers, model.layers
    if (len(old_layers) != len(new_layers)
            or any(type(a) is not type(b) for a, b in zip(old_layers, new_layers))
            or [tuple(w.shape) for w in previous.trainable_weights]
            != [tuple(w.shape) for w in model.trainable_weights]):
        raise ValueError("warm-start architectures differ")
    for old, new in zip(old_layers, new_layers):
        if isinstance(old, tf.keras.layers.Activation) and old.get_config()['activation'] != new.get_config()['activation']:
            raise ValueError("warm-start activations differ")
    old_dense = [l for l in old_layers if isinstance(l, tf.keras.layers.Dense)]
    new_dense = [l for l in new_layers if isinstance(l, tf.keras.layers.Dense)]
    corrections = []
    old_norm = [l for l in old_layers if isinstance(l, tf.keras.layers.Normalization)]
    new_norm = [l for l in new_layers if isinstance(l, tf.keras.layers.Normalization)]
    old_head = [l for l in old_layers if isinstance(l, TargetDenormalization)]
    new_head = [l for l in new_layers if isinstance(l, TargetDenormalization)]
    if not (len(old_norm) == len(new_norm) == len(old_head) == len(new_head) == 1
            and old_dense and all(l.use_bias for l in old_dense + new_dense)):
        raise ValueError("function-preserving warm start requires CLiENT affine scalers")
    # Match Normalization.call(), including its epsilon for constant features.
    def input_stats(layer):
        mean = np.asarray(layer.mean, dtype=np.float64).ravel()
        scale = np.maximum(np.sqrt(np.asarray(layer.variance, dtype=np.float64).ravel()),
                           tf.keras.backend.epsilon())
        return mean, scale
    mo, so = input_stats(old_norm[0])
    mn, sn = input_stats(new_norm[0])
    yo, sy_o = float(old_head[0].target_mean.numpy()), float(old_head[0].target_std.numpy())
    yn, sy_n = float(new_head[0].target_mean.numpy()), float(new_head[0].target_std.numpy())
    if not (np.isfinite([yo, sy_o, yn, sy_n]).all() and sy_n > 0
            and np.isfinite(np.r_[mo, so, mn, sn]).all()):
        raise ValueError("invalid warm-start normalization statistics")
    first_w, first_b = [np.asarray(w, dtype=np.float64) for w in old_dense[0].get_weights()]
    first_new = ((sn / so)[:, None] * first_w,
                 first_b + ((mn - mo) / so) @ first_w)
    if len(old_dense) == 1:
        # A linear model has one Dense layer serving both roles.
        last_w, last_b = first_new
    else:
        corrections.append((new_dense[0], first_new))
        last_w, last_b = old_dense[-1].get_weights()
    corrections.append((new_dense[-1],
                        ((sy_o / sy_n) * last_w,
                         (yo - yn + sy_o * last_b) / sy_n)))
    if any(not np.isfinite(w).all() for _, weights in corrections for w in weights):
        raise ValueError("non-finite warm-start scaler compensation")
    for new_weight, old_weight in zip(model.trainable_weights, previous.trainable_weights):
        new_weight.assign(old_weight)
    for layer, weights in corrections:
        layer.set_weights(weights)
