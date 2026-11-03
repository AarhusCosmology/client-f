import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


def save_history(history, path):
    from run.storage import atomic_path
    with atomic_path(path) as temporary:
        pd.DataFrame(history).to_csv(temporary, index=False)


def build_optimizer(learning_rate):
    """Adam. The alternatives explored during development -- Muon, and an Adam variant
    carrying a K-FAC style input preconditioner -- are not included: Muon measured 10.6x
    slower per epoch on this architecture, and the preconditioner recovered only 11-55%
    of an effect that was itself harmful to the credible metric. See CHANGES.md."""
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class StopAfterScheduleExhausted(Callback):
    """EarlyStopping that does not start counting until the lr schedule bottoms out.

    A plain EarlyStopping cannot be used with an annealing schedule. Annealing does what
    it is meant to -- it smooths the loss curve -- and on a smooth, monotonically creeping
    curve a patience with min_delta=0 NEVER fires, because every epoch improves the best
    by some tiny amount and resets the counter. Measured: an annealed arm ran past 1500
    epochs while its fixed-lr twin stopped at 1184, so the two were being halted by
    different mechanisms and every epoch comparison between them was meaningless.

    A relative-improvement test does not fix it either. Simulated on two recorded annealed
    curves, "stop if the best has not improved by 1% over 50 epochs" never triggered on
    either: post-min_lr improvement is slow but genuinely monotone.

    So: let the schedule finish, then stop after a fixed grace period. Robust where "has
    it stopped improving" is not, because on these curves it does not stop improving.

    The grace period is a deliberate trade. Simulated on the same curves, stopping at
    min_lr + 100 saved ~99 epochs for a 1.11x val_loss cost -- and a 2.01x val_loss
    spread within the annealed family produced credible-metric medians of 0.0150 against
    0.0138, a 9% difference far below that metric's 0.025 noise floor. Grinding out the
    tail buys a number that does not predict the metric anyone cares about.
    """

    def __init__(self, lr_min, grace, verbose=1, restore=True):
        super().__init__()
        self.lr_min, self.grace, self.verbose, self.restore = lr_min, grace, verbose, restore
        self.best = float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.epoch_at_min = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss")
        if val_loss is None:
            return
        if val_loss < self.best:
            self.best, self.best_epoch = val_loss, epoch
            self.best_weights = self.model.get_weights()
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if lr <= self.lr_min * 1.01:
            if self.epoch_at_min is None:
                self.epoch_at_min = epoch
            elif epoch - self.epoch_at_min >= self.grace:
                self.model.stop_training = True
                if self.verbose:
                    print(f"\nStopping: lr reached {self.lr_min:g} at epoch "
                          f"{self.epoch_at_min + 1}, {self.grace} further epochs elapsed. "
                          + (f"Restoring best weights from epoch {self.best_epoch + 1}."
                             if self.restore else
                             f"Keeping the last weights (epoch {epoch + 1}); the lowest validation "
                             f"loss was at epoch {self.best_epoch + 1}."),
                          flush=True)

    def on_train_end(self, logs=None):
        if self.restore and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_callbacks(lr_schedule, lr_factor, lr_patience, lr_min, lr_grace, patience, restore_best=True):
    """ReduceLROnPlateau plus a schedule-gated stopper, or the plain EarlyStopping.

    lr_schedule 'plateau' anneals (a YAML without the key gets 'none'). On the 31D ridge at a matched epoch
    budget it gave 2.1x lower validation loss AND 2.4x better credible metric than a fixed
    rate in that dense-design experiment. Sparse 58D tests later worsened posterior
    accuracy: this is not a general recommendation; the default remains 'none'.
    The mechanism in the dense-design loss curve is visible: the
    coefficient of variation over the 250 epochs before the best epoch collapses from 124%
    to 24%, and the number of those epochs within 10% of the best rises from 2 to 97. At a
    fixed rate the optimiser was sampling from a wide noise ball and restore_best_weights
    was catching a lucky dip; annealed, it settles.

    lr_schedule 'none' restores the previous behaviour exactly.

    restore_best=False keeps the weights at the end (for a warm-started network
    of the boost phase, whose validation loss is an in-sample one at its first epochs).
    """
    if lr_schedule == "none":
        return [EarlyStopping(monitor="val_loss", patience=patience,
                              restore_best_weights=restore_best, verbose=1)]
    if lr_schedule != "plateau":
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}: use 'plateau' or 'none'")
    return [
        ReduceLROnPlateau(monitor="val_loss", factor=lr_factor,
                          patience=lr_patience, min_lr=lr_min, verbose=1),
        StopAfterScheduleExhausted(lr_min=lr_min, grace=lr_grace, restore=restore_best),
    ]


def train_model(
    model,
    inputs,
    targets,
    loss,
    learning_rate,
    n_epochs,
    batch_size,
    validation_split,
    patience,
    lr_schedule="plateau",
    lr_factor=0.3,
    lr_patience=75,
    lr_min=1e-6,
    lr_grace=100,
    validation_data=None,
    restore_best=True,
):
    model.compile(
        optimizer=build_optimizer(learning_rate),
        loss=loss,
        jit_compile=True,
    )

    history = model.fit(
        inputs,
        targets,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_split=validation_split if validation_data is None else 0.0,
        validation_data=validation_data,
        verbose=2,
        callbacks=build_callbacks(lr_schedule, lr_factor, lr_patience,
                                  lr_min, lr_grace, patience, restore_best=restore_best),
    )

    # The epoch whose weights the model holds: the lowest validation loss, or the last.
    best_epoch_idx = min(
        range(len(history.history["val_loss"])),
        key=lambda i: history.history["val_loss"][i],
    ) if restore_best else len(history.history["val_loss"]) - 1
    metrics = {
        "epoch": best_epoch_idx + 1,
        "loss": float(history.history["loss"][best_epoch_idx]),
        "val_loss": float(history.history["val_loss"][best_epoch_idx]),
    }
    return history, metrics
