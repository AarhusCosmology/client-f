"""The error of a network at points it had not seen.

Every acquisition is evaluated with the true likelihood, and the network that chose the
points had not been trained on them. Its predictions there against the true values are a
test of that network that costs nothing: at the tempered batch, where the next training
set is being built, and at the boost points, where the posterior is.
"""
import numpy as np


def held_out_error(predicted, truth, ignore=None):
    """Network minus truth at an acquisition: dict(n, bias, sd, rms, largest, left_out).

    ignore : a value that marks a point the likelihood could not evaluate (a launcher may
        give such points a fixed value so that they are trained on); they are left out of
        the error and counted in `left_out`, with every point whose value is not finite.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    keep = np.isfinite(predicted) & np.isfinite(truth)
    if ignore is not None:
        keep &= truth != ignore
    error = predicted[keep] - truth[keep]
    out = dict(n=int(len(error)), left_out=int(len(truth) - len(error)))
    if len(error):
        out.update(bias=float(error.mean()), sd=float(error.std()), rms=float(np.sqrt(np.mean(error ** 2))),
                   largest=float(np.abs(error).max()))
    return out


def describe(error):
    """One line for the log."""
    if not error["n"]:
        return "no point with a true value"
    return (f"rms {error['rms']:.3g} (mean {error['bias']:+.3g}, scatter {error['sd']:.3g}, largest {error['largest']:.3g}) "
            f"at {error['n']} points" + (f", {error['left_out']} left out" if error["left_out"] else ""))
