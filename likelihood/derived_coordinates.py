import json
from pathlib import Path

import numpy as np

from .base import BaseLikelihood, ParameterInfo


class DerivedCoordinatesLikelihood(BaseLikelihood):
    """The original likelihood in a derived-coordinate view's coordinates.

    z -> x by the view's pivot solve (derived/coordinates.py), in float64 exactly as the
    view's network does it, then the original backend at x. Where a target is out of the
    pivot's reach the same feasibility penalty is subtracted as in the network, so truth
    and surrogate differ only by what the original network gets wrong.

    Deliberately exposes no `cobaya_model`: profiles/evaluation.py would otherwise call
    Cobaya directly with the view's coordinate names.
    """

    def __init__(self, spec_path):
        from derived.coordinates import to_physical  # noqa: F401  (TF only when needed)
        from likelihood.base import build_likelihood
        from model.network import copy_model_dtype, load_model

        self._spec = json.loads(Path(spec_path).read_text())
        self._inner = build_likelihood(self._spec["wrapper"], self._spec["input"])
        if self._inner.param_names != self._spec["physical_names"]:
            raise ValueError("original backend parameter order differs from the view's")
        self._map = copy_model_dtype(load_model(self._spec["pivot_map"]), "float64")
        self._penalty = float(self._spec["penalty"])
        super().__init__()

    def _extract_params(self):
        return [ParameterInfo(**p) for p in self._spec["parameters"]]

    def _extract_derived(self):
        # The view's own coordinates are not also derived parameters of the view; their
        # exact values come from exact().
        coordinates = {p["name"] for p in self._spec["parameters"]}
        self._derived_columns = [i for i, p in enumerate(self._inner._derived)
                                 if p.name not in coordinates]
        return [self._inner._derived[i] for i in self._derived_columns]

    @property
    def derived_input_names(self):
        return self.param_names

    def physical(self, z):
        """(x, residual) for one point z."""
        from derived.coordinates import to_physical

        x, residual = to_physical(self._map, np.asarray(z, float)[None, :])
        return x[0], residual[0]

    def loglkl(self, z):
        x, residual = self.physical(z)
        return float(self._inner.loglkl(x)) - self._penalty * float(residual @ residual)

    def loglkl_and_derived(self, z):
        loglkl, derived, _, _ = self.exact(z)
        return loglkl, derived[self._derived_columns]

    def exact(self, z):
        """(objective, every derived value of the original backend, x, residual) at z.

        The derived values include those of the view's coordinates: the exact value of
        a derived coordinate at a point, against which its (emulated) coordinate value
        can be compared.
        """
        x, residual = self.physical(z)
        loglkl, derived = self._inner.loglkl_and_derived(x)
        return float(loglkl) - self._penalty * float(residual @ residual), np.asarray(derived, float), x, residual

    def close(self):
        inner = getattr(self, "_inner", None)
        if inner is not None:
            if hasattr(inner, "cobaya_model"):
                inner.cobaya_model.close()
            elif hasattr(inner, "close"):
                inner.close()
