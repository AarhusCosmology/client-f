from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    label: str
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None
    center: float | None = None
    sigma: float | None = None


class BaseLikelihood(ABC):
    # What `loglkl` returns for a given backend input, as a number in the run's objective
    # identity (run/identity.py). A wrapper raises its own when a change to it alters the
    # value returned at a point (for example whether the backend's prior is included), so
    # that a run started before is not continued across the change unnoticed. The text of a
    # wrapper file is not part of the identity: an edit that leaves the values alone is free.
    objective_convention = 1

    def __init__(self):
        self._params = self._extract_params()
        self._derived = self._extract_derived()
        self._effective_bounds = None

        overlap = set(self.param_names) & set(self.derived_names)
        if overlap:
            raise ValueError(f"Parameters both varying and derived: {sorted(overlap)}")

    @abstractmethod
    def _extract_params(self):
        pass

    def _extract_derived(self):
        """
        Return the ``ParameterInfo`` of every derived parameter the backend reports alongside its log-probability, in the order ``loglkl_and_derived`` returns them. Backends without derived parameters need not override this.
        """
        return []

    @abstractmethod
    def loglkl(self, x):
        """
        Evaluate the backend's native log-probability at ``x``. For MontePython, this calls ``compute_lkl``. For Cobaya, this calls ``logposterior(..., return_derived=False)`` and returns its ``logpost`` value. Despite the method name, the returned value will include both log-likelihood and backend-defined log-prior contributions. Additional bounds imposed by ``restrict_prior_bounds`` are handled separately by ``logprior``.
        """
        pass

    @property
    def param_names(self):
        return [param.name for param in self._params]

    @property
    def param_labels(self):
        return [param.label for param in self._params]

    @property
    def param_scales(self):
        return [param.scale for param in self._params]

    @property
    def param_centers(self):
        return [param.center for param in self._params]

    @property
    def param_sigmas(self):
        return [param.sigma for param in self._params]

    @property
    def ndim(self):
        return len(self._params)

    # getattr: a subclass that sets up its own state without calling
    # BaseLikelihood.__init__ simply has no derived parameters.
    @property
    def derived_names(self):
        return [param.name for param in getattr(self, "_derived", [])]

    @property
    def derived_labels(self):
        return [param.label for param in getattr(self, "_derived", [])]

    @property
    def derived_scales(self):
        return [param.scale for param in getattr(self, "_derived", [])]

    @property
    def n_derived(self):
        return len(getattr(self, "_derived", []))

    @property
    def derived_input_names(self):
        """
        The varying parameters the derived parameters can depend on, in ``param_names`` order. All of them unless a backend knows better; a derived-parameter emulator uses only these as inputs.
        """
        return self.param_names

    def loglkl_and_derived(self, x):
        """
        Evaluate ``loglkl`` at ``x`` together with the backend's derived parameters. Returns ``(loglkl, derived)`` with ``derived`` a float array of length ``n_derived`` in ``derived_names`` order. Derived values are NaN wherever the backend could not supply them, which includes every point whose ``loglkl`` is not finite. This default reports none; backends that have derived parameters override it.
        """
        return self.loglkl(x), np.full(self.n_derived, np.nan)

    def evaluate_with_derived(self, x):
        """
        ``loglkl_and_derived`` flattened to one row ``[loglkl, *derived]``, the form ``broadcast_and_evaluate`` carries between MPI ranks. ``split_evaluations`` undoes it.
        """
        loglkl, derived = self.loglkl_and_derived(x)
        return np.concatenate(([loglkl], derived))

    def split_evaluations(self, values):
        """
        Split rows produced by ``evaluate_with_derived`` into ``(targets, derived)`` with shapes ``(n,)`` and ``(n, n_derived)``. Also accepts the empty array an MPI worker gets back.
        """
        values = np.asarray(values, dtype=float).reshape(-1, 1 + self.n_derived)
        return values[:, 0], values[:, 1:]

    @property
    def prior_bounds(self):
        if self._effective_bounds is not None:
            return dict(self._effective_bounds)
        return {param.name: (param.lower, param.upper) for param in self._params}

    def restrict_prior_bounds(self, n_sigma):
        restricted_bounds = {}
        for param in self._params:
            if param.center is None or param.sigma is None:
                # Nothing to restrict it around (a Cobaya parameter without ref or proposal).
                if param.lower is None or param.upper is None:
                    raise ValueError(
                        f"Parameter {param.name} has infinite prior bounds and no center or sigma: "
                        "prior.n_sigma cannot restrict it"
                    )
                restricted_bounds[param.name] = (param.lower, param.upper)
                continue
            lower = param.center - n_sigma * param.sigma
            upper = param.center + n_sigma * param.sigma
            if param.lower is not None:
                lower = max(lower, param.lower)
            if param.upper is not None:
                upper = min(upper, param.upper)
            restricted_bounds[param.name] = (lower, upper)
        self._effective_bounds = restricted_bounds

    def logprior(self, x):
        """
        Return the additional bounds-based log-prior at ``x``. Returns zero inside the current effective bounds and negative infinity outside them. This does not reproduce any prior terms already evaluated by the backend.
        """
        bounds = self.prior_bounds
        for value, param in zip(x, self._params):
            lower, upper = bounds[param.name]
            if lower is not None and value < lower:
                return -np.inf
            if upper is not None and value > upper:
                return -np.inf
        return 0.0

    def logpost(self, x):
        """
        Return the total log-posterior value at ``x``. The result is the sum of the backend's native log-probability, returned by ``loglkl``, and the additional bounds-based log-prior, returned by ``logprior``. Consequently, the result is negative infinity when ``x`` lies outside the current effective bounds.
        """
        lp = self.logprior(x)
        if not np.isfinite(lp):
            return -np.inf
        return self.loglkl(x) + lp


def build_likelihood(wrapper, input_path):
    if wrapper == "montepython":
        from .montepython import MontePythonLikelihood

        return MontePythonLikelihood(param_path=input_path)
    if wrapper == "cobaya":
        from .cobaya import CobayaLikelihood

        return CobayaLikelihood(yaml_path=input_path)
    if wrapper == "derived_coordinates":
        # A derived-coordinate view of another run (derived/coordinates.py).
        from .derived_coordinates import DerivedCoordinatesLikelihood

        return DerivedCoordinatesLikelihood(spec_path=input_path)
    raise ValueError(f"Unknown likelihood wrapper: {wrapper!r}")
