import numpy as np

from .base import BaseLikelihood, ParameterInfo


def native_value(result):
    # A result without a log-posterior is a point without a value, which the loop drops.
    return float(result.logpost) if result.logpost is not None else -np.inf


class CobayaLikelihood(BaseLikelihood):
    def __init__(self, yaml_path):
        # Load the Cobaya modules
        from cobaya.model import get_model
        from cobaya.yaml import yaml_load_file

        # Store the Cobaya objects as attributes
        self.cobaya_info = yaml_load_file(yaml_path)
        self.cobaya_model = get_model(self.cobaya_info)

        # Let BaseLikelihood extract parameters and initialize bounds state
        super().__init__()

    def _extract_params(self):
        params = []
        for name, info in self.cobaya_info["params"].items():
            # Keep only entries with metadata and a prior;
            # these define the coordinates varied by this wrapper.
            if not isinstance(info, dict):
                continue
            prior = info.get("prior")
            if prior is None:
                continue

            # Use Cobaya's reference point as the center for optional
            # n-sigma prior restriction. Fall back to the proposal width
            # when ref does not provide an explicit scale.
            ref = info.get("ref")
            proposal = info.get("proposal")
            if isinstance(ref, dict):
                center = ref.get("loc")
                sigma = ref.get("scale", proposal)
            else:
                center = ref
                sigma = proposal

            # Store a plain label without math delimiters.
            label = info.get("latex", name).replace("$", "")

            # Store the metadata needed by the shared BaseLikelihood methods.
            params.append(
                ParameterInfo(
                    name=name,
                    label=label,
                    scale=1.0,
                    lower=prior.get("min"),
                    upper=prior.get("max"),
                    center=center,
                    sigma=sigma,
                )
            )

        return params

    def _extract_derived(self):
        # Take the names from Cobaya's parameterization rather than from the YAML:
        # it is the list LogPosterior.derived is reported in, in the same order.
        #
        # Cobaya also adds a chi2__<type> total for every likelihood type (the Planck
        # likelihoods are type CMB). Those are the log-likelihood itself, not smooth
        # functions of the parameters, and they depend on nuisance parameters the input
        # selection knows nothing about, so they are left out; _derived_columns keeps
        # the positions of the rest in what Cobaya reports.
        names = list(self.cobaya_model.parameterization.derived_params())
        self._n_reported = len(names)
        self._derived_columns = [i for i, n in enumerate(names) if not n.startswith("chi2__")]

        derived = []
        for name in (names[i] for i in self._derived_columns):
            info = self.cobaya_info["params"].get(name)
            label = info.get("latex", name) if isinstance(info, dict) else name
            derived.append(ParameterInfo(name=name, label=label.replace("$", "")))

        return derived

    @property
    def derived_input_names(self):
        # Best effort from Cobaya's bookkeeping, part of which is private; anything
        # unexpected falls back to every input, which is always correct.
        try:
            names = self._derived_dependencies()
        except Exception:
            return self.param_names
        return names or self.param_names

    def _derived_dependencies(self):
        """Sampled parameters the derived parameters depend on, resolved through
        derived-parameter functions, input parameters that are functions of sampled
        ones, and, for a component's outputs, every sampled parameter that reaches that
        component.

        The last uses Cobaya's own record, Model.sampled_dependence, which is what it
        recomputes components by. It follows requirements between components as well as
        parameters: CAMB's outputs (H0, sigma8, ...) come from `camb`, whose own inputs
        are only As and ns, but `camb` requires `camb.transfers`, which takes the rest.
        A name that resolves to none of these raises, so the caller falls back to every
        input rather than silently dropping a dependence."""
        model = self.cobaya_model
        parameterization = model.parameterization
        functions = parameterization._derived_dependencies
        provider = {
            output: component
            for component in model.components
            for output in (component.output_params or [])
        }
        reaches = {name: set(components) for name, components in model.sampled_dependence.items()}
        inputs_of = parameterization.sampled_input_dependence()
        constants = set(parameterization.constant_params())

        found, pending, seen = set(), list(self.derived_names), set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in functions:
                pending.extend(functions[name])
            elif name in provider:
                found.update(p for p, components in reaches.items() if provider[name] in components)
            elif name in reaches:
                found.add(name)
            elif any(name in inputs for inputs in inputs_of.values()):
                found.update(p for p, inputs in inputs_of.items() if name in inputs)
            elif name not in constants:
                raise LookupError(f"cannot resolve what {name!r} depends on")
        return [name for name in self.param_names if name in found]

    def loglkl_and_derived(self, x):
        position = {param.name: float(value) for param, value in zip(self._params, x)}
        result = self.cobaya_model.logposterior(position, return_derived=True)
        loglkl = native_value(result)

        # Outside Cobaya's own prior, result.derived is an empty list rather than a
        # list of NaNs, and a point whose log-posterior is not finite has no derived
        # values worth keeping either.
        derived = np.full(self.n_derived, np.nan)
        if np.isfinite(loglkl) and len(result.derived) == self._n_reported:
            derived[:] = np.asarray(result.derived, dtype=float)[self._derived_columns]

        return loglkl, derived

    def loglkl(self, x):
        # Convert the input vector into the parameter dictionary expected by Cobaya.
        position = {param.name: float(value) for param, value in zip(self._params, x)}

        # Evaluate Cobaya's native log-posterior without derived parameters.
        result = self.cobaya_model.logposterior(position, return_derived=False)

        return native_value(result)
