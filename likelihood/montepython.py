import sys
import tempfile

import numpy as np
import yaml

from .base import BaseLikelihood, ParameterInfo


class MontePythonLikelihood(BaseLikelihood):
    def __init__(self, param_path, mp_config_path="config/montepython.yaml"):
        # Load montepython.yaml and extract the paths to
        # the .conf file and the montepython directory
        with open(mp_config_path, encoding="utf-8") as f:
            mp_config = yaml.safe_load(f)
        conf_path = mp_config["conf"]
        montepython_path = mp_config["path"]

        # Load MontePython modules
        if montepython_path not in sys.path:
            sys.path.append(montepython_path)
        from initialise import initialise as mp_initialize
        from sampler import compute_lkl

        # Create a temporary directory for MontePython output stored as an attribute
        # to prevent it from being garbage collected after initialization
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mp_")

        # Initialize MontePython
        mp_command = f"run -p {param_path} --conf {conf_path} -o {self._temp_dir.name} --chain-number 0"
        self.cosmo, self.data, _, _ = mp_initialize(mp_command)
        self.compute_lkl = compute_lkl

        # Let BaseLikelihood extract parameters and initialize bounds state
        super().__init__()

    def _extract_params(self):
        from io_mp import get_tex_name

        params = []

        # Get the names of the varying parameters
        names = self.data.get_mcmc_parameters(["varying"])

        for name in names:
            # Read MontePython's stored metadata for this parameter
            mp_param = self.data.mcmc_parameters[name]

            # MontePython stores parameter values in scaled units. Convert the
            # center, bounds, and proposal width to the physical units used by x
            scale = mp_param["scale"]
            initial = mp_param["initial"]
            center = initial[0] * scale
            lower = None if initial[1] is None else initial[1] * scale
            upper = None if initial[2] is None else initial[2] * scale
            sigma = initial[3] * scale

            # Store a plain label without math/control characters
            label = get_tex_name(name, scale)
            label = label.replace("$", "").replace("*", "").replace("&", "")

            # Store the metadata needed by the shared BaseLikelihood methods
            params.append(
                ParameterInfo(
                    name=name,
                    label=label,
                    scale=scale,
                    lower=lower,
                    upper=upper,
                    center=center,
                    sigma=sigma,
                )
            )

        return params

    def _extract_derived(self):
        from io_mp import get_tex_name

        derived = []

        # 'derived' values come from CLASS, and compute_lkl divides them by their
        # scale, exactly as it stores the varying ones. 'derived_lkl' values are
        # written by a likelihood and never rescaled. ParameterInfo.scale is kept as
        # the factor from MontePython's stored value to physical units, so both are
        # reported physical, the same convention x already uses.
        #
        # The label is the UNSCALED one. get_tex_name(name, scale) would prefix the
        # scale ('10^{-9}A_s'), which describes MontePython's stored number, not the
        # physical value CLiENT reports.
        for role, rescaled in (("derived", True), ("derived_lkl", False)):
            for name in self.data.get_mcmc_parameters([role]):
                scale = self.data.mcmc_parameters[name]["scale"] if rescaled else 1.0
                label = get_tex_name(name)
                label = label.replace("$", "").replace("*", "").replace("&", "")
                derived.append(ParameterInfo(name=name, label=label, scale=scale))

        return derived

    @property
    def derived_input_names(self):
        # CLASS computes the 'derived' role from the 'cosmo' arguments alone, so the
        # nuisance parameters cannot enter. A 'derived_lkl' value comes out of a
        # likelihood and may depend on anything.
        roles = {self.data.mcmc_parameters[name]["role"] for name in self.derived_names}
        if roles <= {"derived"}:
            return [
                param.name for param in self._params
                if self.data.mcmc_parameters[param.name]["role"] == "cosmo"
            ]
        return self.param_names

    def loglkl_and_derived(self, x):
        # compute_lkl writes the derived values only after a successful computation.
        # A failed CLASS run returns boundary_loglike early and leaves the PREVIOUS
        # point's values in place. The finiteness check below discards those; the
        # blanking here also covers any path that returns a finite value without
        # refreshing them, so a point can never inherit its predecessor's values.
        for param in self._derived:
            self.data.mcmc_parameters[param.name]["current"] = np.nan

        loglkl = self.loglkl(x)

        derived = np.full(self.n_derived, np.nan)
        if np.isfinite(loglkl):
            derived[:] = [
                self.data.mcmc_parameters[param.name]["current"] * param.scale
                for param in self._derived
            ]

        return loglkl, derived

    def loglkl(self, x):
        # Convert the input vector into MontePython's internal unscaled units
        for value, param in zip(x, self._params):
            self.data.mcmc_parameters[param.name]["current"] = value / param.scale

        # This wrapper treats every call as an independent external evaluation
        # Do not rely on MontePython's internal MCMC caching state
        self.data.need_cosmo_update = True
        for likelihood in self.data.lkl.values():
            likelihood.need_update = True

        # Build CLASS arguments for this point and evaluate the log-likelihood.
        self.data.update_cosmo_arguments()
        loglkl = float(self.compute_lkl(self.cosmo, self.data))

        # MontePython uses data.boundary_loglike, typically -1e30, as a rejection
        # sentinel for invalid points, e.g. failed CLASS evaluations or prior-boundary
        # violations. Since finite likelihood terms may be added to this sentinel,
        # treat any value still in that extreme negative regime as -inf
        if not np.isfinite(loglkl):
            return -np.inf
        if loglkl <= self.data.boundary_loglike / 2:
            return -np.inf

        return loglkl

    def close(self):
        # Release the temporary MontePython output directory, if it still exists
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self):
        # Support use as a context manager with "with MontePythonLikelihood(...) as ..."
        return self

    def __exit__(self, exc_type, exc, tb):
        # Clean up temporary files when leaving the context manager
        self.close()
