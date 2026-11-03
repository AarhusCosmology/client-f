"""Derived parameters recorded alongside every true evaluation.

Analytic only: Cobaya with lambda likelihoods, and a stand-in for MontePython's
compute_lkl, so neither CLASS nor Planck data is needed.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from cobaya.theory import Theory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.dataset import TrainingDataset
from derived.metadata import DerivedMetadata
from likelihood.base import BaseLikelihood, ParameterInfo
from likelihood.cobaya import CobayaLikelihood
from likelihood.montepython import MontePythonLikelihood
from likelihood.surrogate import SurrogateMetadata
from utils.mpi_utils import broadcast_and_evaluate

# One derived parameter defined in the YAML (h2, a function of the inputs) and one
# reported by the likelihood itself through output_params (r2), so both of Cobaya's
# routes are covered. The prior on x_0 is the one a point is taken outside of. The
# likelihood has a type, as the Planck ones do, so Cobaya also reports a chi2__CMB
# total, which the wrapper must leave out.
COBAYA_INFO = dict(
    likelihood=dict(
        toy=dict(
            external="lambda x_0, x_1: (-0.5 * (x_0**2 + x_1**2), "
                     "{'r2': x_0**2 + x_1**2})",
            output_params=["r2"],
            type="CMB",
        )
    ),
    params=dict(
        x_0=dict(prior=dict(min=-2.0, max=2.0), ref=0.0, proposal=0.1, latex=r"x_0"),
        x_1=dict(prior=dict(min=-2.0, max=2.0), ref=0.0, proposal=0.1),
        h2=dict(derived="lambda x_0, x_1: (x_0 + 3.0) ** 2 * x_1", latex=r"$h^2$"),
        r2=dict(latex="r^2"),
    ),
)


def cobaya_likelihood(directory):
    path = Path(directory) / "toy.yaml"
    path.write_text(yaml.safe_dump(COBAYA_INFO, sort_keys=False))
    return CobayaLikelihood(str(path))


class NoDerived(BaseLikelihood):
    def _extract_params(self):
        return [ParameterInfo(name="x", label="x", lower=-1.0, upper=1.0)]

    def loglkl(self, x):
        return -0.5 * float(x[0]) ** 2


class CobayaDerivedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.likelihood = cobaya_likelihood(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_and_labels_follow_cobaya(self):
        reported = list(self.likelihood.cobaya_model.parameterization.derived_params())
        self.assertIn("chi2__CMB", reported)
        self.assertEqual(self.likelihood.param_names, ["x_0", "x_1"])
        self.assertEqual(self.likelihood.derived_names, ["h2", "r2"])
        self.assertEqual(self.likelihood.derived_labels, ["h^2", "r^2"])
        self.assertEqual(self.likelihood.n_derived, 2)

    def test_values_inside_the_prior(self):
        x = np.array([0.5, -1.0])
        loglkl, derived = self.likelihood.loglkl_and_derived(x)
        self.assertAlmostEqual(loglkl, self.likelihood.loglkl(x))
        np.testing.assert_allclose(derived, [3.5**2 * -1.0, 1.25])

    def test_nan_outside_the_prior(self):
        # Cobaya reports an empty list here, not NaNs; the wrapper must pad it.
        loglkl, derived = self.likelihood.loglkl_and_derived(np.array([3.0, 0.0]))
        self.assertEqual(loglkl, -np.inf)
        self.assertEqual(derived.shape, (2,))
        self.assertTrue(np.isnan(derived).all())

    def test_row_round_trip(self):
        rows = np.array([self.likelihood.evaluate_with_derived(x)
                         for x in ([0.5, -1.0], [3.0, 0.0])])
        targets, derived = self.likelihood.split_evaluations(rows)
        self.assertEqual(targets.shape, (2,))
        self.assertEqual(derived.shape, (2, 2))
        self.assertEqual(targets[1], -np.inf)
        # What an MPI worker gets back from broadcast_and_evaluate.
        targets, derived = self.likelihood.split_evaluations(np.array([]))
        self.assertEqual((targets.shape, derived.shape), ((0,), (0, 2)))

    def test_serial_broadcast_and_evaluate_carries_rows(self):
        points = np.array([[0.5, -1.0], [0.1, 0.2], [3.0, 0.0]])
        _, values = broadcast_and_evaluate(points, self.likelihood.evaluate_with_derived)
        self.assertEqual(values.shape, (3, 3))
        _, scalars = broadcast_and_evaluate(points, self.likelihood.loglkl)
        np.testing.assert_array_equal(values[:, 0], scalars)

    def test_derived_inputs_resolved_through_cobaya(self):
        # h2 is a lambda of x_0 and x_1; r2 comes from the likelihood, whose inputs
        # are x_0 and x_1. Nothing else exists here, so check the resolution on a
        # parameter that nothing derived depends on.
        info = yaml.safe_load(yaml.safe_dump(COBAYA_INFO))
        info["params"]["x_2"] = dict(prior=dict(min=-1.0, max=1.0), ref=0.0, proposal=0.1)
        info["likelihood"]["other"] = dict(external="lambda x_2: -0.5 * x_2**2")
        path = Path(self.tmp.name) / "three.yaml"
        path.write_text(yaml.safe_dump(info, sort_keys=False))
        likelihood = CobayaLikelihood(str(path))
        self.assertEqual(likelihood.param_names, ["x_0", "x_1", "x_2"])
        self.assertEqual(likelihood.derived_input_names, ["x_0", "x_1"])

    def test_derived_metadata_round_trip(self):
        metadata = DerivedMetadata.from_likelihood(self.likelihood)
        metadata.save(self.tmp.name)
        loaded = DerivedMetadata.load(self.tmp.name)
        self.assertEqual(loaded, metadata)
        self.assertEqual(loaded.names, ["h2", "r2"])
        self.assertEqual(loaded.labels, ["h^2", "r^2"])
        self.assertEqual(loaded.inputs, ("x_0", "x_1"))

    def test_change_of_derived_set_is_detected(self):
        # Continuing a run warns on this: with CAMB, declaring sigma8 changes the likelihood.
        from derived.metadata import changed_since_recorded

        self.assertIsNone(changed_since_recorded(self.tmp.name, ["h2", "r2"]))  # no record
        DerivedMetadata.from_likelihood(self.likelihood).save(self.tmp.name)
        self.assertIsNone(changed_since_recorded(self.tmp.name, ["h2", "r2"]))
        self.assertEqual(changed_since_recorded(self.tmp.name, ["h2", "sigma8"]),
                         (["sigma8"], ["r2"]))

    def test_metadata_json_is_unchanged(self):
        # metadata.json is fingerprinted by profile studies; derived parameters must
        # not appear in it.
        path = Path(self.tmp.name) / "metadata.json"
        SurrogateMetadata.from_likelihood(self.likelihood).save(path)
        self.assertEqual(list(json.loads(path.read_text())), ["parameters"])


class ToyTransfers(Theory):
    """Takes u and hands on a product, not a parameter, like CAMB's camb.transfers."""

    def get_can_support_params(self):
        return ["u"]

    def get_can_provide(self):
        return ["transfer"]

    def calculate(self, state, want_derived=True, **params):
        state["transfer"] = 2.0 * params["u"]

    def get_transfer(self):
        return self.current_state["transfer"]


class ToySpectra(Theory):
    """Takes v, requires the transfer product and outputs d, like CAMB's camb."""

    def get_can_support_params(self):
        return ["v"]

    def get_can_provide_params(self):
        return ["d"]

    def get_requirements(self):
        return {"transfer": None}

    def calculate(self, state, want_derived=True, **params):
        state["derived"] = {"d": self.provider.get_transfer() + params["v"]}


class CobayaComponentDependenceTests(unittest.TestCase):
    def test_outputs_depend_on_the_inputs_of_required_components(self):
        # d comes from ToySpectra, whose own input is only v; it requires ToyTransfers,
        # which takes u. Matching outputs to their component's inputs alone gave [v],
        # which on CAMB was [logA, ns, tau] for H0 and sigma8.
        info = dict(
            theory=dict(transfers=dict(external=ToyTransfers), spectra=dict(external=ToySpectra)),
            likelihood=dict(
                toy=dict(external="lambda _self: -0.5 * _self.provider.get_param('d') ** 2",
                         requires=dict(d=None)),
                other=dict(external="lambda w: -0.5 * w**2"),
            ),
            params=dict(
                u=dict(prior=dict(min=-1.0, max=1.0), ref=0.1, proposal=0.1),
                v=dict(prior=dict(min=-1.0, max=1.0), ref=0.1, proposal=0.1),
                w=dict(prior=dict(min=-1.0, max=1.0), ref=0.1, proposal=0.1),
                d=dict(latex="d"),
            ),
        )
        with mock.patch("cobaya.yaml.yaml_load_file", return_value=info):
            likelihood = CobayaLikelihood("in-memory")
        self.assertEqual(likelihood.param_names, ["u", "v", "w"])
        self.assertEqual(likelihood.derived_names, ["d"])
        self.assertEqual(likelihood._derived_dependencies(), ["u", "v"])
        loglkl, derived = likelihood.loglkl_and_derived(np.array([0.25, 0.1, 0.0]))
        np.testing.assert_allclose(derived, [0.6])
        # The log-posterior: three uniform priors of width 2.
        self.assertAlmostEqual(loglkl, -0.5 * 0.6**2 + 3 * np.log(0.5))


@unittest.skipUnless(importlib.util.find_spec("camb"), "CAMB is not installed")
class CobayaCambTests(unittest.TestCase):
    """Cobaya + CAMB with the planck_2018_camb parameterisation: theta_MC_100 sampled
    and dropped, cosmomc_theta a value lambda, H0 derived with bounds, CAMB's names.
    One cheap likelihood on H0, so only CAMB's background and sigma8 are computed."""

    @classmethod
    def setUpClass(cls):
        from cobaya.cosmo_input import create_input

        info = create_input(preset="planck_2018_camb")
        keep = ("logA", "As", "ns", "theta_MC_100", "cosmomc_theta", "H0", "ombh2", "omch2",
                "mnu", "tau", "omegam", "omegamh2", "sigma8", "zrei", "rdrag")
        params = {n: v for n, v in info["params"].items() if n in keep}
        params["thetastar"] = dict(latex=r"100\theta_*")
        params["S8"] = dict(derived="lambda sigma8, omegam: sigma8*np.sqrt(omegam/0.3)", latex="S_8")
        params["c"] = dict(prior=dict(min=-1.0, max=1.0), ref=0.0, proposal=0.1)
        cls.info = dict(
            theory=dict(camb=dict(extra_args=dict(theta_H0_range=[20, 100]))),
            likelihood=dict(
                h0=dict(external="lambda _self: -0.5 * ((_self.provider.get_param('H0') - 67.4) / 0.5) ** 2",
                        requires=dict(H0=None), type="H0"),
                other=dict(external="lambda c: -0.5 * c**2"),
            ),
            params=params,
        )
        with mock.patch("cobaya.yaml.yaml_load_file", return_value=cls.info):
            cls.likelihood = CobayaLikelihood("in-memory")
        cls.point = dict(logA=3.044, ns=0.9649, theta_MC_100=1.04092, ombh2=0.02237,
                         omch2=0.1200, tau=0.0544, c=0.0)

    def x(self, **changes):
        point = dict(self.point, **changes)
        return np.array([point[n] for n in self.likelihood.param_names])

    def test_names_labels_and_inputs(self):
        lk = self.likelihood
        self.assertIn("chi2__H0", lk.cobaya_model.parameterization.derived_params())
        self.assertNotIn("cosmomc_theta", lk.derived_names)
        self.assertEqual(set(lk.derived_names),
                         {"As", "H0", "omegam", "omegamh2", "sigma8", "zrei", "rdrag", "thetastar", "S8"})
        self.assertEqual(dict(zip(lk.derived_names, lk.derived_labels))["thetastar"], r"100\theta_*")
        self.assertEqual(set(lk.derived_input_names),
                         {"logA", "ns", "theta_MC_100", "ombh2", "omch2", "tau"})

    def test_values(self):
        loglkl, derived = self.likelihood.loglkl_and_derived(self.x())
        values = dict(zip(self.likelihood.derived_names, derived))
        self.assertAlmostEqual(loglkl, self.likelihood.loglkl(self.x()))
        self.assertAlmostEqual(values["H0"], 67.36, delta=0.3)
        self.assertAlmostEqual(values["As"], 1e-10 * np.exp(3.044))
        self.assertAlmostEqual(values["omegamh2"], values["omegam"] * (values["H0"] / 100) ** 2)
        self.assertAlmostEqual(values["S8"], values["sigma8"] * np.sqrt(values["omegam"] / 0.3))
        self.assertAlmostEqual(values["thetastar"], 1.0411, delta=0.001)

    def test_nan_on_camb_failure_and_outside_the_prior(self):
        # No H0 in theta_H0_range gives 100 theta_MC = 1.2; ombh2 = 0.2 is outside the prior.
        for x in (self.x(theta_MC_100=1.2), self.x(ombh2=0.2)):
            loglkl, derived = self.likelihood.loglkl_and_derived(x)
            self.assertEqual(loglkl, -np.inf)
            self.assertTrue(np.isnan(derived).all())


class NoDerivedTests(unittest.TestCase):
    def test_backend_without_derived(self):
        likelihood = NoDerived()
        self.assertEqual(likelihood.n_derived, 0)
        row = likelihood.evaluate_with_derived(np.array([0.5]))
        np.testing.assert_array_equal(row, [-0.125])
        targets, derived = likelihood.split_evaluations(np.array([row]))
        self.assertEqual(derived.shape, (1, 0))

    def test_all_inputs_by_default(self):
        self.assertEqual(NoDerived().derived_input_names, ["x"])


class MontePythonDerivedTests(unittest.TestCase):
    """The wrapper against stand-ins for MontePython's data object and compute_lkl.

    As in MontePython: parameters carry a role and a scale; compute_lkl writes the
    'derived' (CLASS) values divided by their scale and the 'derived_lkl' (likelihood)
    values unscaled, and only after a successful CLASS run -- on a failure it returns
    boundary_loglike early and leaves the previous point's values in place.
    """

    def setUp(self):
        self.parameters = {
            "omega_b": {"current": None, "scale": 0.01, "role": "cosmo"},
            "A_planck": {"current": None, "scale": 1.0, "role": "nuisance"},
            "H0": {"current": None, "scale": 1.0, "role": "derived"},
            "A_s": {"current": None, "scale": 1e-9, "role": "derived"},
            "M_lkl": {"current": None, "scale": 0.1, "role": "derived_lkl"},
        }

        def get_mcmc_parameters(roles):
            return [n for n, p in self.parameters.items() if p["role"] in roles]

        def compute_lkl(cosmo, data):
            omega_b = data.mcmc_parameters["omega_b"]["current"] * 0.01
            if omega_b < 0:
                return data.boundary_loglike
            data.mcmc_parameters["H0"]["current"] = 1000.0 * omega_b
            data.mcmc_parameters["A_s"]["current"] = omega_b * 1e-7
            data.mcmc_parameters["M_lkl"]["current"] = -19.0 - omega_b
            for name in get_mcmc_parameters(["derived"]):
                data.mcmc_parameters[name]["current"] /= data.mcmc_parameters[name]["scale"]
            return -0.5 * omega_b**2

        likelihood = MontePythonLikelihood.__new__(MontePythonLikelihood)
        likelihood.data = SimpleNamespace(
            mcmc_parameters=self.parameters,
            get_mcmc_parameters=get_mcmc_parameters,
            boundary_loglike=-1e30,
            lkl={},
            update_cosmo_arguments=lambda: None,
        )
        likelihood.cosmo = None
        likelihood.compute_lkl = compute_lkl
        likelihood._params = [ParameterInfo(name="omega_b", label="omega_b", scale=0.01),
                              ParameterInfo(name="A_planck", label="A_planck")]
        # io_mp.get_tex_name puts a scale into the label, as MontePython's own does.
        io_mp = SimpleNamespace(
            get_tex_name=lambda name, number=1: name if number == 1 else f"10^{{{np.log10(number):.0f}}}{name}"
        )
        with mock.patch.dict(sys.modules, {"io_mp": io_mp}):
            likelihood._derived = likelihood._extract_derived()
        self.likelihood = likelihood

    def test_roles_scales_and_labels(self):
        self.assertEqual(self.likelihood.derived_names, ["H0", "A_s", "M_lkl"])
        # derived_lkl values are never rescaled by MontePython, whatever their scale.
        self.assertEqual(self.likelihood.derived_scales, [1.0, 1e-9, 1.0])
        # Values are physical, so the labels carry no scale prefix.
        self.assertEqual(self.likelihood.derived_labels, ["H0", "A_s", "M_lkl"])

    def test_physical_units(self):
        loglkl, derived = self.likelihood.loglkl_and_derived(np.array([0.02, 1.0]))
        self.assertAlmostEqual(loglkl, -0.5 * 0.02**2)
        np.testing.assert_allclose(derived, [20.0, 2e-9, -19.02])

    def test_failed_point_does_not_inherit_previous_values(self):
        self.likelihood.loglkl_and_derived(np.array([0.02, 1.0]))
        loglkl, derived = self.likelihood.loglkl_and_derived(np.array([-0.02, 1.0]))
        self.assertEqual(loglkl, -np.inf)
        self.assertTrue(np.isnan(derived).all())

    def test_input_dependencies_follow_roles(self):
        # A likelihood-computed value may depend on nuisance parameters...
        self.assertEqual(self.likelihood.derived_input_names, ["omega_b", "A_planck"])
        # ...CLASS-derived ones only on the 'cosmo' arguments.
        self.parameters["M_lkl"]["role"] = "derived"
        self.assertEqual(self.likelihood.derived_input_names, ["omega_b"])


class TrainingDatasetDerivedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.likelihood = cobaya_likelihood(self.tmp.name)
        rng = np.random.default_rng(3)
        self.inputs = rng.uniform(-1.5, 1.5, (40, 2))
        rows = np.array([self.likelihood.evaluate_with_derived(x) for x in self.inputs])
        self.targets, self.derived = self.likelihood.split_evaluations(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, n=30):
        return TrainingDataset(self.inputs[:n], self.targets[:n], self.likelihood,
                               n_neighbors=5, target_temperature=1.0,
                               derived=self.derived[:n])

    def save(self, dataset, iteration):
        dataset.save(self.dir / "training_data" / f"data_it_{iteration}.csv")
        dataset.save_derived(self.dir / "derived_data" / f"derived_it_{iteration}.csv")

    def load(self, iteration, likelihood=None):
        return TrainingDataset.load(self.dir / "training_data", likelihood or self.likelihood,
                                    n_neighbors=5, target_temperature=1.0,
                                    iteration=iteration,
                                    derived_data_dir=self.dir / "derived_data")

    def test_training_csv_layout_is_unchanged(self):
        (self.dir / "training_data").mkdir()
        self.save(self.dataset(), 0)
        header = (self.dir / "training_data" / "data_it_0.csv").read_text().splitlines()[0]
        self.assertEqual(header, "x_0,x_1,loglkl")

    def test_round_trip_through_add_data(self):
        (self.dir / "training_data").mkdir()
        dataset = self.dataset()
        dataset.add_data(self.inputs[30:35], self.targets[30:35], self.derived[30:35])
        # A caller that has no derived values for its points gets NaN rows.
        dataset.add_data(self.inputs[35:], self.targets[35:])
        self.save(dataset, 1)
        loaded = self.load(1)
        np.testing.assert_allclose(loaded.derived[:35], self.derived[:35], rtol=1e-12)
        self.assertTrue(np.isnan(loaded.derived[35:]).all())

    def test_run_recorded_before_derived_parameters(self):
        (self.dir / "training_data").mkdir()
        self.dataset().save(self.dir / "training_data" / "data_it_0.csv")
        loaded = self.load(0)
        self.assertEqual(loaded.derived.shape, (30, 2))
        self.assertTrue(np.isnan(loaded.derived).all())

    def test_misaligned_file_is_ignored(self):
        (self.dir / "training_data").mkdir()
        dataset = self.dataset()
        self.save(dataset, 0)
        path = self.dir / "derived_data" / "derived_it_0.csv"
        df = pd.read_csv(path)
        df.iloc[::-1].to_csv(path, index=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = self.load(0)
        self.assertTrue(any("does not line up" in str(w.message) for w in caught))
        self.assertTrue(np.isnan(loaded.derived).all())

    def test_derived_parameter_added_since(self):
        (self.dir / "training_data").mkdir()
        self.save(self.dataset(), 0)
        path = self.dir / "derived_data" / "derived_it_0.csv"
        pd.read_csv(path).drop(columns=["r2"]).to_csv(path, index=False)
        loaded = self.load(0)
        np.testing.assert_allclose(loaded.derived[:, 0], self.derived[:30, 0], rtol=1e-12)
        self.assertTrue(np.isnan(loaded.derived[:, 1]).all())

    def test_nothing_written_without_derived(self):
        dataset = TrainingDataset(self.inputs[:, :1], self.targets, NoDerived(),
                                  n_neighbors=5, target_temperature=1.0)
        self.assertEqual(dataset.derived.shape, (40, 0))
        dataset.save_derived(self.dir / "derived_data" / "derived_it_0.csv")
        self.assertFalse((self.dir / "derived_data").exists())


if __name__ == "__main__":
    unittest.main()
