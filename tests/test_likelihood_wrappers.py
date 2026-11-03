"""The shared bounds of a likelihood wrapper, and what the Cobaya wrapper returns."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.base import BaseLikelihood, ParameterInfo
from likelihood.cobaya import CobayaLikelihood


class Toy(BaseLikelihood):
    def __init__(self, *params):
        self.given = params
        super().__init__()

    def _extract_params(self):
        return list(self.given)

    def loglkl(self, x):
        return 0.0


class RestrictPriorBounds(unittest.TestCase):
    def test_a_parameter_with_a_center_and_sigma_is_restricted_inside_its_prior(self):
        toy = Toy(ParameterInfo("a", "a", lower=0.0, upper=10.0, center=5.0, sigma=1.0),
                  ParameterInfo("b", "b", lower=4.0, upper=10.0, center=5.0, sigma=1.0))
        toy.restrict_prior_bounds(3.0)
        self.assertEqual(toy.prior_bounds, dict(a=(2.0, 8.0), b=(4.0, 8.0)))

    def test_a_parameter_without_them_keeps_its_finite_prior(self):
        for center, sigma in ((None, None), (5.0, None), (None, 1.0)):
            toy = Toy(ParameterInfo("a", "a", lower=0.0, upper=10.0, center=center, sigma=sigma))
            toy.restrict_prior_bounds(3.0)
            self.assertEqual(toy.prior_bounds, dict(a=(0.0, 10.0)))

    def test_without_them_and_without_a_finite_prior_the_error_names_the_parameter(self):
        toy = Toy(ParameterInfo("a", "a", lower=0.0))
        with self.assertRaisesRegex(ValueError, "Parameter a has infinite prior bounds"):
            toy.restrict_prior_bounds(3.0)


def cobaya_with(logpost, derived=()):
    likelihood = CobayaLikelihood.__new__(CobayaLikelihood)
    likelihood._params = [ParameterInfo("a", "a")]
    likelihood._derived = []
    likelihood._n_reported = 0
    likelihood._derived_columns = []
    result = SimpleNamespace(logpost=logpost, derived=list(derived))
    likelihood.cobaya_model = SimpleNamespace(logposterior=lambda position, return_derived: result)
    return likelihood


class CobayaValue(unittest.TestCase):
    def test_the_profile_backend_has_the_same_guard(self):
        import profiles.evaluation as evaluation

        for logpost, expected in ((None, -np.inf), (-12.5, -12.5)):
            backend = SimpleNamespace(cobaya_model=SimpleNamespace(
                logposterior=lambda position, return_derived, cached, value=logpost: SimpleNamespace(logpost=value)))
            with mock.patch.object(evaluation, "_BACKEND", backend), mock.patch.object(evaluation, "_NAMES", ["a"]):
                self.assertEqual(evaluation.evaluate_backend([0.0]), expected)

    def test_a_result_without_a_log_posterior_is_minus_infinity(self):
        likelihood = cobaya_with(None)
        self.assertEqual(likelihood.loglkl([0.0]), -np.inf)
        self.assertEqual(likelihood.loglkl_and_derived([0.0])[0], -np.inf)

    def test_a_value_comes_back_as_it_is(self):
        likelihood = cobaya_with(-12.5)
        self.assertEqual(likelihood.loglkl([0.0]), -12.5)
        self.assertEqual(likelihood.loglkl_and_derived([0.0])[0], -12.5)


if __name__ == "__main__":
    unittest.main()
