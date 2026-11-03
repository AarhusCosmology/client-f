"""The launcher's handling of points the likelihood refuses (hpc/refused.py): the
CLIENT_REFUSED_VALUE settings, the value read off a design, and the wrapped likelihood."""

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hpc"))

try:
    from cobaya.log import LoggedError
except ImportError:  # the wrapper is about Cobaya's own errors
    LoggedError = None

import refused  # noqa: E402
from likelihood.base import BaseLikelihood, ParameterInfo  # noqa: E402


def setting(text):
    with mock.patch.dict(os.environ, {} if text is None else {"CLIENT_REFUSED_VALUE": text}, clear=text is None):
        return refused.refused_setting()


class SettingTests(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(setting(None))
        self.assertIsNone(setting("  "))
        self.assertEqual(setting("-250000"), -250000.0)
        self.assertEqual(setting("design"), ("design", 0.5))
        self.assertEqual(setting("design:0.25"), ("design", 0.25))
        for bad in ("design:0", "design:1.5", "inf", "nan", "median"):
            with self.assertRaises(ValueError):
                setting(bad)
        with mock.patch.dict(os.environ, {"CLIENT_REFUSED_VALUE": "design"}):
            self.assertIsNone(refused.refused_value())  # not a number until a design has set it
        with mock.patch.dict(os.environ, {"CLIENT_REFUSED_VALUE": "-3e4"}):
            self.assertEqual(refused.refused_value(), -30000.0)

    def test_value_from_a_design(self):
        # The quantile of the finite values, rounded down to three significant digits: the two
        # boxes of the first real use had medians -238 197.7 and -30 648.07.
        self.assertEqual(refused.design_value([-30648.07, -20000.0, -113462.2, -np.inf, np.nan]), (-30700.0, -30648.07))
        self.assertEqual(refused.design_value([-238197.7] * 3)[0], -239000.0)
        self.assertEqual(refused.design_value([1234.5, 10.0, 5000.0])[0], 1230.0)  # down, also above zero
        values = np.linspace(-1000.0, 0.0, 101)
        self.assertEqual(refused.design_value(values, 0.25)[1], -750.0)
        self.assertLess(refused.design_value(values, 0.25)[0], refused.design_value(values, 0.5)[0])
        # never equal to a value the likelihood really gave: refused points are counted by it
        value, raw = refused.design_value([-30700.0, -1.0, -50000.0])
        self.assertEqual((value, raw), (-30800.0, -30700.0))
        with self.assertRaises(ValueError):
            refused.design_value([-np.inf, np.nan])

    def test_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "refused_value.json"
            self.assertIsNone(refused.read_record(path))
            refused.write_record(path, dict(value=-30700.0, setting="design"))
            self.assertEqual(refused.read_record(path), dict(value=-30700.0, setting="design"))
            self.assertEqual([p.name for p in Path(d).iterdir()], ["refused_value.json"])  # no temporary file left
            path.write_text("{")  # another rank still writing: not an error, just not there yet
            self.assertIsNone(refused.read_record(path))


class Fake(BaseLikelihood):
    """Finite below x0 = 0.6, a plain -inf (as a Boltzmann code's refusal) up to 0.8, and one of
    Cobaya's logged errors, which stop_at_error: false does not catch, above."""

    def _extract_params(self):
        return [ParameterInfo(name=f"p{i}", label=f"p{i}", scale=1.0, lower=0.0, upper=1.0, center=0.5, sigma=0.1)
                for i in range(2)]

    def _extract_derived(self):
        return [ParameterInfo(name="d0", label="d0")]

    def loglkl(self, x):
        return self.loglkl_and_derived(x)[0]

    def loglkl_and_derived(self, x):
        if x[0] > 0.8:
            raise LoggedError(QUIET, "Not possible to extrapolate to k=%g 1/Mpc", 3000 + 1000 * x[0])
        if x[0] > 0.6:
            return -np.inf, np.full(1, np.nan)
        if x[0] > 0.55:                 # a theory code's spectra not finite: Cobaya passes NaN on
            return np.nan, np.full(1, np.nan)
        return -float(np.sum((np.asarray(x) - 0.3) ** 2)), np.array([2.0 * x[0]])


FINITE, MINUS_INF, ERROR, NAN = [0.3, 0.3], [0.7, 0.3], [0.9, 0.3], [0.58, 0.3]
QUIET = logging.getLogger("refused-points-test")  # a logged error writes itself to its logger
QUIET.setLevel(logging.CRITICAL + 1)


@unittest.skipIf(LoggedError is None, "Cobaya is not installed")
class WrapperTests(unittest.TestCase):
    def wrapped(self, text, **arguments):
        with mock.patch.dict(os.environ, {} if text is None else {"CLIENT_REFUSED_VALUE": text}, clear=text is None):
            with mock.patch("builtins.print"):  # the first error of each kind is reported
                return refused.refuse_logged_errors(Fake(), **arguments)

    def values(self, truth):
        with mock.patch("builtins.print"):
            return [truth.loglkl(x) for x in (FINITE, MINUS_INF, ERROR)]

    def test_unset_a_refused_point_is_minus_infinity(self):
        truth = self.wrapped(None)
        self.assertEqual(self.values(truth), [0.0, -np.inf, -np.inf])  # the logged error no longer raises
        with mock.patch("builtins.print"):
            value, derived = truth.loglkl_and_derived(ERROR)
        self.assertTrue(np.isneginf(value) and np.isnan(derived).all() and derived.shape == (1,))
        self.assertEqual(sum(truth.refused_kinds.values()), 2)
        self.assertEqual((truth.refused_setting, truth.refused_value), (None, None))

    def test_a_nan_is_a_refused_point(self):
        # unset: -inf (left out of training, weight zero in a boost or a true check), counted once
        truth = self.wrapped(None)
        with mock.patch("builtins.print") as printed:
            self.assertTrue(np.isneginf(truth.loglkl(NAN)))
            value, derived = truth.loglkl_and_derived(NAN)
        self.assertTrue(np.isneginf(value) and np.isnan(derived).all())
        self.assertEqual(truth.refused_kinds["NaN"], 2)
        self.assertEqual(printed.call_count, 1)                          # the first one is reported
        self.assertEqual(truth.loglkl(FINITE), 0.0)                      # a real point is untouched
        # a value, and design before its value is set, as for any other refused point
        with mock.patch("builtins.print"):
            self.assertEqual(self.wrapped("-650").loglkl(NAN), -650.0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "refused_value.json"
            truth = self.wrapped("design", value_file=path)
            with mock.patch("builtins.print"):
                self.assertTrue(np.isneginf(truth.loglkl(NAN)))
                refused.write_record(path, dict(value=-214.0))
                self.assertEqual(truth.loglkl(NAN), -214.0)

    def test_a_number_replaces_every_refused_value(self):
        truth = self.wrapped("-650")
        self.assertEqual(self.values(truth), [0.0, -650.0, -650.0])
        with mock.patch("builtins.print"):
            self.assertEqual(truth.loglkl_and_derived(MINUS_INF)[0], -650.0)
            np.testing.assert_array_equal(truth.loglkl_and_derived(FINITE)[1], [0.6])  # a real point is untouched
        self.assertEqual(truth.refused_value, -650.0)

    def test_design_waits_for_the_value(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "refused_value.json"
            truth = self.wrapped("design", value_file=path)
            self.assertEqual((truth.refused_setting, truth.refused_value), (("design", 0.5), None))
            self.assertEqual(self.values(truth), [0.0, -np.inf, -np.inf])  # the design itself: nothing set yet
            refused.write_record(path, dict(value=-214.0))               # rank 0, after the design
            self.assertEqual(self.values(truth), [0.0, -214.0, -214.0])   # any rank, at its next refused point
            self.assertEqual(truth.refused_value, -214.0)
            path.unlink()
            self.assertEqual(self.values(truth), [0.0, -214.0, -214.0])   # read once, then kept
        # rank 0 sets its own copy directly, and a continued run sets it at start-up
        truth = self.wrapped("design")
        truth.set_refused_value(-30700)
        self.assertEqual(self.values(truth), [0.0, -30700.0, -30700.0])


if __name__ == "__main__":
    unittest.main()
