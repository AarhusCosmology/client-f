"""The true stage's evaluation ledger (profiles.evaluation) and its curvature geometry (profiles.hybrid)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from profiles.evaluation import EvaluationLedger, CallBudgetExceeded
from profiles.hybrid import curvature_geometry


class TestTrueProfile(unittest.TestCase):
    def test_accounting_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def fun(x): calls.append(x.copy()); return -float(x@x)
            l=EvaluationLedger(fun,[[-2,2]],journal=Path(d)/'calls.jsonl',max_calls=2)
            l.import_value([0],0,source='old checkpoint diagnostics')
            self.assertEqual(l([0]),0)
            l([1]); l([1]); l([-1])
            self.assertEqual(l([5]),-np.inf)
            with self.assertRaises(CallBudgetExceeded): l([.5])
            self.assertEqual(len(calls),2)
            self.assertEqual(l.counts['backend_calls'],2)
            self.assertEqual(l.counts['cache_hits'],2)
            events=[json.loads(s) for s in l.journal.read_text().splitlines()]
            self.assertEqual(sum(e['event']=='start' for e in events),2)

    def test_failure_counted(self):
        with tempfile.TemporaryDirectory() as d:
            def fail(x): raise RuntimeError('backend failure')
            l=EvaluationLedger(fail,[[-1,1]],journal=Path(d)/'calls.jsonl',max_calls=2)
            with self.assertRaises(RuntimeError): l([0])
            self.assertEqual(l.counts['backend_calls'],1)
            self.assertEqual(l.counts['exceptions'],1)

    def test_flat_geometry_keeps_all_directions(self):
        matrix,info=curvature_geometry(np.diag([10.,0.,-1.]))
        self.assertEqual(np.linalg.matrix_rank(matrix),3)
        self.assertTrue(np.isfinite(matrix).all())


if __name__=='__main__': unittest.main()
