"""utils/devices.py: which ranks keep the GPUs, which GPU a process takes, and TensorFlow's set-up
on a machine without a GPU (the CI's and the laptop's). The GPU path itself needs a GPU node:
tests/gpu_members_smoke.py."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import devices  # noqa: E402


def config(members=12, member_device="auto", boosts=True):
    return SimpleNamespace(training=SimpleNamespace(members=members, member_device=member_device),
                           acquisition=SimpleNamespace(boost=object() if boosts else None))


class WhoKeepsTheGpus(unittest.TestCase):
    def test_rank_zero_and_the_member_ranks(self):
        from training.members import member_ranks

        ranks = member_ranks(96, 12)
        keep = [r for r in range(96) if devices.keeps_gpus(r, 96, 12, "auto", True)]
        self.assertEqual(keep, [0] + ranks)

    def test_no_member_ranks_without_members_boosts_or_with_cpu(self):
        for members, device, boosts in ((1, "auto", True), (12, "cpu", True), (12, "auto", False)):
            keep = [r for r in range(48) if devices.keeps_gpus(r, 48, members, device, boosts)]
            self.assertEqual(keep, [0], (members, device, boosts))

    def test_the_gpus_of_a_node_in_turn(self):
        self.assertEqual([devices.gpu_index(r, 2) for r in range(5)], [0, 1, 0, 1, 0])
        self.assertEqual({devices.gpu_index(r, 1) for r in range(48)}, {0})
        self.assertEqual(devices.node_rank(None), 0)

    def test_an_evaluation_rank_hides_the_gpus(self):
        saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            devices.set_up_devices(config(), None, 3)          # rank 3 of 1: not a member rank
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            devices.set_up_devices(config(), None, 0)          # rank 0 keeps them
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


TENSORFLOW = """import sys
sys.path.insert(0, {root!r})
from utils import devices
{body}
"""


def in_a_new_process(body):
    """TensorFlow's devices can be set once per process: each case runs in its own."""
    done = subprocess.run([sys.executable, "-c", TENSORFLOW.format(root=str(ROOT), body=body)], text=True,
                          capture_output=True, timeout=300,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3",
                                   PYTHONDONTWRITEBYTECODE="1"))
    return done.returncode, done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-2000:]


class TensorFlowWithoutAGpu(unittest.TestCase):
    def test_auto_takes_the_cpu_with_the_threads_asked_for(self):
        code, out = in_a_new_process("print(devices.set_up_tensorflow('auto', 2)); print(devices.set_up_tensorflow('gpu'))")
        self.assertEqual(code, 0, out)
        self.assertEqual(out, "CPU (2 threads)")       # the second call returns the first's answer

    def test_gpu_stops_without_one(self):
        code, out = in_a_new_process(
            "try:\n    devices.set_up_tensorflow('gpu')\nexcept RuntimeError as e:\n    print('refused:', e)")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("refused: training.member_device: gpu"), out)

    def test_after_tensorflow_has_started_it_reports_what_runs(self):
        code, out = in_a_new_process("import tensorflow as tf\ntf.constant(1.0) + 1\n"
                                     "print(devices.set_up_tensorflow('auto', 3))")
        self.assertEqual(code, 0, out)
        self.assertTrue(out.startswith("CPU"), out)

    def test_an_unknown_device_is_refused(self):
        with self.assertRaises(ValueError):
            devices.set_up_tensorflow("tpu")


if __name__ == "__main__":
    unittest.main()
