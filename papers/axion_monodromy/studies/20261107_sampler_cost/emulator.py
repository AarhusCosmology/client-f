"""The PDP+ combined emulator of the seven parts (omega <= 200), inside its region (the wall outside), as a likelihood
for standard samplers, one point per call, with a count of the calls. Network only: no CLASS++.

The target is that of the emulator's own posterior and evidence (BEST): the network restricted to the combined region
(model.network.restrict_model), over the training box (metadata.json), at double precision.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
CLIENT = os.environ.get("CLIENT_REPO") or str(Path(__file__).resolve().parents[4])   # this branch of client-f
sys.path.insert(0, CLIENT)
import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from model.network import load_model, restrict_model  # noqa: E402
from sampling.region import Region  # noqa: E402

RUN = Path(os.environ.get("EMULATOR_RUN") or Path(os.environ.get("CLIENT_PAPER_KIT", Path(__file__).resolve().parents[2]))
           / "results/20261107_084147_axion_pp_combined200")      # the combined emulator of PDP+ (README.md)
META = json.loads((RUN / "metadata.json").read_text())
NAMES = [p["name"] for p in META["parameters"]]
BOX = np.array([[p["lower"], p["upper"]] for p in META["parameters"]], float)
MODEL = restrict_model(load_model(RUN / "trained_models/model_it_0.keras"), Region.load(RUN / "boost/region_it_0.npz"))
_f = tf.function(lambda x: MODEL(x, training=False)[:, 0], input_signature=[tf.TensorSpec([None, len(NAMES)], tf.float64)])
CALLS = 0


def logp(x):
    """The emulator at one point (physical parameters in NAMES order)."""
    global CALLS
    CALLS += 1
    return float(_f(tf.constant(np.asarray(x, float).reshape(1, -1)))[0])


def logp_many(X):
    return _f(tf.constant(np.asarray(X, float))).numpy()


def posterior_samples():
    """The emulator's own posterior chain (the kit's report: each cell on its own, weighted by its evidence)."""
    z = np.load(RUN / "analysis/posterior_it_0.npz", allow_pickle=True)
    assert [str(n) for n in z["names"]] == NAMES
    return z["x"]
