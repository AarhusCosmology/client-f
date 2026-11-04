"""One network of a run trained again on its saved training set, on a GPU with TF32 (TensorFlow's default on Ampere
and newer GPUs), on a GPU in full float32, and on the CPU, from the same random weights, and compared at the points of
that network's boost, which it never saw: epochs, seconds and the error. A check of whether TF32's matrix products
cost a network accuracy, and of what a GPU gains in time.

The training is the run's for network N: data_it_N.csv, the run's model and training settings, its training domain
(acquisition.boost.train_domain, as saved in RUN/boost) and the quadratic background of network N - 1 when it has
one. Every arm runs in a process of its own, since TensorFlow's devices and TF32 are fixed per process.

    python benchmarking/gpu_training.py RUN --network N [--arms gpu-tf32,gpu-fp32,cpu] [--seed 1] [--threads 8] [--epochs E]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gpu-tf32", "gpu-fp32", "cpu")


def arm(run, n, name, seed, threads, epochs=None):
    """One training, in this process."""
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    import yaml

    from utils.devices import set_up_tensorflow

    device = set_up_tensorflow("cpu" if name == "cpu" else "gpu", threads)
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(name == "gpu-tf32")
    from model.network import build_model, load_model, model_background
    from training.audit import _predict
    from training.domain import saved_domain
    from training.losses import build_loss
    from training.training import train_model

    # The settings the training needs, read as written: a run made by another branch of CLiENT may
    # carry keys this one's parser no longer knows.
    raw = yaml.safe_load((run / "client.yaml").read_text())
    t, m, boost = raw["training"], raw["model"], (raw["acquisition"].get("boost") or {})
    data = pd.read_csv(run / "training_data" / f"data_it_{n}.csv")
    names = [c for c in data.columns if c != "loglkl" and not c.startswith("derived")]
    data = data[np.isfinite(data.loglkl)]
    inputs, targets = data[names].to_numpy(np.float32), data[["loglkl"]].to_numpy(np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(inputs))
    inputs, targets = inputs[order], targets[order]
    if boost.get("train_domain") is not None:
        path, domain = saved_domain(run / "boost", n)
        if domain is not None:
            keep = domain.kept(inputs, float(boost["train_domain"]), path.name)
            inputs, targets = inputs[keep], targets[keep]
    previous = run / "trained_models" / f"model_it_{n - 1}.keras"
    try:
        background = model_background(load_model(previous)) if previous.exists() else None
    except Exception as error:                   # a network this CLiENT cannot load: none, and said so
        print(f"no background: {previous.name} does not load here ({type(error).__name__}: {error})", flush=True)
        background = None

    tf.keras.utils.set_random_seed(seed)
    model = build_model(inputs=inputs, targets=targets, n_layers=int(m["n_layers"]), n_neurons=int(m["n_neurons"]),
                        activation=m["activation"], **({} if background is None else {"background": background}))
    loss = build_loss(name=t["loss"], sigma_level=float(t["sigma_level"]), chi2_dof=len(names),
                      max_loglkl=float(np.max(targets)), targets=targets, ess_floor=float(t.get("msre_ess_floor", 0.1)))
    started = time.monotonic()
    history, metrics = train_model(model=model, inputs=inputs, targets=targets, loss=loss, restore_best=True,
                                   learning_rate=float(t["learning_rate"]), n_epochs=int(epochs or t["n_epochs"]),
                                   batch_size=int(t["batch_size"]), validation_split=float(t["validation_split"]),
                                   patience=int(t["patience"]), lr_schedule=t.get("lr_schedule", "none"),
                                   lr_factor=float(t.get("lr_factor", 0.3)), lr_patience=int(t.get("lr_patience", 75)),
                                   lr_min=float(t.get("lr_min", 1e-6)), lr_grace=int(t.get("lr_grace", 100)))
    seconds = time.monotonic() - started

    points = pd.read_csv(run / "boost" / f"boost_it_{n}.csv")
    points = points[np.isfinite(points.loglkl)]
    refused = run / "refused_value.json"
    if refused.exists():
        points = points[points.loglkl != json.loads(refused.read_text())["value"]]
    error = _predict(model, points[names].to_numpy(np.float32)) - points.loglkl.to_numpy(float)
    return dict(arm=name, device=device, tf32=name == "gpu-tf32", background=background is not None,
                seconds=round(seconds, 1),
                epochs=len(history.history["loss"]), best_epoch=int(metrics["epoch"]), val_loss=float(metrics["val_loss"]),
                training_points=len(inputs), boost_points=len(points), mean=float(error.mean()), scatter=float(error.std()),
                rms=float(np.sqrt(np.mean(error ** 2))), largest=float(np.abs(error).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--network", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, help="fewer epochs than the run's, for a quick check of the script only")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print("RESULT " + json.dumps(arm(args.run.resolve(), args.network, args.one, args.seed, args.threads, args.epochs)), flush=True)
        return
    rows = []
    for name in args.arms.split(","):
        if name not in ARMS:
            raise SystemExit(f"arm {name!r}: one of {', '.join(ARMS)}")
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", **({"CUDA_VISIBLE_DEVICES": ""} if name == "cpu" else {}))
        done = subprocess.run([sys.executable, __file__, str(args.run), "--network", str(args.network), "--one", name,
                               "--seed", str(args.seed), "--threads", str(args.threads),
                               *(["--epochs", str(args.epochs)] if args.epochs else [])],
                              text=True, capture_output=True, env=env)
        line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
        if done.returncode or line is None:
            print(f"{name}: failed (exit {done.returncode})\n{(done.stdout + done.stderr)[-3000:]}", flush=True)
            continue
        rows.append(json.loads(line[7:]))
        r = rows[-1]
        print(f"{name:9s} {r['device']}: {r['seconds']:7.1f} s, {r['epochs']} epochs (best {r['best_epoch']}), val loss "
              f"{r['val_loss']:.3e}; at {r['boost_points']} boost points mean {r['mean']:+.3f}, scatter {r['scatter']:.3f}, "
              f"largest {r['largest']:.2f}", flush=True)
    print(json.dumps(dict(run=str(args.run), network=args.network, seed=args.seed, arms=rows)))


if __name__ == "__main__":
    main()
