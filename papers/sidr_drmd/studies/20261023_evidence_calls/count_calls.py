"""How many network evaluations one BEST evidence computation of a CLiENT emulator takes (23 October 2026).

    OMP_NUM_THREADS=4 python results/20261023_evidence_calls/count_calls.py RUN [client.py evidence options ...]

Runs `python client.py evidence RUN ...` as it is, with the function BEST is given wrapped so that it counts its calls
(network calls on a batch of points) and the rows in them (points evaluated by the network). BEST's slice sampler is
vectorised: every step of the stepping-out and the shrinkage evaluates all chains updated together, also those that
have already found their new point, so the rows are the evaluations the network actually performed. The run stops
after the sampling, before the command writes anything into RUN; the counts, the dead points and the time are written
to calls_<run>_seed<seed>.json beside this script.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import evidence.__main__ as cli  # noqa: E402
import evidence.sampler as sampler  # noqa: E402  (the command imports run_nested from here when it runs)


class Sampled(Exception):
    pass


def main():
    argv = sys.argv[1:]
    run = Path(argv[0]).resolve()
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 42
    original = sampler.run_nested
    record = {}

    def counted_run_nested(log_prob, lower, upper, **kwargs):
        import tensorflow as tf

        rows = tf.Variable(0, dtype=tf.int64, trainable=False)
        calls = tf.Variable(0, dtype=tf.int64, trainable=False)

        def counted(x):
            rows.assign_add(tf.cast(tf.shape(x)[0], tf.int64))
            calls.assign_add(1)
            return log_prob(x)

        start = time.time()
        result = original(counted, lower, upper, **kwargs)
        record.update(run=run.name, seed=seed, n_live=kwargs.get("n_live"), n_live_updates=kwargs.get("n_live_updates"),
                      tolerance=kwargs.get("tolerance"), ndim=int(len(lower)), n_dead=int(result["n_dead"]),
                      iterations=int(result["n_dead"]) // int(kwargs.get("n_live_updates")),
                      network_evaluations=int(rows.numpy()), network_calls=int(calls.numpy()),
                      lnZ_box=result["lnZ_box"], sigma=result["sigma"], H=result["H"],
                      seconds=time.time() - start, device=result["device"])
        raise Sampled

    sampler.run_nested = counted_run_nested
    try:
        cli.main(argv)
    except Sampled:
        pass
    if not record:
        raise SystemExit("the evidence command stopped before it sampled")
    r = record
    r["evaluations_per_dead_point"] = r["network_evaluations"] / r["n_dead"]
    r["evaluations_per_call"] = r["network_evaluations"] / r["network_calls"]
    out = HERE / f"calls_{run.name}_seed{seed}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
