"""A small real CLI run with storage.training_sets: increments, continued, rewound and continued again.

Every set is read back through run/layout.py and must be what a whole file would hold: the
earlier set's rows first, unchanged. The run's index (run.json) must list every set.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import layout  # noqa: E402


def invoke(args, expected=0, timeout=900):
    result = subprocess.run([sys.executable, str(ROOT / "client.py"), *map(str, args)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if (result.returncode == 0) != (expected == 0):
        raise AssertionError(result.stdout[-5000:])
    return result.stdout


def check_sets(run):
    stored = layout.set_iterations(run)
    previous = None
    for n in sorted(stored):
        frame = layout.training_set(run, n)
        if previous is not None:
            assert len(frame) >= len(previous), (n, len(frame), len(previous))
            assert np.array_equal(frame.iloc[:len(previous)].to_numpy(), previous.to_numpy(), equal_nan=True), n
        assert len(frame) == layout.set_rows(run, n)
        previous = frame
    return stored


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    config = yaml.safe_load((ROOT / "docs/site/examples/gaussian_client.yaml").read_text())
    config["likelihood"]["input"] = str(ROOT / "docs/site/examples/gaussian_likelihood.yaml")
    config["training"]["n_epochs"] = 2
    config["acquisition"].update(n_append=10, pool_factor=2)
    config["storage"] = dict(training_sets="increments")
    cfg = d / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    invoke([cfg, "-i", 3, "-o", d / "out"])
    run = next((d / "out").iterdir())
    stored = check_sets(run)
    assert stored == {0: "whole", 1: "added", 2: "added"}, stored
    assert sorted(p.name for p in (run / "training_data").iterdir()) == ["added_it_1.csv", "added_it_2.csv", "data_it_0.csv"]
    index = json.loads((run / "run.json").read_text())
    assert [e["training_set"]["rows"] for e in index["iterations"] if "training_set" in e] == \
        [layout.set_rows(run, n) for n in range(3)], index["iterations"]

    # Continued: the loaded set is extended by an increment.
    text = invoke([run, "-i", 1])
    assert "from data_it_0.csv + added_it_1.csv + added_it_2.csv" in text, text[-3000:]
    stored = check_sets(run)
    assert stored == {0: "whole", 1: "added", 2: "added", 3: "added"}, stored

    # Rewound to network 1: the later increments are set aside, nothing deleted; continued again.
    invoke([run, "-s", 1, "--rewind", "-i", 1])
    stored = check_sets(run)
    assert stored == {0: "whole", 1: "added", 2: "added"}, stored
    aside = next(p for p in (run / "interrupted").iterdir() if p.name.startswith("rewind_to_it_1"))
    assert (aside / "training_data" / "added_it_3.csv").exists()

    # The index command prints and writes.
    assert "training set" in invoke(["index", run, "--write"])
    print("increments smoke passed:", {n: layout.set_rows(run, n) for n in sorted(stored)})
