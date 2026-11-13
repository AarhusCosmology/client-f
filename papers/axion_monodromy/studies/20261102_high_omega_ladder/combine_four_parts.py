"""A four-part client.py combine on the analytic runs of tests/test_combine.py: each network exactly in its own range
(inside its region), the smootherstep blend in each band, and the record."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_combine import make_run, values  # noqa: E402
from run.combine import Part, combine  # noqa: E402
from model.network import load_model  # noqa: E402
from sampling.region import Region  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    boxes = [(1.0, 3.1), (2.5, 5.2), (4.5, 7.2), (6.5, 10.0)]
    bands = [(2.6, 3.0), (4.6, 5.0), (6.6, 7.0)]
    runs = [make_run(root, f"part{k}", b, seed=k + 1) for k, b in enumerate(boxes)]
    out = root / "combined"
    combine(out, [Part(r) for r in runs], "w", bands, log=print)
    c = load_model(out / "trained_models/model_it_0.keras")
    nets = [load_model(r / "trained_models/model_it_3.keras") for r in runs]
    regions = [Region.load(r / "boost/region_it_3.npz") for r in runs]
    x = np.random.default_rng(9).uniform([-2, 1, -2], [2, 10, 2], (20000, 3)).astype(np.float32)
    w, fc = x[:, 1], values(c, x)
    f = [values(n, x) for n in nets]
    inside = [r.rho(x) <= r.inflate for r in regions]
    edges = [1.0] + [b for band in bands for b in band] + [10.0]
    for k in range(4):
        lo = 1.0 if k == 0 else bands[k - 1][1]
        hi = 10.0 if k == 3 else bands[k][0]
        m = (w >= lo) & (w <= hi) & inside[k]
        assert m.sum() > 50, (k, m.sum())
        np.testing.assert_array_equal(fc[m], f[k][m])
        print(f"part {k}: w in [{lo}, {hi}]: {m.sum()} points, exactly its own network")
    for j, (lo, hi) in enumerate(bands):
        m = (w > lo) & (w < hi) & inside[j] & inside[j + 1]
        t = (w[m] - lo) / (hi - lo); s = t ** 3 * (t * (6 * t - 15) + 10)
        np.testing.assert_allclose(fc[m], (1 - s) * f[j][m] + s * f[j + 1][m], rtol=1e-5, atol=1e-4)
        print(f"band {j}: [{lo}, {hi}]: {m.sum()} points, the blend of parts {j} and {j + 1}")
    rec = json.loads((out / "combine.json").read_text())
    assert rec["bands"] == [list(b) for b in bands] and rec["box"] == [1.0, 10.0], rec
    print("combine.json: bands", rec["bands"], "box", rec["box"], "parts", len(rec["parts"]))
print("four-part combine: OK")
