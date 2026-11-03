"""The whole workflow on an analytic likelihood, through the command line. Run by itself:

    python tests/release_smoke.py

A 16-parameter banana in miniature, with the settings recommended for likelihoods known to be
non-Gaussian (docs/boost.md): the audit's chain search and posterior pilot, the density filter,
the quadratic background, the region with its quantile reach, the training domain, modes by
measured mixing. Four tempered networks, two boosts, then the evidence (needs BEST, skipped
without it), the modes, a small true profile and an export. It checks that the commands work
and what they write, nothing about accuracy. About three minutes on one core.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = dict(
    likelihood=dict(wrapper="cobaya", input="input/cobaya/banana_lcdm16.yaml"),
    prior=dict(n_samples=400, sampling_strategy="lhs", n_sigma=10.0, n_inject=20),
    acquisition=dict(
        n_append=200, pool_factor=5, n_neighbors=10, target_temperature=7.0, n_inject=20,
        audit=dict(chain=True, tau=5, posterior=20, starts=64, steps=100, rounds=2),
        boost=dict(n=200, start_iteration=3, boosts=2, density_filter=True, pool_factor=5, background="quadratic",
                   region=1.5, region_quantile=0.999, train_domain=1.25, mixing=0.1, min_share=0.005,
                   mode_depth="auto", walkers=64, cool_steps=100, steps=400)),
    model=dict(n_layers=2, n_neurons=64, activation="alsing"),
    training=dict(n_epochs=150, batch_size=64, loss="msre", sigma_level=3, msre_ess_floor=0.1, learning_rate=0.001,
                  validation_split=0.1, patience=20, lr_schedule="plateau", lr_factor=0.3, lr_patience=20, lr_min=1.0e-5,
                  lr_grace=20, warm_start=True),
    sampling=dict(sampler="aies", temperature=7.0, n_walkers=64, n_steps=800, thin=2, burn_in=100, chunk_size=200),
    convergence=dict(metric="tempered_batch_scatter", threshold=1.2, consecutive=2, max_iterations=6),
    seed=42)
PROFILE = dict(parameters=["omega_b", "n_s"], profile_points=5, sigma_range=2.0, workers=2, maxfun=40,
               total_call_budget=2000, audit_points=2, scan_points=11, hybrid_walkers=4)


def step(name, *arguments):
    started = time.time()
    result = subprocess.run([sys.executable, "client.py", *map(str, arguments)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"{name}: exit {result.returncode}, {time.time() - started:.0f} s", flush=True)
    if result.returncode:
        raise SystemExit(result.stdout[-6000:])


def has_best():
    from evidence.sampler import import_best

    try:
        import_best()
    except Exception:
        return False
    return True


with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    (d / "run.yaml").write_text(yaml.safe_dump(RUN))
    (d / "profile.yaml").write_text(yaml.safe_dump(PROFILE))
    step("run", d / "run.yaml", "-n", "smoke", "-o", d / "results")
    (run,) = (d / "results").iterdir()
    last = 5
    for name in [f"trained_models/model_it_{last}.keras", "boost/boost_it_3.json", "boost/boost_it_4.json",
                 "boost/region_it_4.npz", "boost/train_domain_it_4.npz", "audit/audit_it_3.json", "metrics.json",
                 "objective_identity.json"]:
        assert (run / name).exists(), f"the run did not write {name}"
    if has_best():
        step("evidence", "evidence", run, "-i", last, "--n-live", 256, "--tolerance", 0.01, "--seed", 1)
        assert list((run / "evidence").glob("*.json")), "no evidence written"
    else:
        print("evidence: skipped, BEST is not available", flush=True)
    step("modes", "modes", run, "-i", last)
    assert list((run / "modes").iterdir()), "no modes written"
    step("profiles", "profiles", run, "--config", d / "profile.yaml", "--iteration", last, "-o", d / "profile")
    assert (d / "profile" / "manifest.json").exists(), "no profile study written"
    step("export", "export", run, d / "exported")
    assert (d / "exported" / "saved_model").is_dir(), "no export written"
print("release smoke: passed")
