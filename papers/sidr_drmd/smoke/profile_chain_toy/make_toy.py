"""The toy of smoke/profile_chain.py's test (run.sh): writes into a scratch kit T the Cobaya file (two Gaussian branches,
the second 1.15 lower in ln L, a-b correlation 0.9, one nuisance parameter with a Gaussian prior), the specs and the
exact profiles.

    python make_toy.py T
"""
import json, sys
from pathlib import Path
import numpy as np
T = Path(sys.argv[1])
names = ["a", "b", "c", "d"]
mu1 = np.array([0.3, 1.0, -0.5, 2.0])
S1 = np.array([[0.04, 0.036, 0.0, 0.01], [0.036, 0.04, 0.0, 0.009], [0.0, 0.0, 0.25, 0.0], [0.01, 0.009, 0.0, 0.09]])  # a-b correlation 0.9
mu2 = np.array([-0.3, 0.2, 0.5, 1.6])
S2 = np.diag([0.03, 0.05, 0.2, 0.1])
off2 = 1.15                                    # the second branch's peak, below the first in ln L
def qform(mu, S):
    P = np.linalg.inv(S)
    terms = [f"({float(P[i,j])!r})*({names[i]}-({float(mu[i])!r}))*({names[j]}-({float(mu[j])!r}))" for i in range(4) for j in range(4)]
    return "+".join(terms)
lam = (f"lambda a, b, c, d, n1: max(-0.5*({qform(mu1, S1)}), -0.5*({qform(mu2, S2)})-{off2}) "
       f"- 0.5*(n1-0.3)**2/0.25")
box = {"a": (-1, 1.2), "b": (-1, 2.5), "c": (-2, 2), "d": (0.5, 3.5)}
yaml = "likelihood:\n  toy:\n    external: \"" + lam + "\"\nparams:\n"
for n in names:
    yaml += f"  {n}:\n    prior: {{min: {box[n][0]}, max: {box[n][1]}}}\n"
yaml += "  n1:\n    prior: {dist: norm, loc: 0.0, scale: 1.0}\n"
(T / "toy" / "cobaya_toy.yaml").write_text(yaml)
(T / "toy" / "job.env").write_text("# toy: nothing to set\n")
nodes_up = [round(x, 4) for x in np.arange(0.35, 0.95, 0.1)]
nodes_dn = [round(x, 4) for x in np.arange(0.25, -0.75, -0.1)]
seed1 = dict(a=0.32, b=1.05, c=-0.4, d=2.1, n1=0.0)
seed2 = dict(a=-0.25, b=0.3, c=0.45, d=1.5, n1=0.0)
chains = [dict(label="main_up", profile="a", seed=seed1, nodes=nodes_up, note="near branch 1"),
          dict(label="main_down", profile="a", seed=seed1, nodes=nodes_dn, note="near branch 1"),
          dict(label="second_down", profile="a", seed=seed2, nodes=[round(x, 4) for x in np.arange(-0.25, -0.75, -0.1)], note="near branch 2"),
          dict(label="second_up", profile="a", seed=seed2, nodes=[round(x, 4) for x in np.arange(-0.15, 0.95, 0.1)], note="near branch 2"),
          dict(label="main_c", profile="c", seed=seed1, nodes=[round(x, 4) for x in np.arange(-0.3, 1.0, 0.2)], note="c, branch 1")]
spec = dict(made="make_toy.py", cobaya="toy/cobaya_toy.yaml", job_env="toy/job.env", fixed={}, nuisance=["n1"],
            covariance=dict(names=names, matrix=(S1 * 1.3).tolist()), chains=chains)
(T / "toy" / "spec.json").write_text(json.dumps(spec, indent=1))
edge = dict(spec, chains=[dict(label="broken", profile="zzz", seed=seed1, nodes=[0.1]),
                          dict(label="outside", profile="a", seed=seed1, nodes=[5.0, 0.4]),
                          dict(label="fine", profile="a", seed=seed1, nodes=[0.35, 0.45])])
(T / "toy" / "spec_edge.json").write_text(json.dumps(edge, indent=1))
bad = S1 * 1.3
bad[0, 1] = bad[1, 0] = -bad[0, 1]                # the a-b correlation flipped: the shift along it goes the wrong way
(T / "toy" / "spec_badcov.json").write_text(json.dumps(dict(spec, covariance=dict(names=names, matrix=bad.tolist())), indent=1))
# exact profiles (ln L, nuisance at its optimum, the Gaussian prior on n1 included) per branch
def prof(mu, S, k, x, off):
    return -0.5 * (x - mu[k]) ** 2 / S[k, k] - off
best_n1 = 0.3 / (1 + 0.25)                     # max of -0.5(n1-0.3)^2/0.25 - 0.5 n1^2
nuis = -0.5 * (best_n1 - 0.3) ** 2 / 0.25 - 0.5 * best_n1 ** 2 - 0.5 * np.log(2 * np.pi)
exact = {}
for k, n in enumerate(names):
    exact[n] = dict(mu1=float(mu1[k]), var1=float(S1[k, k]), mu2=float(mu2[k]), var2=float(S2[k, k]), off2=off2, nuis=float(nuis))
(T / "toy" / "exact.json").write_text(json.dumps(exact, indent=1))
print("toy written; uniform prior constants:", -sum(np.log(box[n][1] - box[n][0]) for n in names))
