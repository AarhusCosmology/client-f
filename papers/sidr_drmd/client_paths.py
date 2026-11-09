"""CLiENT and its generic cluster tools, for this folder's programs.

The launcher, the training box, refused points, the build check, the true check, the report,
branching and the true-profile launcher are CLiENT's hpc/, in the repository this folder lies
in (a branch of client-f). A program here imports this module after putting this folder on
sys.path, and then finds them:

    sys.path.insert(0, str(HERE))
    import client_paths  # noqa: E402,F401

CLiENT is CLIENT_REPO when set, otherwise the repository root two levels up.
"""
import os
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
CLIENT_REPO = Path(os.environ.get("CLIENT_REPO") or KIT.parents[1])
HPC = CLIENT_REPO / "hpc"

# Right after this folder, ahead of site-packages: an installed module of the same name
# (report, status) must not stand in for CLiENT's.
at = sys.path.index(str(KIT)) + 1 if str(KIT) in sys.path else 0
for folder in (HPC / "true_profiles", HPC / "analysis", HPC, CLIENT_REPO):
    if str(folder) not in sys.path:
        sys.path.insert(at, str(folder))
