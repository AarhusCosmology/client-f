"""Which build of a Boltzmann code a job may evaluate with: launch.py and
true_profiles/cluster_backend.py ask this.

The environment's own classy and camb, unless the run's job.env sets CLASSPP_GUARD_DIR: then classy must be
the build in that folder's lib/ (patches/classpp_extrap_guard.patch; README.md). camb is always the environment's.
"""
import os


def location_problem(root, file, env_prefix=None, guard_dir=None):
    """Why Cobaya may not evaluate with module `root` ("classy" or "camb") imported from `file`, or None.
    env_prefix and guard_dir default to CLIENT_ENV and CLASSPP_GUARD_DIR."""
    env_prefix = os.environ.get("CLIENT_ENV") if env_prefix is None else env_prefix
    guard_dir = (os.environ.get("CLASSPP_GUARD_DIR") or "").strip() if guard_dir is None else guard_dir
    real = os.path.realpath(file)
    if root == "classy" and guard_dir:
        lib = os.path.realpath(os.path.join(guard_dir, "lib")) + os.sep
        if not real.startswith(lib):
            return f"Cobaya uses classy from {file}, not the patched build in {lib} that CLASSPP_GUARD_DIR asks for"
        return None
    if env_prefix and not real.startswith(os.path.realpath(env_prefix)):
        return f"Cobaya uses {root} from {file}, not from the environment {env_prefix}"
    return None
