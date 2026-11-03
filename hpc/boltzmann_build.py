"""Which build of a Boltzmann code a job may evaluate with: launch.py and
true_profiles/cluster_backend.py ask this.

The environment's own classy and camb, unless the run's job.env sets one of two variables:
- CLASSPP_GUARD_DIR: classy must be the patched build in that folder's lib/ (patches/classpp_extrap_guard.patch;
  README.md);
- CLASSPP_BUILD_DIR: classy must be the unpatched build of another CLASS++ commit in that folder's lib/, for a run
  that needs a commit other than the environment's.
Not both. camb is always the environment's. Only the path is checked; CLASSPP_BUILD_COMMIT, which names the commit
of a CLASSPP_BUILD_DIR, is recorded by launch.py (commits).
"""
import os
import subprocess


def commits(client_repo, environ=None):
    """The commits launch.py records in logs/backend_<job>.json. CLiENT's from its checkout (None when it is not
    one; " (DIRTY)" after it when tracked files differ). CLASS++'s from the job's settings: CLASSPP_BUILD_COMMIT
    with CLASSPP_BUILD_DIR, otherwise CLASSPP_COMMIT, which a site's environment sets for its own build (None when
    unset: nothing reads a commit out of a built classy)."""
    environ = os.environ if environ is None else environ
    client = None
    try:
        head = subprocess.run(["git", "-C", str(client_repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if head.returncode == 0:
            changed = subprocess.run(["git", "-C", str(client_repo), "status", "--porcelain", "--untracked-files=no"],
                                     capture_output=True, text=True, timeout=30)
            client = head.stdout.strip() + (" (DIRTY)" if changed.returncode == 0 and changed.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    build_dir = (environ.get("CLASSPP_BUILD_DIR") or "").strip()
    classpp = (environ.get("CLASSPP_BUILD_COMMIT") if build_dir else environ.get("CLASSPP_COMMIT")) or ""
    build = "CLASSPP_BUILD_DIR" if build_dir else "CLASSPP_GUARD_DIR" if (environ.get("CLASSPP_GUARD_DIR") or "").strip() else "environment"
    return dict(client_commit=client, classpp_commit=classpp.strip() or None, classpp_build=build)


def location_problem(root, file, env_prefix=None, guard_dir=None, build_dir=None):
    """Why Cobaya may not evaluate with module `root` ("classy" or "camb") imported from `file`, or None.
    env_prefix, guard_dir and build_dir default to CLIENT_ENV, CLASSPP_GUARD_DIR and CLASSPP_BUILD_DIR."""
    env_prefix = os.environ.get("CLIENT_ENV") if env_prefix is None else env_prefix
    guard_dir = (os.environ.get("CLASSPP_GUARD_DIR") or "").strip() if guard_dir is None else guard_dir
    build_dir = (os.environ.get("CLASSPP_BUILD_DIR") or "").strip() if build_dir is None else build_dir
    real = os.path.realpath(file)
    if root == "classy" and guard_dir and build_dir:
        return "CLASSPP_GUARD_DIR and CLASSPP_BUILD_DIR are both set: a run evaluates with one CLASS++ build"
    if root == "classy" and guard_dir:
        lib = os.path.realpath(os.path.join(guard_dir, "lib")) + os.sep
        if not real.startswith(lib):
            return f"Cobaya uses classy from {file}, not the patched build in {lib} that CLASSPP_GUARD_DIR asks for"
        return None
    if root == "classy" and build_dir:
        lib = os.path.realpath(os.path.join(build_dir, "lib")) + os.sep
        if not real.startswith(lib):
            return f"Cobaya uses classy from {file}, not the CLASS++ build in {lib} that CLASSPP_BUILD_DIR asks for"
        return None
    if env_prefix and not real.startswith(os.path.realpath(env_prefix)):
        return f"Cobaya uses {root} from {file}, not from the environment {env_prefix}"
    return None
