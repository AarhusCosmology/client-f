"""Building a likelihood on every MPI rank, a few ranks at a time.

Some likelihoods need far more memory while their data are read than afterwards: HiLLiPoP's
inverse covariance (6.4 GB on disk) takes a rank to 13 GB for half a minute and leaves it at
about a third of that. Forty-eight ranks reading it together do not fit in a node that holds
them comfortably once they have. With CLIENT_LOAD_GROUP=G in the environment the ranks build
G at a time, in rank order; unset or 0, all at once.
"""
import os
import time


def memory_gb():
    """(resident now, peak) of this process in GB; now is None where /proc is missing."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    try:
        with open("/proc/self/statm") as handle:
            now = int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except OSError:
        now = None
    return now, peak


def build_in_turn(comm, build, group=None, label="[load]"):
    """build() on this rank, while at most `group` ranks run theirs. Every rank passes the same
    barriers, so the groups follow one another. A rank that fails ends the job as it would have
    without the turns (mpirun stops the others)."""
    group = int(os.environ.get("CLIENT_LOAD_GROUP") or 0) if group is None else int(group)
    if group <= 0 or comm.size <= group:
        return build()
    result, started = None, time.time()
    for first in range(0, comm.size, group):
        if first <= comm.rank < first + group:
            result = build()
        comm.Barrier()
        if comm.rank == 0:
            print(f"{label} likelihood built on ranks {first} to {min(first + group, comm.size) - 1} "
                  f"of {comm.size} ({time.time() - started:.0f} s since the first)", flush=True)
    return result
