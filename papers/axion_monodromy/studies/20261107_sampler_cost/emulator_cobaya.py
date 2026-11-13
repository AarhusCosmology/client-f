"""The combined emulator (emulator.py) as a Cobaya external likelihood with keyword arguments, counting calls, for
PROSPECT's cobaya kernel. The call count of each process is written to prospect/calls_<pid>.txt at exit."""
import atexit
import os
from pathlib import Path

import emulator as em


def loglike(**kw):
    return em.logp([kw[n] for n in em.NAMES])


@atexit.register
def _write_calls():
    out = Path(__file__).resolve().parent / "prospect"
    out.mkdir(exist_ok=True)
    (out / f"calls_{os.getpid()}.txt").write_text(f"{em.CALLS}\n")
