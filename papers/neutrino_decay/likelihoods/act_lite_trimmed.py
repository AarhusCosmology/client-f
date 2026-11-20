"""ACT-lite (act_dr6_cmbonly) asking for C_l only as far as its windows reach: l = 6327 instead of 9001.

Thomas Tram's subclass from CLASSpp #486 (docs/superpowers/specs/2026-10-29-act-dr6-small-scale-preset-design.md,
merged 29 October 2026). With its default cuts (600 to 6500) every window ACT-lite uses is zero above l = 6325, but it
asks for max(lmax_theory, top of the window arrays + 1) + 1 = 9001 (lmax_theory 9000, windows to 8501), and Cobaya's
classy theory computes to the largest request. Cutting the windows to their non-zero rows leaves the chi^2 identical to
the last bit (Thomas, against the unmodified code) and lowers the request to 6327; the other likelihoods of the H0WC
runs ask for at most 5000 (MUSE's unlensed EE). Checked against ACT-lite at commit 627aeafb (env.sh): its
initialize() fills self.spec_meta with each spectrum's sacc window (values, weight) and chi_square() reads only those.

In a Cobaya input (30 October 2026, the fast H0WC settings, runs/axion_wc_fast):

    act_lite_trimmed.ACTDR6CMBonlyTrimmed:
      python_path: likelihoods          # relative to hpc_client/, where every job runs

smoke/act_lite_trim_check.py compares it with the original on the same spectra.
"""
from types import SimpleNamespace

import numpy as np
from act_dr6_cmbonly import ACTDR6CMBonly


class ACTDR6CMBonlyTrimmed(ACTDR6CMBonly):
    lmax_theory: int = 0    # 0: the floor comes from the windows

    def initialize(self):
        super().initialize()
        top = 0
        for m in self.spec_meta:
            w = m["window"]
            last = int(np.flatnonzero(np.any(w.weight != 0, axis=1)).max())
            m["window"] = SimpleNamespace(values=w.values[:last + 1], weight=w.weight[:last + 1])
            top = max(top, int(w.values[last]))
        self.lmax_theory = top + 1
