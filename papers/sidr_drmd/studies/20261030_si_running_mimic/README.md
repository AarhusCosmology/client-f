# Can the neutrino self-interaction be mimicked by n_s and its running? (30 October 2026)

Prompted by Martin Sloth's comment on the SIDR+DRMD draft: self-interacting neutrinos look like an alternative to
the running (and running of the running) with which Garny, Niedermann & Sloth (arXiv:2604.26541, `Garny:2026gcs`)
make dark radiation compatible with ACT.

**Method** (`mimic.py`; `plot.py` makes `mimic_residuals.png`).
- **Spectra:** at the H0WC+SH0ES best fit of the SInu+DRMD model (paper Sec. 5.5), compute the lensed TT, TE and EE
  and phi-phi with CLASS++ bfe6729e. Use the World Cup run's settings (`runs/sidr_drmd_wcmb_high/cobaya.yaml`,
  l_max 5000), and a primordial spectrum passed as `external_Pk` (checked against the analytic one to 1e-9).
- **The comparison:** the spectra with the self-interaction are compared with those of log10 G_eff = -5.5 (no
  self-interaction), everything else fixed.
- **The fit:** A_s, n_s, alpha_s, beta_s (and optionally 100 theta_s) of the model without self-interaction are
  fitted to the spectra with it. The fit is Gauss-Newton on a Gaussian chi^2 of TT, TE and EE, with one Jacobian and
  the residual recomputed at every step; 36 to 38 CLASS++ calls per case, about 6 s each on 4 threads.
- **Errors**, approximate:
  - "P+A": Planck-like for 30 <= l < 1000 and ACT-like for 1000-6500;
  - "CV": cosmic variance to l = 4000, f_sky 0.4.

  The other cosmological parameters are fixed, so the remaining chi^2 is an upper bound for these freedoms.

**Result.** chi^2 between the two sets of spectra, P+A errors (CV in brackets):

| freed in the model without SI | best fit, log10 G_eff = -2.81 | strong mode, log10 G_eff = -1.7 |
|---|---|---|
| nothing | 2944 (26128) | 15007 (90377) |
| A_s, n_s | 67 (595) | 690 (2314) |
| + alpha_s | 26 (392) | 663 (2103) |
| + alpha_s, beta_s | 26 (191) | 657 (2042) |
| + alpha_s, beta_s, theta_s | 9 (95) | 28 (115) |

- **At the best fit,** the self-interaction acts on the primary spectra like a higher spectral index and a positive
  running: n_s +0.018, alpha_s +0.021, A_s +1%. Running of the running adds nothing at Planck+ACT precision, and a
  small shift of the peaks (100 theta_s -0.0006, the neutrino phase shift) takes most of the rest.
- **The lensing tells them apart.** The model with running has 3% (L = 400) to 5.5% (L = 763) less lensing power, and
  9% less at L = 2000. ACT DR6 lensing measures the amplitude to about 2.3% over 40-763.
- **The matter power** at k = 3 /Mpc is 36% higher with running.
- **In the strong mode** the effect is mainly a phase shift of the acoustic peaks. The primordial spectrum cannot
  reproduce it (chi^2 663 with running), only a shift of theta_s six times larger does, and then the running is about
  zero.

Files: `result_best.json`, `result_strong.json` (shifts, chi^2 by band, lensing and P(k) deviations),
`spectra_*.npz`, `log_*.txt`.
