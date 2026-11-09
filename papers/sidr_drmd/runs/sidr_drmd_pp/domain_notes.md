# Priors and training box of the SIDR + DRMD runs (20 October 2026)

For both runs, `sidr_drmd_pp` (Pantheon+) and `sidr_drmd_ppshoes` (Pantheon+SH0ES): the same
priors (`cobaya.yaml`) and the same box (`domain_bounds.json`). 21 parameters. Every range lies
inside the parameter's prior (checked when the files were written; `domain.py` refuses a box
that does not).

**The rule.** Box = centre ± 6 σ, cut at the prior. Where the preprint's prior cut its own
posterior within 6 σ, the prior is widened to about 6 σ (Steen, 20 October 2026: "stick to our
usual ±6 σ convention").

**The σ.** They are estimates, from the nearest analyses of this model and these data:
- **ΛCDM parameters:** the preprint's Table 2, for both data sets, with asymmetric errors. That
  table was made with Planck lite (its placeholders); plik-lite constrains almost as the full
  plik does.
- **n_s and logGeff:** the strong mode of the kit's SINU run (`results/20261015_120356_e2e_sinu16`,
  network 31: massless SIDR on CamSpec TTTEEE + low-l TT + DESI DR2). Its weak mode has
  n_s 0.9674 ± 0.0042 and logGeff −4.23 ± 0.50. Its strong mode has 25% of the posterior,
  n_s 0.9369 ± 0.0036 and logGeff −1.572 ± 0.106.

| parameter | preprint prior | prior here | box | box in σ: Pantheon+ / +SH0ES |
|---|---|---|---|---|
| omega_b | 0.019 to 0.025 | 0.019 to **0.0255** | 0.0203 to 0.0255 | −6.1/+6.3, −7.7/+6.4 |
| omega_cdm (all dark matter) | 0.095 to 0.145 | **0.079 to 0.181** | 0.079 to 0.181 | −6.0/+6.0, −8.1/+6.5 |
| H0 (preprint: h) | 55 to 80 | 55 to **85** | 57.8 to 85 | −6.0/+6.1, −9.4/+7.6 |
| logA | 2.5 to 3.7 | as the preprint | 2.80 to 3.28 | −6.5/+6.0, −6.1/+6.8 |
| n_s | 0.9 to 1.04 | **0.88 to 1.06** | 0.88 to 1.06 | upper +6.5, +6.0; lower: 15 σ below SINU's strong mode |
| tau_reio | 0.01 to 0.1 | as the preprint | 0.01 to 0.1 (the prior) | full Planck low-l EE: σ ≈ 0.007, top ≈ +6 |
| m_int [eV] | 0.0005 to 1 | as the preprint | 0.0005 to 0.15 | 0.013 + 6 × 0.022 (upper error) |
| logGeff | −5.5 to −0.1 | as the preprint | −5.5 to −0.8 | top +7.3 above SINU's strong mode (also SINU's own cap) |
| delta_Neff_drmd | 0 to 2 | as the preprint | the prior | the posterior reaches the top within 6 σ |
| f_idm_drmd | 0 to 1 | as the preprint | the prior | see below |
| log10_zstop | 2 to 5 | as the preprint | the prior | unconstrained over most of it |
| log10_G_over_aH | 2 to 14 | as the preprint | the prior | unconstrained (7.3 +6.0 −5.3) |
| 9 CamSpec nuisances | | as SINU | SINU's box | A_planck, calTE, calEE ± 6 σ; amp and n their priors |

**What the preprint's prior had cut.** omega_cdm's top was 1.5 σ above the Pantheon+SH0ES mean
(0.1344 +0.0072). H0's top was 4.0 σ (Pantheon+) and 4.5 σ (SH0ES) above the means. n_s's bottom
was 2.9 σ below the Pantheon+ mean, by its lower error, which includes the strong mode.

**Why f_idm keeps its whole prior.** Its marginal (0.012 +0.023) is small, but it is a funnel.
With z_stop near 10⁵ the interaction ends long before recombination and f_idm can be large:
CLASS++ computes f_idm 1 at z_stop 10⁵, and only σ8 drops (0.51). With ΔNeff near 0, f_idm,
z_stop and G/aH do nothing at all. A box cut by the marginal would cut those directions.

**To check at the end of each run:** that no posterior comes near a box edge that is not a prior
edge (m_int's 0.15, logGeff's −0.8, logA's, omega_b's bottom and H0's bottom). If it does, the
box is too narrow and the run has to be repeated with a wider one.
