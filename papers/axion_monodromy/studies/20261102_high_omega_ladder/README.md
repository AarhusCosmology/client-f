# Can CLASS++ resolve the oscillation at 60 <= omega <= 200? (2 November 2026)

CLASS++ 0a57bbc6 (the axion runs' build; the laptop copy in the session scratchpad, its bbn/hyrec data restored from git),
one thread per setting, A_osc 0.02 minus A_osc 0 ("the signal"), errors against the densest setting as a noise-weighted
chi^2 of the difference: Planck-like noise for PDP+ (TT to 2500, TE/EE to 2000, f_sky 0.6), ACT-like for H0WC
(l 30-6327, f_sky 0.3). A relative error e of the signal gives e^2 times the signal's own norm.

| file | |
|---|---|
| `ladder3.py`, `generate_Pk_kpd.py` | one setting per process (`pp` or `wc`); custom7 = points per decade of the primordial table |
| `run_all.sh`, `run_extra.sh` | the settings as run (BUILD=install directory) |
| `compare3.py pp|wc`, `compare_lensing.py` | the tables below |

PDP+ (Planck noise; error against REFP = l 1.005/2, q 0.025/22, 2000 per decade; REF = 1.01/3, q 0.05/45, 1000 agrees
to < 0.0005):

| omega | signal | runs P (1.026/8, 200/dec), 1.6 s | L8Q (1.013/8, q 0.15/90, 1000/dec), 11 s | L4Q (1.013/4), 19 s |
|---|---|---|---|---|
| 60 | 5.4 | 0.004-0.009 | 0.000 | 0.000 |
| 80 | 3.8 | 0.010-0.015 | 0.000 | 0.000 |
| 100 | 2.9 | 0.021-0.033 | 0.000 | 0.000 |
| 130 | 2.1 | 0.079-0.092 | 0.000 | 0.000 |
| 160 | 1.7 | 0.19-0.25 | 0.001 | 0.000 |
| 200 | 1.5 | 0.35-0.37 | 0.008-0.010 | 0.001-0.003 |

One change at a time: the primordial table (200 -> 1000 per decade) changes nothing with the runs' l and q, and costs
0.04 at omega 200 once l and q are dense (L8Q200); dense q alone (PQ) or dense l alone (L4K) leaves 0.29 and 0.06-0.10 at
200: both are needed. Fast lensing (the PDP+ runs: correlation function on theta < pi/16) against accurate_lensing 1 at
converged sampling (REFL): <= 0.002 up to omega 130 (0.011 at 60, phi 0), 0.010-0.024 at 160-200, where the signal
drops from 1.5 to 1.25 with accurate lensing.

H0WC (ACT noise; error against WREF = l 1.01/3, q 0.05/45, 1000/dec, ACT preset with accurate lensing):

| omega | signal | runs W (1.026/20, 200/dec), 8 s | WL (1.013/8), 15 s | WLQ (1.013/8, q 0.15/90, 1000/dec), 42 s |
|---|---|---|---|---|
| 60 | 10.1 | 0.001 | 0.000 | 0.000 |
| 100 | 4.7 | 0.088 | 0.013 | 0.000 |
| 150 | 2.8 | 1.08 | 0.026 | 0.000 |
| 200 | 2.0 | 1.97 | 0.092 | 0.006 |

The runs' settings are good to omega ~ 90 (PDP+, as the probe found) and ~ 80-100 (H0WC). Above that the signal is off
by 20% (PDP+) and 60% (H0WC) at omega 130-150 and by 50% and 100% at 200 (e = sqrt(error / signal)). For a network on
[55, 200]: PDP+ L8Q + accurate lensing (~7 times the runs' CLASS++ time), H0WC WLQ (~5 times). The signal itself falls
by 3.6 (PDP+) and 5 (H0WC) from omega 60 to 200 (unbinned; binned likelihoods lose more), so the amplitude limits at
omega ~ 200 are about twice weaker.

## The probe to omega 200 on Grendel (set up 2 November 2026)

`make_omega200.py KIT` writes the kit's runs/axion_pp_omega200 (from axion_pp_high) and runs/axion_wc_omega200 (from
axion_wc_combined, the H0WC final fast likelihood): omega prior [1, 200], box omega [55, 200], the converged settings
above (custom7 = 1000 points per decade, read by their own generate_Pk.py, a copy of generate_Pk_kpd.py), nothing else
changed (diff them against the sources). `make_probe_smoke200.py` makes a stand-in smoke kit (CMB stand-in about a
fiducial with A_osc 0.02, omega 150, phi 1.0); the smoke ran smoke/axion_probe.py on 4 ranks (probe_smoke200.out in the
scratchpad smoke200/).

On Grendel, the phase probe over omega 55-200 (step 0.25, 8 phases, A_osc 0.02 and 0.05: 9297 evaluations at the
reference point) and the sampling check of the strongest peaks and omega 60-200 against the runs' old settings and REF:
see the commands in the session's closing message (sbatch smoke/submit_axion_probe.sh axion_pp_omega200 / axion_wc_omega200).

Smoke (probe_smoke200.out, 4 ranks, omega 148-152 step 1): the injected oscillation found exactly (omega 150.00,
A_osc 0.0200, phi +1.000, chi2 gain 1.72); sampling check: the new settings within 0.002 in ln L of REF, the runs' old
settings (1.026/8, default q, 200 per decade, fast lensing) 0.09 lower at omega 150 (a tenth of the gain); 11-14 s per
evaluation on the laptop against 4 s with the old settings, REF 92 s.

Literature (checked 2 November 2026 from the arXiv HTML and the Planck 2015 PDF text): no feature search reports a
convergence test of the Boltzmann code at high frequency. Flauger et al. 2009 (every multipole), Planck 2015 ("a
high-resolution version of CAMB"), Planck 2018 ("suitably increased numerical precision settings"), Hamann & Wons 2021
(every multipole, denser k, no numbers) state what they changed; Easther & Flauger 2014 (cached transfer functions),
Flauger et al. 2014 (template checked against the numerical primordial spectrum only, scan to alpha 1000), Meerburg &
Spergel 2014 (perturbative, "sufficient k"; higher frequencies would need higher orders), Antony et al. 2024, Peng et al.
2025, Calderon et al. 2025, Raffaelli et al. 2026 give no settings; Nerval et al. 2026 refer to another paper ("C25").

## The probes to omega 200 (jobs 7010679 PDP+, 7010680 H0WC; 2 November 2026, 2 nodes each)

After a failed start (Grendel's Slurm 26.05: a Python that starts MPI by itself aborts; kit fixed in env.sh and the job
scripts). 9297 evaluations each, 24.5 s (PDP+) and 78 s (H0WC) per evaluation on one core. `plot_scan.py` ->
`scan_omega200.png` (PDP+ joined with the earlier probes 6973447 below 40 and 6973710 for 40-55).

| | largest chi^2 gains above omega 60 (reference point, profiled over A_osc and phi) |
|---|---|
| PDP+ | 8.57 at 196.0 (A 0.048), 7.95 at 179.0 (0.047), 6.52 at 100.75 (0.029), 5.83 at 104.75; the omega ~ 51 peak has 10.2 |
| H0WC | 6.20 at 157.75 (0.054), 5.56 at 141.5, 5.01 at 81.5, 4.97 at 195.5 (phase 1.34, as PDP+'s 196 at 1.03), 4.95 at 172.0 |

Sampling check (true likelihood, gain in ln L, converged settings against REF): PDP+ within 0.05 up to omega 196, 0.25
and 0.11 at omega 200; H0WC within 0.03 up to 160, 0.09-0.12 at 195-200. The runs' old settings: PDP+ 1.22 against 4.36
at omega 196 and 0.44 against 1.86 at 200; H0WC -0.22 against +3.10 at 157.75, -2.53 against +2.41 at 195.5, and 3.36
against 2.81 at 141.5. PDP+ overlap with 6973710 (old settings) over 55-100: Delta chi^2 rms 0.05 (55-70), 0.11 (70-85),
0.13 (85-95), 0.32 (95-100). A network on [55, 200] cannot be combined with the existing parts as it stands: client.py
combine requires the same objective, and the settings differ (A_osc = 0 baseline 0.03 apart in ln L).

## The training runs of omega [55, 200] (set up 3 November 2026)

`make_runs200.py KIT` (after make_omega200.py) turns the two probe inputs into training runs: omega prior = training box
[55, 200], phi windows from the probes (PDP+ phi0 +1.0009, H0WC +1.9007), client.yaml = the high parts' settings (warm
start, convergence then ten boosts of 10 000). Smoke: `make_standin200.py` (analytic stand-in, peak at omega 150) through
the kit's launch.py on 4 ranks, -i 4, both runs (scratchpad smoke_run200/): four networks each, two boosts, training points
inside omega [55, 200] and the phi windows. Launch: sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh
axion_pp_omega200 (and axion_wc_omega200). Results joined with the existing [1, 60] emulators by evidence weights
(client.py combine refuses the two settings); a softer combine to be discussed with Steen once trained.

## The PDP+ high part in four omega zones (set up 4 November 2026)

Why: after six boosts (networks 8 to 13) axion_pp_high200 (omega [2.5, 200]) had localised no omega peak. Its best true
value stayed at -6181.84 (omega 178.7), its boosts spread over omega 30-195, and its networks were tilted (about 1 too
low near the best, 1.5-2 too high deep in the tails, scatter about 1 at every depth). The omega < 60 run settled at its
network 8, exactly when its boosts locked onto the omega ~ 51 peak (41% within +-1 of 51.6). Steen: run the range in
about four zones, PDP+ first, H0WC later if needed; no added profile points.

The cuts (`omega_cuts.py KIT_LOGS`) lie where the probe scans have no peak in either data set: within +-1 of each band
the largest PDP+ / H0WC gain is 0.8 / 1.7 (61), 1.1 / 0.7 (111) and 1.1 / 0.6 (154). The zones are 58, 50, 44 and 45 wide
(the omega < 60 run was 57) and most have one leading peak: 51.5 (10.2), 100.75 (6.5), and 196 (8.6) with 179 (8.0);
zone 3 has eight peaks of 3 to 4 and none that stands out. 88 is the cleanest cut of all (0.9 within +-2 in both), for a
fifth zone.

`make_runs_zones.py KIT` makes runs/axion_pp_zone1..4 from runs/axion_pp_high200: cobaya.yaml and job.env byte-identical
(prior omega U[1, 200]; every part runs the high part's generate_Pk.py), client.yaml's settings unchanged (members 12,
member_pilot 100, ten boosts), domain_bounds.json with omega [2.5, 63], [59.5, 113], [109.5, 156], [152.5, 200]. Each box
reaches 1 beyond its bands. Checked: the four configurations load with CLiENT's schema (dev-sth1509 d3dfcbe), every box
lies inside the uniform priors, the bands inside both adjacent boxes; `combine_four_parts.py CLIENT_ROOT` runs a four-part
client.py combine on tests/test_combine.py's analytic runs (each network exactly in its range, the blend in each band),
since the unit tests combine two parts only. With the existing low part the five are combined by

    python client.py combine OUTPUT --parameter omega --part results/<time>_axion_pp_low200 \
        --part results/<time>_axion_pp_zone1 --part results/<time>_axion_pp_zone2 \
        --part results/<time>_axion_pp_zone3 --part results/<time>_axion_pp_zone4 \
        --band 2.6:3.0 --band 60.5:62.0 --band 110.5:112.0 --band 153.5:155.0

Launch, two nodes each: sbatch --nodes=2 --ntasks=96 --time=5-00:00:00 submit_client.sh axion_pp_zone1 (2, 3, 4).

### Zone 2 split again at omega = 88 (5 November 2026)

At about network 10 (jobs 7031156-9, logs of 08:30) the boost scatter, boost by boost, was 1.64 ... 0.384, 0.287 (zone 1),
1.33, 1.06, 1.10, 0.952, 0.974, 1.07, 1.00 (zone 2), 1.63 ... 0.240, 0.145 (zone 3) and 1.54 ... 0.189, 0.137 (zone 4). Zone 2's
warm networks were 26 to 47 too high at their own pilot points four times running, as axion_pp_high200's were; once
settled, zones 1, 3 and 4 kept their warm networks. Zone 2 holds the pair 100.75 (6.5) and 104.75 (5.8). Steen: split it.
`make_runs_zone2_split.py KIT` makes runs/axion_pp_zone2a (omega [59.5, 89.75]) and axion_pp_zone2b ([86.25, 113]) in the
same way, band 87.25:88.75 (largest gain in it 0.55 PDP+, 0.81 H0WC); the four zone runs' files are left as they are. Six
parts with the low part, combined with --band 2.6:3.0 --band 60.5:62.0 --band 87.25:88.75 --band 110.5:112.0 --band
153.5:155.0. Should 2b stay stuck as well: between its two peaks the gain falls to about 0.9 near omega 102.5, the only
place left to cut it.

## The H0WC high part in omega zones (set up 5 November 2026)

axion_wc_high200 stuck as axion_pp_high200 was: boost scatter 0.99-1.10 over boosts 8-13, warm networks rejected for
false maxima, best true value unchanged at -2861.11 (omega 195.4). Steen: split it as PDP+. `make_runs_wc_zones.py KIT`
makes runs/axion_wc_zone1 [5.5, 63], 2a [59.5, 89.75], 2b [86.25, 113], 3 [109.5, 156], 4a [152.5, 176.25] and 4b
[172.75, 200] from runs/axion_wc_high200 (cobaya.yaml, job.env byte-identical; settings unchanged, balance_floor 0.1
included). The PDP+ cuts are quiet on H0WC's scan too (largest gain in the bands 1.04, 0.81, 0.68, 0.66); the extra cut
at 174.5 (band 173.75-175.25, largest gain 0.75) parts H0WC's top zone, nine peaks of 3.4-6.2, into the comb
157.75-172 and the group 178-198. Seven parts with the low part axion_wc_low200 ([1, 6.1]):

    python client.py combine OUTPUT --parameter omega --part results/<time>_axion_wc_low200 \
        --part results/<time>_axion_wc_zone1 --part results/<time>_axion_wc_zone2a --part results/<time>_axion_wc_zone2b \
        --part results/<time>_axion_wc_zone3 --part results/<time>_axion_wc_zone4a --part results/<time>_axion_wc_zone4b \
        --band 6.0:6.1 --band 60.5:62.0 --band 87.25:88.75 --band 110.5:112.0 --band 153.5:155.0 --band 173.75:175.25

Zone 3 holds a pair of comparable peaks (141.5 (5.6), 146.25 (4.8)), like PDP+'s zone 2; if it sticks, the gain
between them falls near omega 143.5.
