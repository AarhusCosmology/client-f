# Non-linear corrections at points with dark acoustic oscillations (21 October 2026)

`report.tex` / `report.pdf`: the report. `build.py` makes `figures/`, `tables/` and `numbers.tex` (every number
quoted in the text) from `data/`; then `pdflatex report.tex` twice. Not for GitHub.

Where the data came from:

| file | source |
|---|---|
| `training_points.npz` | the first SIDR + DRMD Pantheon+ run (kit `results/20261020_122013_sidr_drmd_pp`, `data_it_18.csv`): points inside the second set's box, and the 41 refused ones |
| `pk_lin.npz` | `compute_pk.py`, the laptop's CLASS++ (unpatched; linear spectra are not affected by the patch) |
| `table_end_probe.txt`, `table_end.json` | a debug build of the patched CLASS++ (fast-math off) printing the end of the source table where a fallback branch acts; `k_eq` from classy |
| `kmax_ladder_laptop_*` | kit `smoke/kmax_ladder.py` on the laptop, unpatched classy (HMcode `first.json`: the run before the spectra were saved) |
| `guard_ladder_halofit.npz` | the patched CLASS++ executable on the laptop, halofit at k_max 5, 10, 20, 30, 8 NaN points + best fit + Planck LCDM |
| `nl_switch_off_laptop.txt` | unpatched and patched executables, `nonlinear_verbose = 1`, k_max 5: where the non-linear correction stops |
| `grendel_{env,guard}_*` | Grendel job 6927488 (`smoke/submit_guard_test.sh`, patch version 2): env build and patched build |
| `kmax_loglike_laptop_corner.*` | kit `smoke/kmax_loglike.py` on the laptop: CamSpec TTTEEE + low-l TT, EE at 8 corner points, 3 others, the best fit; linear, halofit k_max 5 and 30 (unpatched classy; the patch does not change these points) |
| `corner_spectra.npz` | the same points with both executables: lensed TT and C_L^phiphi, linear and halofit |
