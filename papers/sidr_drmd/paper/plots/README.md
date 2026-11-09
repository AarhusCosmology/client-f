# Figures of the SIDR + DRMD paper: data and plotting scripts

Every figure in `../Figures/` is made by one script in this folder from the files in `data/` alone. The scripts need
Python with numpy, matplotlib and GetDist, and a LaTeX installation with the `lmodern` package. All text in the figures
is typeset by LaTeX in the paper's fonts, Latin Modern in T1 encoding (`common.py`: `text.usetex` and
`LATEX_PREAMBLE`). The figures were made with numpy 2.0.2, matplotlib 3.10.5, GetDist 1.7.1 and TeX Live 2024.

```sh
cd plots
python make_all.py              # the figures of Sec. 4 and App. B (PDP+ and PDP+SH0ES)
python make_all.py --series wc  # the figures of Sec. 5 and App. B ($H_0$WC and $H_0$WC+SH0ES), from data/wc/
python plot_triangle.py         # or one script (PAPER_SERIES=wc python plot_triangle.py for its Sec. 5 version)
```

The scripts write to `../Figures/`, where `main-client.tex` reads them. Settings shared by all the scripts are in
`common.py`: data-set names and colours, parameter labels and units, how the posterior samples are loaded (with
GetDist's smoothing settings), and the axis ranges of the posterior figures that differ from GetDist's own choice
(`PLOT_LIMITS`, below). `PAPER_SERIES=wc` in the environment (what `make_all.py --series wc` sets) switches every script
to the $H_0$ World Cup runs: data from `data/wc/`, data sets wc and wcmb, figure names ending in `_wc` or in the data
set's name. Adjust a figure by editing its script and running it again.

Every figure is drawn at the width of the text (440 pt, jcappub's `\textwidth`; `common.TEXTWIDTH`) and saved through
`common.save_text_width` (28 October 2026). That function widens or narrows the drawing until the figure, cropped to its
ink from a 600 dpi rendering (`common.ink_box`), is exactly that wide, and saves it with that crop. Since the paper
includes every figure at `width=\textwidth`, the figures are printed at scale 1, with the font sizes their scripts give
(ticks 6.5-7 pt, labels 8.5-9 pt, legends 7.5-8.5 pt). Their outermost ink lies on the left and right margins of the
text, measured to within 0.12 pt (one pixel at 600 dpi).

Before that, the figures were drawn 10-13 inches wide and shrunk by 0.39-0.63, so that their labels printed at 3-6 pt.
They were cropped to the boxes of their texts (`common.save`, still there), which can end half a point short of the ink.
The triangle plots now switch GetDist's font scaling off (`settings.scaling = False`). The panels of the profile figures
share their probability axis, and every multi-panel figure has its legend above the panels, no wider than they are.
A figure keeps all this after any change to its script, as long as it is saved through `save_text_width`.

Checked on 28 October 2026: a copy of this folder alone (scripts and `data/`), run with `make_all.py` and
`make_all.py --series wc` in an empty directory, remakes all 10 figures of the draft, identical pixel for pixel (at
100 dpi) to those in `../Figures/`.

The figures, with their numbers in the draft of 28 October 2026 (the LaTeX labels in brackets):

| script | figure(s) in `../Figures/` | in the paper | data |
|---|---|---|---|
| `plot_triangle.py` | `triangle_key.pdf`, `triangle_key_wc.pdf` | Figs. 1 (Sec. 4.1) and 4 (Sec. 5.5): posteriors with the global best fits (diamonds) [`fig:triangle_key`, `fig:triangle_key_wc`] | `posterior_<data>.npz`, `best_fits.json` |
| `plot_profiles.py` | `profiles.pdf`, `profiles_wc.pdf` | Figs. 2 (Sec. 4.2) and 6 (Sec. 5.5): the profile likelihoods (lines) and posteriors (shaded) of both data sets of a section [`fig:profiles`, `fig:profiles_wc`] | `profiles.json`, `posterior_<data>.npz` |
| `plot_profile_logGeff.py` | `profile_logGeff.pdf`, `profile_logGeff_wc.pdf` | Figs. 3 (Sec. 4.3) and 5 (Sec. 5.5): the profile of log10 Geff and the true chi^2 at the emulator's best fits on the boundary Delta N_eff = f_idm = 0 [`fig:profile_logGeff`, `fig:profile_logGeff_wc`] | `profiles.json` |
| `plot_profile_breakdown.py` | `profile_breakdown_pp.pdf`, `_ppshoes.pdf`, `_wc.pdf`, `_wcmb.pdf` | Figs. 7-10 (App. B.1): the profiles split into the contributions of the individual likelihoods [`fig:profile_breakdown_pp`, `_sh`, `_wc`, `_wcmb`] | `profile_breakdown.json`, `profiles.json`, `posterior_<data>.npz` |

The scripts of the figures that were left out of the paper on 27 October 2026 (accuracy against boosts, the noise of
the true likelihood, the emulator's profiles against the true ones, the region of dark acoustic oscillations) and the
full triangle plots were removed from this folder on 28 October, with their data; they are in
`../retired/plots_20261028/`.

The axis ranges of the posterior figures are GetDist's choice from the samples, except `common.PLOT_LIMITS` (27 October
2026): f_idm ends at 0.15 in both series (its prior reaches 0.4; beyond 0.15 lie 1.7% of the PDP+ posterior, a thin
tail, and at most 0.14% of the others), and m_int at 0.2 eV in the $H_0$WC series (0.01% beyond; with PDP+ and PDP+SH0ES
2.6-3.7%, so that axis keeps the prior). The captions say so. The profile figures keep their ranges.

## The data files (`data/` and `data/wc/`, 93 MB)

All posteriors, profiles and best fits leave out 0 < Delta N_eff < 0.01 (Sec. 3.2 of the paper): the posterior samples
there are dropped and the range of Delta N_eff starts at 0.01. Delta N_eff = 0 (LCDM, SInu) is not affected.

- **`posterior_<data>.npz`** (`data/`: pp, ppshoes; `data/wc/`: wc, wcmb): the posteriors of the combined emulators
  after ten boosts, from BEST nested sampling, seeds pooled (four; twelve for wc).
  - `samples` holds the weighted samples (single precision, as BEST and the derived-parameter emulators give them) of
    the twelve cosmological parameters and of the derived parameters sigma8, S8, Omega_m, r_drag and f_idr.
  - `weights` holds the sample weights, and `names` the column names.
  - `ranges` is a JSON string with the prior range of each cosmological parameter, Delta N_eff from the cut. GetDist
    uses it for the boundaries of its density estimates.
- **`best_fits.json`**: the global best fit of each data set (the diamonds of Figs. 1 and 4).
  - `logl` is the point's true log-objective.
  - `source` names the point it comes from (a profile node, a moved point or a minimisation).
  - `values` gives the parameter values; the derived ones are from the derived-parameter emulators.
- **`profiles.json`**: per data set.
  - `tmax`: the best true log-objective found, the reference of every Delta chi^2.
  - `P`: the nodes of each parameter's profile. Each node has `x`; `t`, the true Delta chi^2 (the best value found at
    the node); `s`, the emulator's own Delta chi^2 to the emulator's best; `d`, the true Delta chi^2 at the emulator's
    optimum; `reason`, how the node's refinement ended; and `replaced`, whether the node was replaced by a CLASS++ best
    fit on its face. For wc (the profiles with the cut, from the studies on the run's copy with Delta N_eff >= 0.01,
    the points moved to their nodes, and the node-by-node minimisations with the true likelihood; `../build_wc.py`,
    `cut_profile_nodes`) each node also has `origin`: `study`, `polish` or `chain`, where its best value came from.
  - `extra`: face points outside a profile's range.
  - `face`: the true Delta chi^2 at the emulator best fits of LCDM, SInu (moderately interacting, log10 Geff = -4) and
    SInu (its best fit) on the face Delta N_eff = f_idm = 0.
- **`profile_breakdown.json`**: per data set, the likelihoods and, for every cosmological parameter, the profile nodes
  with x, t (the profile's Delta chi^2), dchi2 (each likelihood's chi^2 minus its value at the global best fit) and
  dprior (the same for -2 ln prior), from the evaluation behind each node's value. A node without such an evaluation
  would be left out (`skipped`; none at present), and `plot_profile_breakdown.py` then breaks the lines there.
- **`accuracy.json`** (`data/` and `data/wc/`): no figure reads it. `../build.py` reads both files to write Table 8
  (`tables/accuracy.tex`, App. A.2 [`tab:accuracy`]).
  - Per data set and part of the emulator (`low` and `high` in Delta N_eff), per boost: the rms of true minus emulated
    ln L at the points of the next boost, which the network had not seen.
  - The true checks of the combined emulator: rms, mean and largest at 500 posterior samples, after 1, 5 and 10 boosts.

## Where the data come from

`../build.py --label boost10` writes `data/` from the synced runs on the laptop and then runs `make_all.py`;
`../build_wc.py` writes `data/wc/` and runs `make_all.py --series wc`. They also write the paper's numbers
(`../numbers.tex`, `../numbers_wc.tex`) and tables. The runs themselves are in the Grendel kit and are not
part of this folder. A rebuild overwrites `data/` but leaves the scripts as they are.
