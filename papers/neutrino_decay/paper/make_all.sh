#!/bin/bash
# Every figure, table and number of the paper, from the outputs of the jobs.
#
#   export DNCDM_HARNESS=/path/to/campaigns     # hpc_prod2/, hpc_gal/, hpc_gal2/, proxy_rta/, proxy_backgrounds/, each with out/ (and logs/)
#   export CLIENT_PAPER_KIT=/path/to/folder     # where the CLiENT jobs ran: its results/, results_profiles/ and logs/
#   bash paper/make_all.sh                      # PYTHON: a Python with numpy, scipy, matplotlib and getdist (and LaTeX for the text)
#
# DNCDM_HARNESS defaults to ../campaigns and CLIENT_PAPER_KIT to this branch's papers/neutrino_decay/. Everything is
# written to paper/fig/; the background tables are cached in paper/cache/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p fig
"$PY" fig_background.py                                  # figure 1 (and fig_bg_ladder.pdf, not in the paper)
(cd ladder && "$PY" fig_summary.py && "$PY" fig_signal.py)  # figures 2 and 5
(cd rate && "$PY" fig_shapes.py && "$PY" fig_ratefit.py)    # figures 3 and 4, and the fit of A3 and A5 (printed)
"$PY" -I fig_data.py                                     # figures 6-9 and fig/numbers_data.txt
"$PY" -I cost_table.py | tee fig/table1_cost.txt         # table 1
"$PY" -I eval_costs.py | tee fig/table2_client.txt       # table 2
