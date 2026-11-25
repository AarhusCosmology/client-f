# A run's files

What CLiENT writes into `results/<time>_<NAME>/`, and the three things that keep it smaller: networks saved
without their optimizer's state, training sets written as increments (opt-in), and one journal of nodes per true
stage of a profile study. None of them changes a result. Runs and studies made before are read and continued as
they are.

## What is where

| folder or file | what | written |
|---|---|---|
| `<name>.yaml`, `likelihood_input/`, `metadata.json`, `objective_identity.json` | settings, backend input, box and parameter order, the objective's identity | at the start |
| `training_data/` | training set N: `data_it_N.csv` whole, or `added_it_N.csv`, the rows it added (below) | per acquisition |
| `derived_data/` | derived parameters of the same rows (`derived_it_N.csv`, `derived_added_it_N.csv`); derived emulators | per acquisition / training |
| `trained_models/model_it_N.keras`, `members/` | network N; with `training.members` every member and the record of the one kept | per training |
| `training_history/`, `convergence_stats/` | losses per epoch; chain summaries | per training / iteration |
| `audit/`, `boost/` | the audit's points and report; boost points, records, regions, training domains | per iteration |
| `phases/` | which files each phase committed, with their sha256, checked at every start ([recovery](run_recovery.md)) | per phase |
| `objective_calls/` | the journal of every true call | appended |
| `metrics.json`, `metrics.md` | each iteration's numbers | rewritten per iteration |
| `run.json` | the index below | rewritten per acquisition |
| `evidence/`, `modes/`, `posterior_chains/`, `analysis/`, `augmentation/` | what the commands and `hpc/analysis` write | per call |
| `interrupted/` | files set aside by a recovery or a rewind; nothing is deleted | when needed |

## Networks without the optimizer's state

`run/storage.save_model` leaves Adam's moments out of every saved network. Nothing ever read them: every network
is loaded with `compile=False` and a warm start builds a new optimizer. They were two thirds of every file: network
18 of a 14-parameter run on Grendel went from 12.9 MB to 4.3 MB, with the same predictions to the last bit. The
networks of the October and November 2026 runs were 93 GB of their 232 GB. Keras 3 has no `include_optimizer`, so
the optimizer is detached for the save and put back.

## Training sets as increments

Runs write training set N whole, `training_data/data_it_N.csv`, at every iteration. Each file starts with the bytes
of the one before, so every point is written again at every later iteration: the training sets of those runs added
up to 62.5 GB, of which the last set of each run was 5.9 GB (and 36 GB to 3.8 GB for the derived parameters).

```yaml
storage:
  training_sets: increments      # default: cumulative
```

With it an iteration writes `added_it_N.csv`, the rows of set N after those of set N - 1, in the same format and with
the same header (`derived_added_it_N.csv` for the derived values). Set N is the last whole file at or before N,
followed by the increments after it in order; put together it is the whole file byte for byte. A whole file is
written instead whenever the earlier rows did not stay as they were, or the earlier set is not the one the run read
or wrote, so the two kinds mix safely in one run, and a run that only ever wrote whole files is read as before. The
increments are committed and checked like the whole files (`phases/`), and a rewind sets them aside like any file of
a later iteration.

Read a training set through `run/layout.py`, never by opening `data_it_N.csv`, which in a run with increments
exists for the first set only:

```python
from run.layout import training_set, set_rows, set_files
frame = training_set("results/RUN", 12)                      # a DataFrame, as the whole file reads
```

Everything in this repository reads them that way. Scripts elsewhere that open `data_it_N.csv` themselves need the
same change before they are used on a run with increments; that is why the setting is opt-in.

## The index, `run.json`

`python client.py index RUN` prints what a run holds: per folder its files and bytes, per iteration its files, the
size of its training set and how it is stored, and whether it has a network. `--write` also saves that as
`RUN/run.json` (`--json` prints it); the loop rewrites it after every acquisition. It records the git commit of the
CLiENT that wrote it (`client`). It is built from the files and can always be built again: nothing needs it to read
or continue a run, and writing it never stops a run.

## Profile studies: one journal of nodes

The true stages of a profile study keep every finished node, so that a study stopped at its time limit goes on
where it was. They wrote one JSON file per node: 2,500 for a 50 x 50 surface, 34,693 of the 35,184 files of the
profile studies of October and November 2026. A stage begun now writes one journal, `true/nodes.jsonl` or
`true_surface/nodes.jsonl`: a line per finished node, written whole and synced to disk before the node counts as
done (`profiles/nodes.py`). The record of a node is what reading its file back gave, so a resumed study sees
exactly what it did before. A stage that already holds node files goes on writing node files, so no study is mixed
and old studies continue as they were. A last line cut by a killed job loses that node only: the fragment is kept
beside the journal (`nodes.truncated-<id>.json`) and removed from it before the next write. Read the nodes with
`profiles.nodes.read_nodes(STAGE_FOLDER)`, which reads both kinds; `hpc/true_profiles/status.py` does.
