# First run

## Install the version described here

Use a separate environment:

```sh
git clone https://github.com/AarhusCosmology/client-f.git
cd client-f
conda env create -f environment.yaml -n clienv
conda activate clienv
```

The [repository installation guide](_generated/library/README.md#getting-started) also describes the interactive setup and backend-specific prerequisites. Analytic Cobaya targets do not require CLASS++, MontePython or Planck data; real cosmological likelihoods require their own theory and data installations.

## A two-parameter Gaussian

The documentation includes a small, explicit Gaussian target and a short CLiENT configuration. From the repository root:

```sh
python client.py docs/site/examples/gaussian_client.yaml \
  -n documentation_gaussian -o results/docs_example -i 1
```

This is an **installation and output smoke test**, not an accuracy benchmark. It requests one fresh iteration (iteration 0), 128 initial evaluations, a small network and short training/sampling budgets. It enables neither audit nor boost nor injection. Do not use those budgets for a cosmological analysis.

The two configurations can be downloaded: [CLiENT YAML](examples/gaussian_client.yaml) and [Cobaya YAML](examples/gaussian_likelihood.yaml). They are also displayed in the [worked example](examples/gaussian.md).

## Inspect the output

The command prints the timestamped run directory. It holds a copy of the settings and `metadata.json`, and:

| In the run directory | What to look for |
|---|---|
| `training_data/` | Native objective values and parameter order |
| `trained_models/` | Saved model for each completed iteration |
| `training_history/` | Training and validation loss of each training |
| `convergence_stats/` | Summaries of the sampled chains, used by the convergence test |
| `metrics.json`, `metrics.md` | Each iteration's training, convergence and acquired-batch error |
| `objective_calls/` | The journal of true evaluations |
| `objective_identity.json`, `phases/` | What a continued run is checked against ([recovery guide](_generated/library/docs/run_recovery.md)) |
| `likelihood_input/` | Copied backend input for the run |

These files answer different questions. Low validation loss measures the fit to held-out training data; it need not measure accuracy in the posterior core or an unseen mode.

## Continue a run

Replace `RUN_DIR` with the printed directory:

```sh
python client.py RUN_DIR -i 1
```

For an existing run, `-i` requests additional iterations. The loader inspects saved data and models. `--start` selects an explicit starting iteration; in a run that has gone further it is refused unless `--rewind` is given, which first moves everything of the later iterations to `RUN/interrupted/` (nothing is deleted). The training command's `--retrain` forces training when a model is already saved. This flag is distinct from the profile command's opt-in training setting.

For expensive acquisition, MPI can distribute independent objective evaluations:

```sh
mpirun -n 6 python client.py RUN_DIR -i 1
```

Choose a rank count for the host and the backend's memory use; prevent nested numerical-library threads from oversubscribing its cores. **Do not put the profile command under `mpirun`**: it manages its own workers. See [inference workflows](inference.md).

## Next steps

Read [training and boosting](workflow.md), then the [configuration reference](configuration.md). The defaults alone are not the method of the real runs: `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml` are the settings to start from for a real likelihood. If you have used the first release of CLiENT, the [migration notes](_generated/library/UPGRADING.md) say what changes for an existing run or setting.
