# CLiENT


**CLiENT** (Cosmological Likelihood Emulator using Neural networks with TensorFlow) is a framework for emulating cosmological likelihood functions, bypassing the need for Einstein-Boltzmann solver codes like [CLASS](https://github.com/lesgourg/class_public) and [CAMB](https://github.com/cmbant/CAMB) for evaluation of the likelihood. CLiENT compares to observable emulators like [CONNECT](https://github.com/AarhusCosmology/connect_public), but has the advantage of producing a surrogate likelihood which is completely auto-differentiable.

**[Documentation](docs/README.md)** · [What this version adds](CHANGES.md) · [Moving from the first CLiENT](UPGRADING.md) · [Performance](PERFORMANCE.md)

To start, take `input/recommended_near_gaussian.yaml` for a likelihood close to a Gaussian, or `input/recommended_non_gaussian.yaml` for oscillations, several modes or curved ridges, and change its likelihood and sizes. Most methods are opt-in and these files switch them on: a YAML that sets only the required keys is not the method of the papers ([docs/boost.md](docs/boost.md#two-recommended-settings)).

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Prerequisites](#prerequisites)
    - [CLASS Setup](#class-setup)
    - [Planck Likelihood Setup (for MontePython)](#planck-likelihood-setup-for-montepython)
    - [MontePython Setup](#montepython-setup)
    - [Cobaya Setup](#cobaya-setup)
- [Usage](#usage)
  - [Running CLiENT](#running-client)
  - [Command Line Options](#command-line-options)
  - [Example Configurations](#example-configurations)
- [Benchmarking](#benchmarking)
  - [Profile likelihoods and Bayesian evidence](#profile-likelihoods-and-bayesian-evidence)
- [Algorithm Overview](#algorithm-overview)
  - [Key Features](#key-features)
    - [Loss Function](#loss-function)
    - [Activation Function](#activation-function)
    - [Temperature Scheme](#temperature-scheme)
- [Configuration](#configuration)
  - [Likelihood Configuration](#likelihood-configuration)
  - [Prior Configuration](#prior-configuration)
  - [Acquisition Configuration](#acquisition-configuration)
  - [Model Architecture](#model-architecture)
  - [Training Configuration](#training-configuration)
  - [Sampling Configuration](#sampling-configuration)
  - [Convergence Configuration](#convergence-configuration)
- [Output Structure](#output-structure)
- [Performance](#performance)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Create profiles from an existing network](#create-profiles-from-an-existing-network)

## Getting Started

### Installation

To use CLiENT, start by cloning the repository:

```bash
git clone https://github.com/AarhusCosmology/client.git
cd client
```

Then, from within the repository, run the interactive setup script:

```bash
./setup.sh
```

The setup script will guide you through creating a conda environment (~6 GB) and installing all dependencies including CLASS (~1 GB), MontePython (~1 GB) or Cobaya (~5 MB), and the Planck likelihood (~200 MB). The complete installation requires approximately ~8 GB of disk space.

### Prerequisites

Alternatively, you can manually set up CLiENT by creating a conda environment containing all necessary dependencies using:

```bash
conda env create -f environment.yaml -n clienv
```

Alternatively, manually install the dependencies listed in `environment.yaml`. To activate the environment, run:

```bash
conda activate clienv
```
or add this command to your `.bashrc` file to activate it automatically.

CLiENT evaluates likelihoods through [Cobaya](https://github.com/CobayaSampler/cobaya) or [MontePython](https://github.com/brinckmann/montepython_public). `environment.yaml` installs Cobaya, and the analytic targets (`input/example_cobaya.yaml`, `input/gaussian.yaml`, `input/banana.yaml`) need nothing more. Real-data likelihoods also need a Boltzmann code such as [CLASS](https://github.com/lesgourg/class_public). If using MontePython, you will also need the C-based `clik` code for Planck likelihoods (see setup below). Cobaya can use either the newer Python-native Planck likelihoods or the original `clik`-based Planck 2018 likelihoods, both installed via `cobaya-install`. For performance, the neural network training and MCMC sampling can use GPU resources via TensorFlow when available. The environment file pins the plain `tensorflow==2.19.0` wheel; for a GPU install use the `tensorflow[and-cuda]` build of the same version in its place.

#### CLASS Setup

Install the CLASS Boltzmann code in your chosen directory:

```bash
git clone https://github.com/lesgourg/class_public.git
```

Build CLASS from the `class_public` directory:

```bash
make clean
make
```

Install the Python wrapper from the `class_public/python` directory:

```bash
python setup.py build
python setup.py install
```

#### Planck Likelihood Setup (for MontePython)

Install the Planck likelihood package in your chosen directory:

```bash
git clone https://github.com/benabed/clik.git clik
wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"
tar -xzf COM_Likelihood_Data-baseline_R3.00.tar.gz
rm COM_Likelihood_Data-baseline_R3.00.tar.gz
```

From the `clik` directory, configure and install:

```bash
./waf configure --install_all_deps
./waf install
```

If `./waf configure --install_all_deps` fails, ensure you have working C and Fortran compilers as well as the BLAS/LAPACK and CFITSIO libraries installed. For more details, including building with Intel MKL, see the [clik documentation](https://github.com/benabed/clik). To source the clik profile, run (replace with your actual installation path):

```bash
source /absolute/path/to/clik/bin/clik_profile.sh
```
or add this command to your `.bashrc` file to source it automatically.

#### MontePython Setup

Install MontePython in your chosen directory:

```bash
git clone https://github.com/brinckmann/montepython_public.git
```

Copy the default configuration template to the CLiENT config directory (or another location of your choice):

```bash
cp /path/to/montepython_public/default.conf.template /path/to/client/config/default.conf
```

Edit `config/default.conf` to point to your actual installation paths:

```python
path['cosmo']    = '/absolute/path/to/class_public'
path['clik']     = '/absolute/path/to/plc-3.01/'
```

Replace with the absolute paths to your installations. The `clik` path is only needed if using Planck likelihoods.

Create `config/montepython.yaml` (`config/montepython.yaml.example` is a template) with:

```yaml
conf: /absolute/path/to/config/default.conf
path: /absolute/path/to/montepython_public/montepython
```

#### Cobaya Setup

`environment.yaml` installs Cobaya (`cobaya==3.6.2`). When installing the dependencies by hand, install it via pip:

```bash
pip install cobaya==3.6.2
```

To install likelihoods, you can either install individual likelihoods directly by name:

```bash
cobaya-install planck_2018_highl_plik.TTTEEE
```

or install all likelihoods referenced in your Cobaya input file:

```bash
cobaya-install your_cobaya_input.yaml
```

Use `--packages-path` to specify where likelihood codes and data will be installed, and add `--skip-global` to skip reinstalling packages already available globally (like CLASS). For more details, see the [Cobaya installation documentation](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html).

## Usage

### Running CLiENT

Start a new run:

```bash
python client.py input/example_cobaya.yaml -n my_run
```

or with MontePython:

```bash
python client.py input/example_montepython.yaml -n my_run
```

Continue from an existing run:

```bash
python client.py results/my_run_directory
```

For parallel likelihood evaluation with MPI (requires OpenMPI and mpi4py, which `environment.yaml` does not install and `setup.sh` offers; for real likelihoods on a cluster, with an explicit training box and refused points, see [hpc/](hpc/README.md)):

```bash
mpirun -n <N_processes> python client.py <input_yaml|run_directory>
```

MPI parallelises initial sampling and resampling likelihood evaluations. Training and MCMC remain serial (with surrogate sampling performed on the master process).

### Command Line Options

**New Run Mode:**
- `-n`, `--name`: Optional run name/tag for output organisation
- `-o`, `--output`: Base results directory (default: `results`)
- `-i`, `--iterations N`: Number of iterations (overrides convergence criterion)

**Continue Mode:**
- `-r`, `--retrain`: Retrain model at the starting iteration (default: skip retraining)
- `-s`, `--start N`: Starting iteration (auto-detected from latest if not specified).
  In a run that has gone further than network N it is refused: the loop would do iteration
  N again in place and leave the later networks and training sets, which the next start
  without `-s` would continue from, without the points of the iteration done again.
- `--rewind`: with `-s N`, go back to network N. Everything of the iterations after N
  (training sets, networks, audits, regions, their phase records, and the evidences, mode
  searches, chains and reports made of them) is moved to
  `RUN/interrupted/rewind_to_it_N-<time>/` under the paths it had, with a list in
  `REWIND.json`; nothing is deleted, and the true calls of those iterations stay counted in
  `objective_calls/`. The move plan and artifact hashes are recorded before any file moves.
  If interrupted, the next start finishes that rewind before selecting its checkpoint.
  A network, training set or record of the plan that has changed or is missing stops that
  start, and the message says what can be done; a report or other analysis file written
  again or removed in the meantime is set aside as it is found and noted. The run then does
  the acquisition of N again and goes on. Do not
  remove later files by hand instead: `phases/acquisition_it_N.json` commits
  `training_data/data_it_<N+1>.csv`, and a start refuses when a committed file is missing.
  To keep the run as it is and try something else from an earlier network, make another
  directory from it; for boosts from an earlier network that is the
  [branch tool](hpc/README.md#branching-a-run).
- `-i`, `--iterations N`: Number of (additional) iterations (overrides convergence criterion).
  Without it, `convergence.max_iterations` on a continued run counts from the starting
  network too: that many iterations more, not a total for the run. With `-i K` the run ends
  K networks after the starting one; when the starting network has to be trained first
  (`--retrain`, or its training was interrupted) that is K + 1 trainings.
- `--accept-identity-change`: continue once although the recorded objective identity differs,
  after checking that the objective itself is unchanged; the error lists what differs

A run whose last job stopped inside a training or a batch of true evaluations is continued
with the same command: the unfinished phase is set aside under `interrupted/` and done again
([docs/run_recovery.md](docs/run_recovery.md)).

CLiENT automatically detects the mode based on whether the path is a directory (continue) or file (new run).

### Example Configurations

**Cosmological Likelihoods:**
- `input/base2018TTTEEE_lensing_bao.yaml` - Base ΛCDM with Planck 2018 TT,TE,EE+lowE+lensing+BAO
- `input/sterileLCDM_TTTEEE_lensing_bao.yaml` - Sterile neutrino extension (N<sub>s</sub>m<sub>s</sub>ΛCDM)

**Test/Example Likelihoods:**
- `input/example_cobaya.yaml` - Simple 2D Gaussian example
- `input/example_montepython.yaml` - 2D example (`h`, `omega_cdm`) on the MontePython `sh0es` and `bao` likelihoods, through CLASS
- `input/gaussian.yaml` - 27D Gaussian with Planck-like covariance
- `input/banana.yaml` - 29D Banana-shaped likelihood
- `input/_smoke.yaml` - Tiny 16D banana run that checks an install; it says nothing about accuracy

**Settings to start from:** `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml` (see above).

## Benchmarking

Compare surrogate likelihood against reference chains:

```bash
python benchmarking/benchmark.py results/my_run_directory
```

Benchmark options:
- `-i`, `--iteration N`: Iteration to benchmark (auto-detects latest if not specified)
- `-n`, `--n-steps N`: Number of MCMC steps (defaults to `n_steps` from config)
- `-t`, `--thin N`: Thinning factor for chains (default: 1)
- `-p`, `--params`: Parameter names, or comma-separated 1-based parameter indices, to include in analysis
- `-c`, `--chains DIR`: Path to MontePython or Cobaya chains directory for comparison
- `--region run|none`: The network inside the region of its boosts when it has one (default), or the bare network
- `--no-training-data`: Skip loading training data visualisation
- `--no-training-history`: Skip loading training history

The benchmark script generates:
- Triangle plots comparing posteriors
- KL divergence metrics between distributions
- MAP point comparisons
- Convergence diagnostics


### Profile likelihoods and Bayesian evidence

Three further modules assess the surrogate on quantities other than the posterior:

```bash
# 1D profile likelihoods, with the true likelihood evaluated at each optimum
python benchmarking/profile_likelihood.py RUN_DIR -it 10 --true \
    -n 61 --grid-frac 0.5 -o RUN_DIR/profiles_it10.json

# the profile metric, and -- read this first -- the grid's resolution floor
python -m profiles.metric RUN_DIR/exact_profiles.json --floor
python -m profiles.metric RUN_DIR/profiles_it10.json -r RUN_DIR/exact_profiles.json

# Bayesian evidence from the network via the BEST nested sampler (Cobaya and MontePython runs)
python client.py evidence RUN_DIR [-i 10] [--n-live 2048] [--prior omega_b=0.019:0.025,...]
```

`client.py evidence` reports ln Z in the run's own prior convention (Cobaya's priors, or
MontePython's `.param` ranges with the run's domain where they are unbounded, or your own
ranges) and flags a sampling box that cuts the posterior; see
[docs/evidence.md](docs/evidence.md). Only the network is evaluated. BEST is not in
`environment.yaml`: install it, set `BEST_PATH`, or pass `--best-path` to its checkout.

### A region the acquisition does not reach

The acquisition samples where the current network puts the likelihood. Where the network is
too pessimistic nothing is evaluated, and nothing corrects it; successive networks agree, so
the convergence metric does not notice. If you suspect this along a parameter (a profile
that disagrees with the posterior, a second solution known from the physics), take a true
profile along it and add its points to the training set:

```bash
python client.py profiles RUN_DIR --config profile.yaml -o STUDY   # over the parameter's whole domain
python client.py augment RUN_DIR --profiles STUDY -n 300           # its true evaluations, through the density filter
python client.py RUN_DIR -i 5                                      # continue; the first training with the points runs longer
```

If a whole region is missed, a mode the networks rate far too low (known from the physics or
from an independent chain), run CLiENT again on a box cut down to it, the same likelihood and
prior with only the training box narrower in one parameter, and combine the runs into one
emulator ([docs/combine.md](docs/combine.md)):

```bash
python client.py combine COMBINED --parameter omega --part RUN_LOW --part RUN_FULL:18 --band 2.6:3.0
```

To use a network outside CLiENT (BEST, another sampler, a notebook), export it as a TensorFlow
SavedModel, which TensorFlow alone loads ([docs/export.md](docs/export.md)):

```bash
python client.py export RUN_OR_COMBINED EXPORTED          # EXPORTED/saved_model, emulator.json, README.md
```

`augment` adds only the points the training set is short of, by the acquisition's own
density rule, writes them beside the run without changing any existing file, and the
continued run picks them up. On self-interacting neutrinos with Planck + DESI DR2 this
taught the network a strong-coupling minimum that twelve ordinary iterations had missed.
See [docs/augment_training.md](docs/augment_training.md), including what it does not
establish.

### A region the network wrongly believes in

The opposite failure: a network can predict a region far from every training point to be
much better than anything evaluated. If the acquisition chain finds it, the next batch is
spent there; if it does not, the false optimum stays in the network for a profile or a
sampler to land on. In a wide box the first iterations of a run can be dominated by this.

```yaml
acquisition:
  audit: {}        # every default; absent or false: no audit
```

After every training the audit searches the network for its own maxima, evaluates the true
likelihood at those predicted more than 10 log-units above the best true value, fine-tunes,
and repeats, with at most a tenth of the batch in five rounds per iteration. The points
stay in the training set. On a 58-parameter Gaussian in a ±10σ box the audited runs drew
their first batch on the posterior at iteration 3 or 4 where the unaudited run needed 11,
and they ended on the exact posterior where the unaudited run ended with a false optimum
in its last network. On ΛCDM with the real Planck and DESI DR2 likelihood the audited run's
second batch was as close to the peak as the unaudited run's fifth, for 242 extra true
evaluations in 12 000. It corrects optimism only: a region the network under-rates is not a
maximum, and needs `augment` above. See [docs/audit_optima.md](docs/audit_optima.md).

For unevaluable audit maxima, optional `failure_policy: penalize` tries three nearby
points and, if all fail, fits a separate synthetic target 1000 log-units below the best
genuine evaluation. Probes share the existing budget; genuine data and exact profiles
remain free of synthetic labels. See the failure policy in the audit documentation.

### Separated modes and their weights

A batch drawn at the acquisition temperature lies far below the posterior (T n / 2 log-units
under the peak against n / 2), so the network's level in the core of each mode is an
extrapolation, and the weights of modes a chain cannot travel between come out anywhere: on
self-interacting neutrinos three good networks gave the strong-coupling mode 12.7%, 35.7% and
38.1%, where the measured value is 25%.

```yaml
acquisition:
  boost:
    n: 1000              # true evaluations, shared equally among the modes found
    start_iteration: 15  # from this iteration on, the acquisition is a boost
```

A boost iteration finds the network's modes, draws the same number of posterior samples
(T = 1) from each, evaluates them in place of the tempered batch, and trains on. It reports
the weights of the modes as the network has them and as corrected by the true values. On that
self-interacting neutrino run one boost of 1000 gave a corrected strong-mode weight of 25.4% and a second 25.1%,
while the network's own weight went from 13.5% to 20.7% and 23.4%. With a single mode the
boost is simply `n` posterior samples. It needs a network that is already good on its own
posterior samples; the report says whether it was. See [docs/boost.md](docs/boost.md).

Two sets of boost settings are recommended ([docs/boost.md](docs/boost.md#two-recommended-settings)):
for near-Gaussian likelihoods (Planck-like parameter estimation) the quadratic background with
`region: 2.0`, the settings of the reference runs; for likelihoods known to be non-Gaussian
(oscillations, several modes, curved ridges) also the training domain, a narrower region,
mixing, `mode_depth: auto` and the audit's chain search and posterior pilot. The training
domain is what an oscillating likelihood needs and what costs a near-Gaussian one accuracy.

Opt-in `background: quadratic` and `region: 2.0` put a fixed quadratic under the network
(one cold start, then warm starts) and keep the chains and the audit inside the region the
boost points reach; see [docs/boost_background_region.md](docs/boost_background_region.md).

Opt-in `mixing: 0.1` decides what counts as a separate mode by running a T = 1 chain and
measuring whether it travels between the candidate basins, in place of the fixed `barrier`
(which is too strict with few parameters and too lax with many). `python client.py modes
RUN_DIR` gives the modes of any saved network, their weights and posterior samples, without
the likelihood; see [docs/boost_modes_mixing.md](docs/boost_modes_mixing.md).

`density_filter: true` filters boost candidates against a balanced T=1
mode mixture; filtered points do not support the unweighted evidence correction.
See [density-filtered boosts and the completed analytic study](docs/posterior_density_boost.md),
including the known limitation that an uncleared audit does not stop the native loop.

The posterior, the profile and the evidence weight the likelihood surface in different
places, and an emulator accurate for one is **not** automatically accurate for another:
measured on a 31-dimensional analytic target, a network with 6.5x the profile residual rms
of another gave an evidence indistinguishable from it. Validation loss predicts neither.
See [docs/PROFILES_AND_EVIDENCE.md](docs/PROFILES_AND_EVIDENCE.md) for the conventions that
have to be right -- the prior offset, the box definition, and the profile metric's
resolution floor -- and for what is not yet established.

## Algorithm Overview

CLiENT implements a temperature-based iterative training scheme:

1. **Initial Sampling**: Configurable prior sampling (LHS, scrambled Sobol, or uniform) within restricted prior bounds (±n<sub>std</sub>σ around fiducial values)

2. **Neural Network Training**:
   - Deep feedforward network with the Alsing activation function
   - Mean Square Relative Error (MSRE) loss function emphasising high-likelihood regions
   - Early stopping with validation monitoring

3. **Tempered MCMC Sampling**:
   - Surrogate sampling with the built-in affine-invariant ensemble sampler (AIES)
   - Temperature T<sub>MCMC</sub> controls exploration

4. **Adaptive Resampling**:
   - Candidates weighted by L<sup>1/T<sub>T</sub> - 1/T<sub>MCMC</sub></sup>
   - k-NN density-aware acceptance criterion targeting a distribution proportional to L<sup>1/T<sub>T</sub></sup>
   - Training temperature T<sub>T</sub> interpolates between evidence-based (T<sub>T</sub> → 1) and uniform (T<sub>T</sub> → ∞) sampling strategies

5. **Convergence Monitoring**:
   - Gaussian posterior drift metric between successive iterations
   - Or, opt-in, the error of each network at its own tempered batch, against true values
     ([docs/tempered_convergence.md](docs/tempered_convergence.md))
   - Iterates until metric < threshold or maximum iterations reached

### Key Features

#### Loss Function

The MSRE loss transitions from absolute to relative error based on distance from best-fit:

```
loss ∝ [(χ²_surrogate - χ²_exact) / (χ²_exact + ε)]²
```

where ε ~ n(1 - 2/9n + k√(2/9n))³ (Wilson-Hilferty transformation) with k controlling the transition scale.

#### Activation Function

The Alsing activation function (which is also used in CONNECT) replaces ReLU (which lacks expressivity at negative values) with a linear function at negative values:

```
f(x) = [γ + (1 + e^(-βx))^(-1) (1 - γ)] x
```

where β controls the broadness of the transition region and γ controls the asymptotic slope at negative values. Both hyperparameters are trainable for each node in each layer.

#### Temperature Scheme

- T<sub>T</sub> → 1: samples proportional to evidence integral
- T<sub>T</sub> → ∞: uniform sampling across parameter space

## Configuration

All hyperparameters are specified in YAML format. See `input/example_cobaya.yaml` or `input/example_montepython.yaml` for documented configuration examples. Below are the keys of each section. A key shown commented out is optional, with its default or, where it is off by default, an example. An unknown key stops the run before any evaluation.

Two optional top-level keys: `seed` fixes the prior design, the network initialisation, the walker start and the acquisition draws, but not the AIES proposals, which run under XLA; `derived` (absent: off) retrains a derived-parameter emulator at every iteration ([docs/derived_parameters.md](docs/derived_parameters.md)). Derived values are recorded with every true evaluation either way.

### Likelihood Configuration

**MontePython:**
```yaml
likelihood:
  wrapper: montepython
  input: input/montepython/example.param  # MontePython .param file
  # identity_files: [/absolute/path]      # Extra objective files to hash at every start (docs/run_recovery.md)
```

**Cobaya:**
```yaml
likelihood:
  wrapper: cobaya
  input: input/cobaya/example.yaml  # Cobaya .yaml file
```

### Prior Configuration
```yaml
prior:
  n_samples: 5000          # Number of initial samples
  sampling_strategy: lhs   # 'lhs', 'sobol', or 'uniform'
  n_sigma: 10.0            # Prior restriction (±10σ from fiducial); null: the likelihood's own bounds
  # n_inject: 0            # Opt-in: points drawn once, after the design, from a quadratic fit (see below)
```

### Acquisition Configuration
```yaml
acquisition:
  n_append: 1000           # Number of new points added per iteration (B_t)
  pool_factor: 20          # Candidate pool size multiplier (pool = pool_factor * n_append)
  n_neighbors: 20          # Neighbours for density estimation
  target_temperature: 7.0  # Training temperature T_T
  # n_inject: 0            # Opt-in, e.g. 100: points drawn every iteration from a quadratic fit of the
                           # training set (prior.n_inject: once, after the design). Both injections
                           # fit only the points within 16 x T n / 2 log-units of the best
                           # (docs/injection_window.md)
  # boost: {n: 1000, start_iteration: 15}   # Opt-in: from that iteration on, the new points are n
                           # posterior samples of the network (T = 1), the same number from every
                           # mode, in place of the tempered batch (docs/boost.md). Further keys and
                           # defaults: barrier 4, min_share 0.02, walkers 512, cool_steps 2000,
                           # steps 10000, components 6; mixing (null; 0.1: modes decided by
                           # measured mixing, docs/boost_modes_mixing.md), candidate_barrier 1;
                           # density_filter false, pool_factor 20; background, region,
                           # region_quantile, train_domain, mode_depth, balance_floor and boosts
                           # all null
  # audit: {}              # Opt-in: true evaluations at the network's own maxima after every
                           # training (docs/audit_optima.md). Keys and defaults: tau 10,
                           # fraction 0.1 (of n_append, per iteration), rounds 5, k 0 (0: fraction *
                           # n_append / rounds per round), epochs 30, learning_rate 1e-5, starts 0
                           # (0: 128 per parameter), steps 400, fit_window 100, start_iteration 0;
                           # failure_policy skip (failure_probes 3, failure_radius 0.001,
                           # failure_penalty 1000, failure_training_fraction 0.1); chain false,
                           # posterior 0 (posterior_tolerance 2, posterior_rounds 2)
```

### Model Architecture
```yaml
model:
  n_layers: 5               # Number of hidden layers
  n_neurons: 512            # Neurons per hidden layer
  activation: alsing        # 'alsing' or TensorFlow activations
```

### Training Configuration
```yaml
training:
  n_epochs: 5000            # Maximum training epochs
  batch_size: 128           # Training batch size
  loss: msre                # 'msre' or TensorFlow loss functions
  sigma_level: 3            # MSRE transition scale (σ)
  learning_rate: 0.0001     # Adam optimiser learning rate
  validation_split: 0.1     # Validation split fraction, drawn afresh at every training. With warm_start the
                            # validation loss is therefore not a held-out loss
  patience: 250             # Early stopping patience (epochs)
  # augment_schedule_factor: 3  # Only after `client.py augment`: the first training with the added
                            # points multiplies lr_patience, lr_grace and patience by this (1: off)
  # msre_ess_floor: 0.1     # Raises the msre constant while the weights collapse (0: off)
  # warm_start: true        # Start each network from the previous one's weights, with
  # lr_schedule: none       # Or plateau, with lr_factor 0.3, lr_patience 75, lr_min 1e-6, lr_grace 100;
                            # patience is then unused
```

### Sampling Configuration
```yaml
sampling:
  sampler: aies             # aies (only sampler currently implemented)
  temperature: 7.0          # MCMC temperature T_MCMC
  n_walkers: 216            # Number of AIES walkers (even, at least twice the parameters)
  n_steps: 100000           # MCMC steps
  burn_in: 5000             # Burn-in steps to discard
  # thin: 1                 # Keep every thin-th step, as it is written
  # chunk_size: 5000        # Steps per sampler chunk
```

### Convergence Configuration
```yaml
convergence:
  metric: gaussian_posterior_drift  # Convergence metric
  threshold: 0.01                   # Convergence threshold
  max_iterations: 20                # Maximum iterations (prevents infinite loops)
  # consecutive: 2                  # tempered_batch_scatter only (below)
```

`metric: tempered_batch_scatter` (opt-in) converges on true evaluations instead of on the
drift between chains, which on real likelihoods never comes near its threshold: the loop has
converged when the scatter of network minus truth at the tempered batch was below
`threshold` × `sampling.temperature` (1.2 when left out) for `consecutive` networks running
(2). A run with `acquisition.boost` then begins its boosts instead of ending. With `-i` the
verdict is printed and nothing follows from it.
[docs/tempered_convergence.md](docs/tempered_convergence.md) has the definition and the
measurements behind 1.2.

Every iteration also prints the error of its network at the points it then acquires, which
that network had not been trained on:

```text
   network 3 at its tempered batch, which it had not seen: network minus truth rms 5.12 (mean -0.717, scatter 5.07, largest 33.6) at 400 points
```

The same numbers are kept in `metrics.json` (`held_out_rms`, `held_out_bias`, `held_out_sd`,
`held_out_largest`, `held_out_n` of each acquisition) and in two columns of `metrics.md`. The
batch comes from the tempered chain, so this is the error some T n / 2 below the peak, not at
the posterior; a boost iteration reports the error at posterior samples instead
([docs/boost.md](docs/boost.md)). With the default metric it is a measurement only;
`metric: tempered_batch_scatter` above is the criterion that reads it.

## Output Structure

```text
results/YYYYMMDD_HHMMSS_run_name/
├── likelihood_input/         # Copy of wrapped likelihood input file
├── training_data/            # CSV datasets (x, loglkl) per iteration
├── trained_models/           # Keras models (.keras format)
├── training_history/         # Training histories as .csv files (loss, val_loss)
├── convergence_stats/        # Saved chain summaries per iteration
├── derived_data/             # Derived parameters per iteration, their emulator and reports (docs/derived_parameters.md)
├── audit/                    # With acquisition.audit: the true values at the networks' own maxima (docs/audit_optima.md)
├── boost/                    # With acquisition.boost: each boost's points and report, the regions, start.json (docs/boost.md)
├── augmentation/             # Profile points added by client.py augment (docs/augment_training.md)
├── phases/                   # One record per finished phase, with the hashes of its files; checked at every start
├── interrupted/              # What an unfinished phase left, set aside when the run was continued
├── objective_calls/          # Journal of every true evaluation, by stage, and summary.json
├── objective_identity.json   # What decides the objective's values; a continued run must match it
├── .client.lock              # Held by the job that owns the run
├── evidence/                 # client.py evidence: ln Z per network and the nested-sampling chain (docs/evidence.md)
├── modes/                    # client.py modes: the modes of a network (docs/boost_modes_mixing.md)
├── posterior_chains/         # GetDist chains with derived columns (client.py derived --chain)
├── benchmark_chains/         # Benchmark MCMC chains (from benchmark.py)
├── benchmark_figures/        # Triangle plots and visualizations (from benchmark.py)
├── benchmark_results/        # Diagnostics logs (from benchmark.py)
├── metrics.md                # Human-readable metrics report
├── metrics.json              # Structured metrics (used for continuation)
├── metadata.json             # Surrogate metadata (names, labels, scales, bounds)
└── example.yaml              # Copy of the input configuration file
```

When continuing from an existing run, the original YAML configuration is preserved.

## Performance

For cosmological likelihoods with ~30 varying parameters, CLiENT typically requires fewer than 2×10⁴ function evaluations to produce credible intervals within better than 0.1σ of those obtained using the true likelihood, while maintaining single-point emulator precision better than Δχ² ~ 0.5 across relevant regions in parameter space.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CLiENT in your publications, please cite [arXiv:2512.17509](https://arxiv.org/abs/2512.17509)

## Contact

For questions, issues, or contributions, please open an issue.

## Create profiles from an existing network

The profile workflow uses the saved network directly by default and writes
surrogate, true-at-surrogate and residual-Newton true profiles. Augmentation
and retraining remain opt-in:

```sh
python client.py profiles results/MY_RUN --iteration 5 \
  --params omega_b,omega_cdm,theta_s_100,logA,n_s,tau_reio \
  --points 15 --sigma-range 3.5 --workers 4 \
  --output results/MY_RUN_profiles
```

Use `--dry-run` for a no-evaluation preflight and `--resume` to reuse completed
work. `--workers` may be up to the host's CPU count; the default `chain_length: 2`
keeps 30-40 of them busy even with one or two profiles (see
[Many workers](docs/create_profiles.md#many-workers)). See the [detailed profile guide](docs/create_profiles.md) and
[example configuration](input/profiles_example.yaml) for explicit ranges,
training settings, objective conventions, evaluation budgets and outputs.
Do not combine this managed-worker command with `mpirun`.

To create **one 2D profile** per run, use `--pair name1,name2`; `--points` then
means nodes per axis:

```sh
python client.py profiles results/MY_RUN --iteration 5 \
  --pair omega_b,omega_cdm --points 21 \
  --sigma-range 3.5 --workers 4 --output results/MY_RUN_2d
```

This uses the saved checkpoint without new training data and produces three
surfaces on a 21×21 grid plus conventional joint contour plots. To opt into
augmentation, request a positive `--add-points N` or set `retrain: true` with
an explicit added-point budget. Use the
[2D example configuration](input/profiles_2d_example.yaml) to change explicit
ranges, training schedules or evaluation budgets. The
[2D profile guide](docs/create_profiles_2d.md) covers preflight, resume, outputs,
mathematical details and limitations.

**Profiles in derived parameters** (H0 when θ_s is sampled, σ8, S8, Ω_m, ...) need a run
with recorded derived values and a derived-parameter emulator. `derived-view` writes a
copy of the run in which a sampled pivot is replaced by the derived parameter. Both
commands above then work on it unchanged, 1D and 2D:

```sh
python client.py derived-view results/MY_RUN --params H0 -o results/MY_RUN_view_H0
python client.py profiles results/MY_RUN_view_H0 --params H0 --workers 4 --output results/MY_RUN_H0
python client.py derived-view --nodes results/MY_RUN_H0 --workers 4   # exact H0 at the nodes
```

See [docs/derived_parameters.md](docs/derived_parameters.md#profiles-in-derived-parameters).
