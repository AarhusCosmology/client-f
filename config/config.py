from dataclasses import dataclass, fields
from typing import get_args
import math

import yaml


def _validate_mapping(cls, data):
    """Validate before coercion; scientific-notation YAML strings remain numeric."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    schema = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(schema)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}; check spelling or migrate obsolete settings")
    nonnegative = {'n_inject', 'n_layers', 'burn_in', 'start_iteration', 'starts', 'k',
                   'lr_grace', 'patience', 'seed'}
    positive = {'n_samples', 'n_append', 'n_neighbors', 'target_temperature', 'pool_factor',
                'batch_size', 'n_neurons', 'learning_rate', 'sigma_level', 'n_epochs',
                'temperature', 'n_walkers', 'n_steps', 'chunk_size',
                'thin', 'lr_patience', 'lr_min', 'n_sigma', 'max_iterations',
                'consecutive', 'threshold', 'n', 'walkers', 'cool_steps', 'steps', 'components',
                'barrier', 'candidate_barrier', 'tau', 'fraction', 'rounds', 'epochs', 'fit_window'}
    for name, value in data.items():
        typ = schema[name].type
        kinds = get_args(typ) or (typ,)
        if value is None:
            if type(None) not in kinds and schema[name].default is not None:
                raise ValueError(f"{cls.__name__}.{name} cannot be null")
            continue
        if bool in kinds and type(value) is not bool:
            raise ValueError(f"{cls.__name__}.{name} must be a YAML boolean, true or false")
        if int in kinds and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{cls.__name__}.{name} must be an integer")
        if str in kinds and isinstance(value, str):
            continue                      # a word the setting allows; its class checks which
        if float in kinds or int in kinds:
            if isinstance(value, bool):
                raise ValueError(f"{cls.__name__}.{name} must be numeric, not boolean")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{cls.__name__}.{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{cls.__name__}.{name} must be finite")
            if name in positive and number <= 0 or name in nonnegative and number < 0:
                raise ValueError(f"{cls.__name__}.{name} is outside its allowed range")
            if name in {'validation_split', 'lr_factor'} and not 0 < number < 1:
                raise ValueError(f"{cls.__name__}.{name} must be between zero and one")
            if name == 'min_share' and not 0 <= number < 1:       # zero: every group of walkers is a mode
                raise ValueError(f"{cls.__name__}.{name} must be at least zero and below one")
            if name in {'msre_ess_floor', 'failure_training_fraction'} and not 0 <= number <= 1:
                raise ValueError(f"{cls.__name__}.{name} must be between zero and one inclusive")
    choices = {'wrapper': {'cobaya', 'montepython', 'derived_coordinates'},
               'sampler': {'aies'}, 'metric': {'gaussian_posterior_drift', 'tempered_batch_scatter'},
               'sampling_strategy': {'lhs', 'sobol', 'uniform'}}
    for name, allowed in choices.items():
        if name in data and data[name] not in allowed:
            raise ValueError(f"{cls.__name__}.{name}: expected one of {sorted(allowed)}")
    if 'identity_files' in data and data['identity_files'] is not None:
        if not isinstance(data['identity_files'], list) or not all(isinstance(p, str) for p in data['identity_files']):
            raise ValueError('likelihood.identity_files must be a list of file paths')
    if 'lr_schedule' in data and data['lr_schedule'] not in ('none', 'plateau'):
        raise ValueError('lr_schedule must be none or plateau')
    if cls.__name__ == 'SamplingConfig':
        if data.get('n_walkers', 2) % 2:
            raise ValueError('sampling.n_walkers must be even')
        if data.get('thin', 1) > data.get('n_steps', 1):
            raise ValueError('sampling.thin must not exceed n_steps')


@dataclass(frozen=True)
class LikelihoodConfig:
    wrapper: str
    input: str
    identity_files: list[str] | None = None  # extra objective code/data files to hash on resume

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        return cls(
            wrapper=str(d["wrapper"]),
            input=str(d["input"]),
            identity_files=d.get("identity_files"),
        )


@dataclass(frozen=True)
class PriorConfig:
    n_samples: int
    sampling_strategy: str
    n_sigma: float | None
    # Extra initial points drawn from the Gaussian implied by the quadratic response
    # surface, at the acquisition temperature; each costs one true likelihood call.
    # Default off: it measurably speeds convergence but did not improve the credible
    # metric on the analytic targets. All real runs have used it.
    n_inject: int = 0

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        n_sigma = d.get("n_sigma")
        return cls(
            n_samples=int(d["n_samples"]),
            sampling_strategy=str(d["sampling_strategy"]),
            n_sigma=None if n_sigma is None else float(n_sigma),
            n_inject=int(d.get("n_inject", 0)),
        )


@dataclass(frozen=True)
class AuditConfig:
    """acquisition.audit: true evaluations at the network's own maxima, after every training
    and before the acquisition chain (training/audit.py, docs/audit_optima.md). Absent (the
    default): no audit, and the loop is exactly as it was."""

    tau: float = 10.0  # a maximum is flagged when predicted this far above the best true value
    fraction: float = 0.1  # true evaluations per iteration: at most this fraction of the batch (n_append)
    rounds: int = 5  # at most this many rounds per iteration
    k: int = 0  # true evaluations per round (the highest flagged maxima); 0: fraction * n_append / rounds
    epochs: int = 30  # fine-tune after each round
    learning_rate: float = 1e-5
    starts: int = 0  # uniform starts of the search, and as many random vertices; 0: 128 per parameter
    steps: int = 400  # ascent steps of the search
    fit_window: float = 100.0  # the fit check uses points within this of the best true value
    start_iteration: int = 0  # first iteration whose network is audited
    failure_policy: str = "skip"  # or "penalize": bounded probes then a training-only constraint
    failure_probes: int = 3  # strictly fewer than five, no recursive retries
    failure_radius: float = 0.001  # per-coordinate prior-box fraction
    failure_penalty: float = 1000.0  # log-units below the best genuine value
    failure_training_fraction: float = 0.1  # minimum synthetic rows / genuine training rows
    chain: bool = False  # in a boost iteration, also look where the cooled walkers of the tempered chain climb to
    # Opt-in, with `chain`: after the search for maxima, in a boost iteration, this many of the
    # cooled T = 1 walkers of the network's tempered chain (where the boost's chains start)
    # are evaluated. When the network is above the truth there by more than
    # posterior_tolerance on average, it is fine-tuned on them and the check is repeated, at
    # most posterior_rounds times. A network can be wrong where its posterior is without a
    # maximum above the best: in the axion run, plateaus some 10 too high drew a third of a
    # boost (docs/audit_optima.md, "The posterior pilot"). 0: no pilot.
    posterior: int = 0
    posterior_tolerance: float = 2.0  # log-units, the network's mean excess over the truth there
    posterior_rounds: int = 2

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        _validate_mapping(cls, d)
        defaults = cls()
        if "failure_probes" in d and (isinstance(d["failure_probes"], bool)
                                      or not isinstance(d["failure_probes"], int)):
            raise ValueError("acquisition.audit.failure_probes must be an integer from 1 to 4")
        out = cls(**{k: type(getattr(defaults, k))(v) for k, v in d.items()})
        if out.posterior < 0 or out.posterior_tolerance <= 0 or out.posterior_rounds < 1:
            raise ValueError("acquisition.audit: posterior >= 0, posterior_tolerance > 0, posterior_rounds >= 1")
        if out.posterior and not out.chain:
            raise ValueError("acquisition.audit.posterior needs acquisition.audit.chain: true "
                             "(the walkers it evaluates are the chain search's)")
        if (out.failure_policy not in ("skip", "penalize") or not 1 <= out.failure_probes <= 4
                or not math.isfinite(out.failure_radius) or not 0 < out.failure_radius <= .01
                or not math.isfinite(out.failure_penalty) or out.failure_penalty <= 0
                or not math.isfinite(out.failure_training_fraction)
                or not 0 <= out.failure_training_fraction <= 1):
            raise ValueError("acquisition.audit: failure_policy skip/penalize; failure_probes 1..4; "
                             "0 < failure_radius <= 0.01; finite failure_penalty > 0; "
                             "0 <= failure_training_fraction <= 1")
        if (out.tau <= 0 or out.fraction <= 0 or out.k < 0 or out.rounds < 1 or out.epochs < 1
                or out.learning_rate <= 0 or out.starts < 0 or out.steps < 1
                or out.fit_window <= 0 or out.start_iteration < 0):
            raise ValueError(
                "acquisition.audit: tau, fraction, learning_rate, fit_window > 0; rounds, "
                "epochs, steps >= 1; k, starts, start_iteration >= 0")
        return out

    def per_round(self, n_append):
        """True evaluations per round for a batch of n_append points."""
        return self.k or max(1, round(self.fraction * n_append / self.rounds))


@dataclass(frozen=True)
class BoostConfig:
    """acquisition.boost: from `start_iteration` on, an iteration's new training points are `n`
    posterior samples of the network (T = 1) in place of the tempered batch, the same number
    from every mode (dataset/boost.py, sampling/modes.py, docs/boost.md). Absent (the default):
    no boost, and the loop is exactly as it was."""

    n: int  # true evaluations per boost, shared equally among the modes found
    # The first iteration whose acquisition is a boost. With convergence.metric:
    # tempered_batch_scatter and no -i, the boosts begin as soon as the ordinary loop has
    # converged, and this is the latest iteration at which they begin.
    start_iteration: int
    barrier: float = 4.0  # log-units: maxima joined by a straight path that dips less are one mode
    min_share: float = 0.02  # a mode needs this share of the cooled walkers
    walkers: int = 512  # ensemble for cooling and for the chain of each mode
    cool_steps: int = 2000  # steps at each temperature on the way down
    steps: int = 10000  # steps of each mode's T = 1 chain (a fifth more as burn-in)
    components: int = 6  # Gaussians in the mixture that each mode's evidence is bridged to
    density_filter: bool = False  # T=1 density-deficit selection
    pool_factor: int = 20  # disjoint candidate pool for filtered boosts
    # Opt-in, both off by default (docs/boost.md, "Background and region").
    # "quadratic": from the first training that holds boost points, the network is a fixed
    # concave quadratic plus a network trained on the difference; that first training starts
    # from random weights, later ones warm-start with the quadratic unchanged.
    background: str | None = None
    # A number > 1 (2 is the measured setting): chains, the boost's own sampling and the
    # audit's searches see the network only inside the region the boost points reach, this
    # many times as far as the farthest of them (sampling/region.py). null: no region.
    region: float | None = None
    # Opt-in, with `region`. A number in (0, 1]: each cloud of the region reaches as far as this
    # quantile of its points' distances from its centre instead of the farthest point. In the
    # axion run one far boost point doubled the reach (9.7 to 18.2) and let a false peak in;
    # the 99.9% point stayed at 7.3 to 7.6. null: the farthest.
    region_quantile: float | None = None
    # Opt-in, with `region`. A number >= 1: from the first training after the boosts' start on,
    # the network is trained only on the points inside an ellipsoid, this many times as far as
    # the farthest of the points within T * ndim of the best at that training (the points the
    # quadratic background is fitted to; training/domain.py). The ellipsoid is fixed from then
    # on (boost/train_domain_it_N.npz); the other points stay in the training set's files. Far
    # points set the network's output scale: 52 280 log-units with them, 22 without, in the
    # axion run, whose oscillations are a few units. null: every point.
    train_domain: float | None = None
    # Opt-in. A number in (0, 1), 0.1 the tested setting: the modes are decided by measured
    # mixing instead of the fixed `barrier`. Maxima are grouped into candidate basins with
    # `candidate_barrier`, a free T = 1 chain of `steps` is run, and two basins are one mode
    # when the memory of which one a walker started in is at most this after the chain
    # (sampling/modes.py, docs/boost.md). null: the fixed barrier.
    mixing: float | None = None
    candidate_barrier: float = 1.0  # log-units; dips under it are the network's own wiggles
    # Opt-in. A number of log-units: a mode whose highest maximum lies more than this below the
    # highest mode's is left out of the boost before its chain is run (sampling/modes.py). A
    # mode that deep carries no posterior weight, and the boost would give it as many points as
    # the posterior: in the axion run a first boost, searched over the whole box, spent half its
    # points in a "mode" 1030 below, where the likelihood refused every one. "auto": T * ndim / 2
    # with T = sampling.temperature (84 for the axion run's 24 parameters). null: every mode.
    mode_depth: float | str | None = None
    # Opt-in. A weight in (0, 1]: a mode whose weight (the network's, from the per-mode
    # evidences) is at least this gets the same share of a boost as the others; a lighter
    # mode gets that share times its weight over this floor (dataset/boost.py, mode_shares).
    # With min_share 0.005 the H0WC axion run's network 8 found two modes of 1.3% and 1.1%,
    # one of them false, and the equal shares gave them 85% of the boost (30 October 2026);
    # with 0.1 they would have had 13% and 11% of an equal share. null: every mode the same
    # share.
    balance_floor: float | None = None
    # Opt-in: the run ends once this many boosts have been acquired and the network trained on
    # them has been sampled. For a run that hands over to its boosts by itself
    # (convergence.metric: tempered_batch_scatter), whose length is not known beforehand.
    # Ignored when the number of iterations is given on the command line (-i). null: the
    # run's last iteration decides, as before.
    boosts: int | None = None

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        _validate_mapping(cls, d)
        missing = {"n", "start_iteration"} - set(d)
        if missing:
            raise ValueError(f"acquisition.boost needs {sorted(missing)}")
        kinds = {name: field.type for name, field in cls.__dataclass_fields__.items()}
        if "density_filter" in d and not isinstance(d["density_filter"], bool):
            raise ValueError("acquisition.boost.density_filter must be a boolean")
        if d.get("background") not in (None, "quadratic"):
            raise ValueError("acquisition.boost.background must be 'quadratic' or absent")
        if d.get("region") is not None and (isinstance(d["region"], bool) or not float(d["region"]) > 1):
            raise ValueError("acquisition.boost.region must be a number above 1 or absent")
        q = d.get("region_quantile")
        if q is not None and (isinstance(q, bool) or not 0 < float(q) <= 1):
            raise ValueError("acquisition.boost.region_quantile must be a number in (0, 1] or absent")
        edge = d.get("train_domain")
        if edge is not None and (isinstance(edge, bool) or not float(edge) >= 1):
            raise ValueError("acquisition.boost.train_domain must be a number of at least 1 or absent")
        if (q is not None or edge is not None) and d.get("region") is None:
            # Without the region every chain and search sees the network over the whole box,
            # where a network trained on part of it has nothing to hold it down.
            raise ValueError("acquisition.boost.region_quantile and train_domain need acquisition.boost.region")
        if d.get("mixing") is not None and (isinstance(d["mixing"], bool) or not 0 < float(d["mixing"]) < 1):
            raise ValueError("acquisition.boost.mixing must be a number between 0 and 1 or absent")
        depth = d.get("mode_depth")
        if depth is not None and depth != "auto" and (isinstance(depth, (bool, str)) or not float(depth) > 0):
            raise ValueError("acquisition.boost.mode_depth must be a positive number, auto, or absent")
        floor = d.get("balance_floor")
        if floor is not None and (isinstance(floor, (bool, str)) or not 0 < float(floor) <= 1):
            raise ValueError("acquisition.boost.balance_floor must be a number in (0, 1] or absent")
        if d.get("boosts") is not None and (isinstance(d["boosts"], bool) or int(d["boosts"]) < 1):
            raise ValueError("acquisition.boost.boosts must be an integer of at least 1 or absent")
        special = {"background": lambda v: v, "region": lambda v: None if v is None else float(v),
                   "region_quantile": lambda v: None if v is None else float(v),
                   "train_domain": lambda v: None if v is None else float(v),
                   "mode_depth": lambda v: None if v is None else v if v == "auto" else float(v),
                   "mixing": lambda v: None if v is None else float(v),
                   "balance_floor": lambda v: None if v is None else float(v),
                   "boosts": lambda v: None if v is None else int(v)}
        out = cls(**{k: (special[k](v) if k in special else bool(v) if kinds[k] in (bool, "bool") else
                        float(v) if kinds[k] in (float, "float") else int(v)) for k, v in d.items()})
        if (out.n < 1 or out.start_iteration < 0 or out.barrier <= 0 or not 0 <= out.min_share < 1
                or out.walkers < 4 or out.walkers % 2 or out.cool_steps < 1 or out.steps < 100
                or out.components < 1 or out.pool_factor < 1 or out.candidate_barrier <= 0):
            raise ValueError(
                "acquisition.boost: n, cool_steps, components, pool_factor >= 1; start_iteration >= 0; "
                "barrier, candidate_barrier > 0; 0 <= min_share < 1; walkers even and >= 4; steps >= 100")
        return out


@dataclass(frozen=True)
class AcquisitionConfig:
    n_append: int
    n_neighbors: int
    target_temperature: float
    pool_factor: int

    # Points re-drawn from the quadratic fit EVERY iteration, using the whole
    # accumulated training set rather than the initial design alone. 0 (the default)
    # leaves the loop exactly as it was. Distinct from prior.n_inject, which injects once
    # after the initial design: this one tracks the fit as it improves, and costs
    # n_inject evaluations per iteration rather than once.
    n_inject: int = 0

    # Optional `acquisition.audit:` mapping (an empty mapping takes every default).
    audit: AuditConfig | None = None

    # Optional `acquisition.boost:` mapping (n and start_iteration are required).
    boost: BoostConfig | None = None

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        return cls(
            n_append=int(d["n_append"]),
            n_neighbors=int(d["n_neighbors"]),
            target_temperature=float(d["target_temperature"]),
            n_inject=int(d.get("n_inject", 0)),
            pool_factor=int(d["pool_factor"]),
            audit=cls._section(AuditConfig, d, "audit"),
            boost=cls._section(BoostConfig, d, "boost"),
        )

    @staticmethod
    def _section(section, d, name):
        """An opt-in section: absent or false is off; null, true or {} is every default.
        (`audit: false` used to switch the audit on with its defaults.)"""
        value = d.get(name, False)
        if value is False:
            return None
        return section.from_dict({} if value is None or value is True else value)


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_neurons: int
    activation: str

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        return cls(
            n_layers=int(d["n_layers"]),
            n_neurons=int(d["n_neurons"]),
            activation=str(d["activation"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    loss: str
    sigma_level: float
    n_epochs: int
    batch_size: int
    validation_split: float
    patience: int
    # Floor on the msre effective sample size, as a fraction of the training set. The
    # default is ON: without it the loss collapses onto one or two points whenever the
    # initial design misses the peak by much, which on a 29D target scored 260x the
    # credible metric's noise floor against 8x with the floor active. It anneals itself
    # away once the training set fills in, so it costs nothing where it is not needed.
    # Set 0.0 to recover the untouched loss exactly.
    msre_ess_floor: float = 0.1

    # Learning-rate annealing. 'plateau' reduces the rate by lr_factor
    # whenever val_loss stalls for lr_patience epochs, down to lr_min, and then stops
    # lr_grace epochs later. 'none' restores the previous fixed-rate behaviour with the
    # plain `patience` EarlyStopping.
    #
    # Absent from a YAML the key is 'none' (the field's default below, and from_dict's);
    # some historical comparison YAMLs set 'plateau'. It is an opt-in experiment,
    # not a generally recommended setting: sparse 58D designs later showed worse
    # posterior accuracy. The following measured gain applies to the dense 31D
    # ridge experiment only, at a matched epoch budget against a fixed 1e-4:
    #
    #     validation loss   2.1x lower      credible metric   2.4x better
    #     median dCM        0.0135 vs 0.0322 (both seeds beat both fixed seeds)
    #     better on 27 of 31 parameters, 6 resolvable wins, 0 resolvable losses
    #
    # The mechanism is in the loss curve. At a fixed rate the coefficient of variation
    # over the 250 epochs before the best epoch is 124% and only 2 of those epochs come
    # within 10% of the best -- the optimiser is sampling a wide noise ball and
    # restore_best_weights catches a lucky dip. Annealed: CV 24%, and 97 epochs within
    # 10%. The weights are a settled minimum rather than a fortunate draw, which also
    # removes an uncontrolled source of run-to-run variance.
    #
    # `patience` is UNUSED when lr_schedule is 'plateau' -- see
    # training/training.py::StopAfterScheduleExhausted for why a plain patience cannot
    # terminate an annealed run at all.
    #
    # ON THE VALUE OF lr_patience. A 2x2 factorial over lr_patience 25/50 and lr_min
    # 1e-6/1e-7 (seed 42; the second seed was not run) found:
    #
    #     lr_patience  lr_min   epochs  min_lr at   val_loss
    #              50    1e-6      472        372   7.47e-05
    #              25    1e-6      305        205   1.95e-03   <- 26x WORSE
    #              50    1e-7      573        473   7.31e-05      (+2% for +101 epochs)
    #              25    1e-7      356        256   1.99e-03   <- 26x WORSE
    #
    # lr_patience 25 is catastrophic: it anneals so fast that the rate is frozen at the
    # floor by epoch 205, before the optimiser has descended, and the 305 epochs are cheap
    # only because the run is dead. The failure mode is therefore annealing TOO EARLY, and
    # 75 is set here to move further from it. lr_min 1e-7 was rejected: 2% of loss for 21%
    # more epochs, and a 2x loss spread in this family moved the credible metric by 9%,
    # far below its 0.025 floor.
    #
    # 75 IS AN EXTRAPOLATION, NOT A MEASUREMENT. The factorial establishes 25 << 50; it
    # does not establish 75 > 50. It is the cautious side of a known failure direction.
    # If it is ever measured, record the result here.
    # DEFAULT REVERTED TO 'none' ON 16 SEPTEMBER 2026 -- constant learning rate.
    #
    # Annealing remains available and is still 2.1x better on validation loss at a
    # matched budget on a DENSE training set (the section 6.2 result, measured on
    # data_it_15.csv with 32,100 points). But it is unsafe on a SPARSE one, and the
    # initial design of any high-dimensional run is sparse by construction.
    #
    # Measured on 58D iteration 0 (2,100 points, c inflated 1008x), three seeds each:
    #
    #     fixed 1e-4   1.4977e-02   1.8308e-02   1.4500e-02     spread 1.26x
    #     annealed     3.2887e-02   1.3641e-01   3.8296e-02     spread 4.15x
    #
    # The two sets do not overlap: the worst fixed run beats the best annealed run by
    # 1.80x, complete separation at one-sided p = 0.05. Annealing is also four times
    # less predictable, and its worst seed landed 9.1x above the fixed-rate result with
    # its first reduction at epoch 86.
    #
    # WHY. On sparse data val_loss is a wide noise ball -- at 58D iteration 0 it swings
    # a factor of two between neighbouring epochs (3.29e-02 at 520, 6.28e-02 at 580).
    # The plateau detector reads that as convergence and reduces the rate, after which
    # improvement is slower still, so the next plateau fires sooner and the rate
    # collapses. The best epoch is then a lucky dip caught by restore_best_weights
    # rather than a settled minimum.
    #
    # THE CONSEQUENCE IS NOT LOCAL. A 2.2x worse iteration-0 emulator produced an
    # acquisition that DISPERSED instead of contracting (selected-point spread 1.32
    # against the baseline's 0.18), which produced a reproducible iteration-1 training
    # failure, which contaminated every later training set. By iteration 10 the 58D
    # posterior sat entirely outside the truth's 95% contour in every cosmological
    # parameter, with 24 of 58 parameters above dCM 3.5 against the fixed-rate arm's 7.
    # An offline replay cleared the walker count, thinning, chain length and the batched
    # selector; swapping only the iteration-0 emulator moved the spread 1.203 -> 0.108.
    #
    # Set 'plateau' deliberately, on a target whose initial design is known to be dense.
    lr_schedule: str = "none"
    lr_factor: float = 0.3
    lr_patience: int = 75
    lr_min: float = 1e-6
    lr_grace: int = 100

    # Warm-starting: initialise iteration N's network from iteration N-1's trained
    # weights instead of at random. Enabled by default. The measurements below were made
    # before the scaler compensation (25 September 2026) was added.
    #
    # The scalers are NOT inherited. build_model adapts an input Normalization to the
    # current inputs and builds a TargetDenormalization from the current targets; both
    # are data statistics that move as the training set grows. Only the 22 trainable
    # weight tensors were transferred, so predictions changed when the scalers
    # changed. model.network.transfer_warm_start now compensates for that. Carrying the
    # scalers over instead ("warm-continue") was measured and is NOT better: it starts
    # far closer (initial val_loss 4.97e-04 against cold's converged 3.96e-04 at 31D
    # iteration 10) but finishes no better, and it accumulates stale statistics.
    #
    # MEASURED on the 31D ridge, cold against warm-transfer, same data and split:
    #
    #     iteration   epochs to reach cold's best val_loss      epochs run
    #            1    164 vs 1022  (6.2x fewer)                 538 vs 1027
    #            5    251 vs 1038  (4.1x fewer)                 425 vs 1044
    #           10      2 vs  722  (warm beat cold's FINAL      402 vs  722
    #                               result in two epochs)
    #
    # Credible metric at iteration 10: median 0.0187 against cold's 0.0198 (a null --
    # both sit at the ~0.018 measurement floor), max excluding x_30 0.0431 against
    # 0.1066. So: equal-or-better accuracy for roughly half the epochs. At iteration 1
    # the dCM comparison was mixed and every arm was far from converged, so the safe
    # claim is "as accurate, much cheaper", not "more accurate".
    #
    # It also reduces exposure to a schedule that anneals too early, because a warm
    # network does not need the long from-scratch descent in the first place.
    #
    # DEFAULT TURNED ON 17 SEPTEMBER 2026, on two full runs each matched to its own
    # baseline on every setting except this one:
    #
    #     target            epochs run           it15 median dCM      it15 max
    #     58D CLOE +/-10s   7,852 vs 23,187      0.2318 vs 0.2551     0.6523 vs 1.0598
    #     31D ridge         8,539 vs 21,724      0.0081 vs 0.0209     0.0565 vs 0.0754
    #
    # i.e. 2.95x and 2.54x fewer epochs at equal or better accuracy. On 58D iteration 10
    # the worst parameter came in at 1.1704 against the baseline's 9.4522, with zero of
    # 58 parameters above dCM 3.5 where the baseline had seven -- roughly the baseline's
    # ITERATION-15 quality reached at iteration 10.
    #
    # Iteration 0 cannot be warm started, which makes it a free control: with nothing to
    # inherit it must reproduce the baseline exactly, and it did on both runs to four
    # digits (58D 2071 epochs at 1.4977e-02; 31D 1697 at 6.8004e-04). That is what
    # confirms warm_start was the only active variable in each comparison.
    #
    # THE OBJECTION DID NOT MATERIALISE. c re-solves every iteration, so inherited
    # weights are optimal for a different loss; the feared signature was val_loss
    # degrading as the weights age. It improved instead. TargetDenormalization is
    # rebuilt from the current target distribution each iteration, so the output is
    # rescaled before a single gradient step; what the weights carry is the SHAPE of the
    # function, which changes slowly.
    #
    # KNOWN BLEMISH: the 31D run is not monotone -- its iteration-10 median (0.0565) is
    # worse than both its own iteration 5 and the baseline's iteration 10, though both
    # sit near the ~0.018-0.026 measurement floor and its maxima are better at both
    # iterations. One seed per run. Set warm_start: false to restore cold starts.
    warm_start: bool = True
    # The first training that includes points added by `client.py augment`
    # (dataset/augment.py, docs/augment_training.md) runs with lr_patience, lr_grace and
    # patience multiplied by this factor. Every later iteration uses the ordinary schedule
    # again. 1 switches it off. It has no effect in a run that is never augmented.
    #
    # WHY. Warm starting works because the training set changes little from one iteration
    # to the next, so the inherited network is already close. An injection of profile points
    # is not such a change: on SINU (Planck + DESI DR2) it raised the best known value by 24
    # log-units and opened a region the network had rated 123 log-units too low. The
    # ordinary schedule then declares a plateau long before the new region is fitted.
    #
    # MEASURED on that run, iteration 11
    # retrained on the same 25,496 points, warm start, c = 19.2, only the schedule changed:
    #
    #     factor   epochs   misfit of the summit's   error on the true    strong mode's
    #                       own training points      profile ridge (rms)  posterior weight
    #          1      405   +5.7 log-units           6.7 log-units         1.6% (it10 network)
    #          3     2513   +0.9                     3.0                  18.4%
    #
    # With the short schedule the network has the strong-coupling minimum 3.0 in chi^2 above
    # the weak one (the true profile has them level), which is what pushed the mode out to
    # the 99% contour. Lowering c instead (19.2 -> 3.4) also fits the summit but erodes the
    # barrier in the network's own profile from 12.9 to 4.2 (truth 13.4). A cold start fits
    # best of all (0.4 log-units on the ridge) but grew a runaway optimum 0.5 box widths from
    # any data that the acquisition chain then goes to with every sample.
    #
    # 3 IS THE ONLY VALUE MEASURED, and it was measured five iterations after the injection,
    # not on the injection's own first training, where this option applies it. It costs that
    # one training about six times its usual epochs (64 minutes against 7 here).
    augment_schedule_factor: float = 3.0

    # Several trainings of each network, the best kept (training/members.py, docs/members.md).
    # A feature of the boosts: it acts from the first training on boost points (the network after
    # the boosts' start) on, and every network of the tempered phase is trained once; it needs
    # acquisition.boost (a run without one is refused).
    # members: K > 1 trains every such network K times on the same training set with the same
    # settings: rank 0 as before (warm-started when warm_start is on) and K - 1 times from
    # random weights, each with its own seed, on ranks that would otherwise wait for the next
    # evaluation (the members left over when the run has fewer ranks are trained on rank 0 after
    # its own). selection_share of the points acquired since the previous training is held out
    # from all of them and decides: the warm network is kept unless a member from random weights
    # has a selection loss lower by more than member_margin (a fraction), otherwise the lowest
    # is kept. member_threads: TensorFlow threads of a member (default TF_NUM_INTRAOP_THREADS, else 4).
    #
    # WHY. A training ends either resolving the fine structure of the likelihood or on a
    # plateau where it is smoothed out, and which one depends on the start: on the saved
    # training sets of networks 13 and 14 of an axion monodromy run, one in three and two in
    # three random starts resolved it (error 0.6-1.0 at posterior points they had not seen,
    # against 3.9-4.0 for the others); a longer or more patient schedule did not
    # (docs/members.md). A warm start keeps the state it inherits.
    # members: 1 (the default) holds out nothing and asks no rank: the run is as before.
    members: int = 1
    selection_share: float = 0.1
    member_margin: float = 0.2
    member_threads: int | None = None
    # The choice between the members by true evaluations (training/members.py, "the pilot";
    # docs/members.md). member_pilot: n > 0 has every member draw n points of its own posterior
    # (T = 1, inside the network's region; member_pilot_steps steps of member_pilot_walkers walkers, at least 2 ndim),
    # evaluates the pool once with the true likelihood (it joins the training set) and keeps the
    # member with the smallest rms error over the pool's points that the truth or that member puts
    # within member_pilot_depth of the best (default: the msre scale 0.5 chi^2(sigma_level, ndim));
    # nothing is held out then. 0: the held-out selection set chooses (above).
    #
    # WHY. The selection loss misses false plateaus: on network 8 of branches of two axion
    # monodromy runs with members: 12, the member it kept was 35.6 and 3.2 too high on average at
    # the points of its own pilot chain, while the pilot's choice was within 0.2; the kept network's
    # boost then had a scatter of 4.0 against 2.1-2.2 for any cold member at the same points
    # (docs/members.md).
    member_pilot: int = 0
    member_pilot_steps: int = 20000
    member_pilot_walkers: int = 256
    member_pilot_depth: float | None = None

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        factor = float(d.get("augment_schedule_factor", 3.0))
        members = int(d.get("members", 1))
        if members < 1:
            raise ValueError("training.members must be at least 1 (1: one training per network, as before)")
        share = float(d.get("selection_share", 0.1))
        if not 0.0 < share < 1.0:
            raise ValueError("training.selection_share must lie between 0 and 1")
        margin = float(d.get("member_margin", 0.2))
        if not 0.0 <= margin < 1.0:
            raise ValueError("training.member_margin must lie in [0, 1)")
        threads = d.get("member_threads")
        if threads is not None and int(threads) < 1:
            raise ValueError("training.member_threads must be at least 1")
        pilot = int(d.get("member_pilot", 0))
        if pilot < 0:
            raise ValueError("training.member_pilot must be at least 0 (0: no pilot)")
        if pilot and members < 2:
            raise ValueError("training.member_pilot needs training.members of at least 2")
        pilot_steps, pilot_walkers = int(d.get("member_pilot_steps", 20000)), int(d.get("member_pilot_walkers", 256))
        if pilot_steps < 4 or pilot_walkers < 2:
            raise ValueError("training.member_pilot_steps must be at least 4 and member_pilot_walkers at least 2")
        pilot_depth = d.get("member_pilot_depth")
        if pilot_depth is not None and not float(pilot_depth) > 0:
            raise ValueError("training.member_pilot_depth must be positive")
        if not factor >= 1.0:
            raise ValueError("training.augment_schedule_factor must be at least 1 (1: the ordinary schedule)")
        return cls(
            augment_schedule_factor=factor,
            members=members,
            selection_share=share,
            member_margin=margin,
            member_threads=None if threads is None else int(threads),
            member_pilot=pilot,
            member_pilot_steps=pilot_steps,
            member_pilot_walkers=pilot_walkers,
            member_pilot_depth=None if pilot_depth is None else float(pilot_depth),
            learning_rate=float(d["learning_rate"]),
            loss=str(d["loss"]),
            sigma_level=float(d["sigma_level"]),
            n_epochs=int(d["n_epochs"]),
            batch_size=int(d["batch_size"]),
            validation_split=float(d["validation_split"]),
            patience=int(d["patience"]),
            msre_ess_floor=float(d.get("msre_ess_floor", 0.1)),
            warm_start=bool(d.get("warm_start", True)),
            lr_schedule=str(d.get("lr_schedule", "none")),
            lr_factor=float(d.get("lr_factor", 0.3)),
            lr_patience=int(d.get("lr_patience", 75)),
            lr_min=float(d.get("lr_min", 1e-6)),
            lr_grace=int(d.get("lr_grace", 100)),
        )


@dataclass(frozen=True)
class SamplingConfig:
    sampler: str
    temperature: float
    n_walkers: int
    burn_in: int
    n_steps: int
    chunk_size: int = 5000  # steps per compiled chain segment (memory, not results)

    # Keep every thin-th step, strided AT WRITE TIME, so a thinned chain never allocates
    # the unthinned one. sampling/aies.py has supported this since the preallocated-buffer
    # change but nothing reached it: client.py did not pass it and this field did not
    # exist. 1 (the default) is the previous behaviour exactly.
    #
    # It is what makes a larger n_walkers affordable. The chain buffer is
    # n_steps/thin x n_walkers x ndim x 4 bytes, so at 100k steps and 31D:
    #
    #     216 walkers, thin 1  ->  2.49 GB      432 walkers, thin 1  ->  4.99 GB
    #     216 walkers, thin 5  ->  0.50 GB      432 walkers, thin 5  ->  1.00 GB
    #
    # Thinning does NOT reduce how many autocorrelation times a chain holds: rows and tau
    # shrink by the same factor, so rows/(tau/thin) = n_steps/tau is invariant. On the 31D
    # ridge tau ~ 5000, so thin 5 leaves 20000 rows at tau/thin = 1000 -- the same 20 tau
    # the unthinned 100k chain carried, at a fifth of the memory.
    #
    # The constraint is instead thin << tau, so the thinned series still resolves the
    # correlation: as tau/thin approaches 1 the series looks uncorrelated and tau becomes
    # unmeasurable. On these targets tau is 800-5000, so thin 5 has two to three orders of
    # magnitude of headroom. max(tau) is reported every run, so it is checkable.
    thin: int = 1

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        return cls(
            sampler=str(d["sampler"]),
            temperature=float(d["temperature"]),
            n_walkers=int(d["n_walkers"]),
            burn_in=int(d["burn_in"]),
            n_steps=int(d["n_steps"]),
            chunk_size=int(d.get("chunk_size", 5000)),
            thin=int(d.get("thin", 1)),
        )

@dataclass(frozen=True)
class DerivedConfig:
    """The derived-parameter emulator: x -> (derived_1, ..., derived_k).

    A separate network from the log-likelihood one, because the problem is different:
    smooth regression of a few quantities with no peak and no dynamic range to speak
    of, so a plain mean-squared error on standardised outputs rather than msre. Its
    inputs are only the parameters the derived ones can depend on (inputs: auto), which
    for CLASS-derived quantities means the cosmological parameters alone -- 6 of 27 on
    Planck. Give a list of names to override, or `all`.

    Annealing is the default here, unlike for the log-likelihood network: the concern
    recorded under TrainingConfig.lr_schedule is msre on a sparse iteration-0 design,
    which does not arise for this loss.

    Present in a run YAML, the emulator is retrained every iteration, so a likelihood on
    derived parameters can use it while sampling; absent, it is trained on demand by
    `python client.py derived RUN_DIR` with these defaults.

    3 x 256 from a sweep over 2/3/4 layers x 64/128/256 neurons, three seeds each, on the
    it3 data (8298 points) of the Planck NPIPE TT LambdaCDM run, scored as held-out error
    in units of the published Planck+DESI widths. Width mattered more than depth. 3 x 256
    had the lowest error on the hardest parameter, 100 theta_s (2.1% of sigma in the
    high-likelihood core, against 3.6% at 3 x 128), the smallest seed-to-seed spread
    there, and the lowest worst-case error over the whole box (11% of sigma). A fourth
    layer did not help (2.6%). It costs 81 s to train on those 8298 points, and about an
    eighth of the log-likelihood network's cost per evaluation.
    """
    inputs: str | tuple[str, ...] = "auto"
    n_layers: int = 3
    n_neurons: int = 256
    activation: str = "alsing"
    learning_rate: float = 1e-3
    n_epochs: int = 3000
    batch_size: int = 256
    validation_split: float = 0.1
    patience: int = 200
    lr_schedule: str = "plateau"
    lr_factor: float = 0.3
    lr_patience: int = 50
    lr_min: float = 1e-6
    lr_grace: int = 100

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        d = d or {}
        inputs = d.get("inputs", "auto")
        if not isinstance(inputs, str) and not (isinstance(inputs, (list, tuple)) and inputs
                                                and all(isinstance(name, str) for name in inputs)):
            raise ValueError("derived.inputs must be auto, all or a list of parameter names")
        return cls(
            inputs=str(inputs) if isinstance(inputs, str) else tuple(map(str, inputs)),
            n_layers=int(d.get("n_layers", 3)),
            n_neurons=int(d.get("n_neurons", 256)),
            activation=str(d.get("activation", "alsing")),
            learning_rate=float(d.get("learning_rate", 1e-3)),
            n_epochs=int(d.get("n_epochs", 3000)),
            batch_size=int(d.get("batch_size", 256)),
            validation_split=float(d.get("validation_split", 0.1)),
            patience=int(d.get("patience", 200)),
            lr_schedule=str(d.get("lr_schedule", "plateau")),
            lr_factor=float(d.get("lr_factor", 0.3)),
            lr_patience=int(d.get("lr_patience", 50)),
            lr_min=float(d.get("lr_min", 1e-6)),
            lr_grace=int(d.get("lr_grace", 100)),
        )


@dataclass(frozen=True)
class ConvergenceConfig:
    # gaussian_posterior_drift: the drift between the chains of successive networks is below
    # `threshold`. tempered_batch_scatter: the scatter of network minus truth at the tempered
    # batch was below `threshold` x sampling.temperature for `consecutive` networks running
    # (convergence/convergence.py, docs/tempered_convergence.md); `threshold` may then be
    # left out and is 1.2.
    threshold: float  # with metric tempered_batch_scatter: in units of sampling.temperature, and 1.2 when left out
    metric: str  # gaussian_posterior_drift, or tempered_batch_scatter (docs/tempered_convergence.md)
    max_iterations: int
    consecutive: int = 2  # tempered_batch_scatter only: networks running below the threshold

    TEMPERED_THRESHOLD = 1.2  # measured (docs/tempered_convergence.md); not a field

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        metric = str(d["metric"])
        if "threshold" in d:
            threshold = float(d["threshold"])
        elif metric == "tempered_batch_scatter":
            threshold = cls.TEMPERED_THRESHOLD
        else:
            raise KeyError("threshold")
        out = cls(
            threshold=threshold,
            metric=metric,
            max_iterations=int(d["max_iterations"]),
            consecutive=int(d.get("consecutive", 2)),
        )
        if out.consecutive < 1 or (metric == "tempered_batch_scatter" and not out.threshold > 0):
            raise ValueError("convergence: consecutive >= 1; threshold > 0 for tempered_batch_scatter")
        return out


@dataclass(frozen=True)
class Config:
    likelihood: LikelihoodConfig
    prior: PriorConfig
    acquisition: AcquisitionConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    convergence: ConvergenceConfig
    # Top-level `seed:`. Absent (None) leaves every RNG as it was. Set, it fixes the
    # prior design, the network initialisation, the walker start and the acquisition
    # draws -- but NOT the AIES proposals, which XLA reseeds regardless.
    seed: int = None
    # Optional `derived:` section. None (absent) trains no derived-parameter emulator
    # during the run; derived values are recorded with every true evaluation either way.
    derived: DerivedConfig | None = None

    @classmethod
    def from_dict(cls, d):
        if d is not None:
            _validate_mapping(cls, d)
        config = cls(
            likelihood=LikelihoodConfig.from_dict(d["likelihood"]),
            prior=PriorConfig.from_dict(d["prior"]),
            acquisition=AcquisitionConfig.from_dict(d["acquisition"]),
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig.from_dict(d["training"]),
            sampling=SamplingConfig.from_dict(d["sampling"]),
            convergence=ConvergenceConfig.from_dict(d["convergence"]),
            seed=(None if d.get("seed") is None else int(d["seed"])),
            # as acquisition.audit and .boost: absent or false is off; null, true or {} every default
            derived=AcquisitionConfig._section(DerivedConfig, d, "derived"),
        )
        if config.training.members > 1 and config.acquisition.boost is None:
            raise ValueError("training.members acts only on networks trained on boost points: it needs "
                             "acquisition.boost")
        return config

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
