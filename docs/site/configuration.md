# Configuration and defaults

The [complete field inventory](_generated/reference/configuration.md) is regenerated from `config/config.py` and `profiles/config.py` during every documentation build. It records field types and omitted-field defaults, using explicit YAML parser fallbacks when they differ from constructor defaults. It does not import TensorFlow or a likelihood backend.

## Defaults and recommended settings

Most new methods are opt-in, so a run that takes only the defaults is not the method of the real runs. `input/recommended_near_gaussian.yaml` and `input/recommended_non_gaussian.yaml` are complete configurations with those settings; [the boost guide](_generated/library/docs/boost.md#two-recommended-settings) says which to take.

## Read absence, null and an empty mapping differently

- An **absent** `acquisition.audit` disables the audit, and so does `audit: false`. `audit: {}` activates default audit settings, as do `audit: true` and, in this parser, `audit: null` (converted to an empty mapping). The same holds for the top-level `derived` section.
- An **absent** `acquisition.boost` disables boosting, and so does `boost: false`. `boost: {}`, `boost: true` and `boost: null` do not form a valid boost: `n` and `start_iteration` are required.
- An absent `derived` section disables emulator training. Use an explicit mapping to configure it.
- Types are not a substitute for validation: e.g. boost `walkers` must be even and at least four, `steps` at least 100, `region` greater than one, and `mixing` strictly between zero and one.

## Current behaviour at a glance

| Setting | Default when omitted | Interpretation |
|---|---|---|
| `training.warm_start` | `true` | Transfer the preceding network's weights, with the raw predictions kept when the scalers change; `false` starts each network from random weights. A network warm-started after the boosts began keeps the weights at the end of its schedule; the ordinary loop and trainings from random weights keep the epoch with the lowest validation loss |
| `training.lr_schedule` | `none` | No learning-rate annealing; the real runs use `plateau` |
| `training.msre_ess_floor` | `0.1` | Dynamic floor for the loss-weight ESS fraction |
| `prior.n_inject`, `acquisition.n_inject` | `0` | No extra quadratic-fit injection |
| `acquisition.audit` | Absent | No audit |
| `acquisition.boost` | Absent | No boost |
| `convergence.consecutive` | `2` | Number of preceding batch scatters for the true-evaluation criterion |
| `convergence.metric` | required | `gaussian_posterior_drift`, or `tempered_batch_scatter` for the criterion on true evaluations (then `threshold` may be omitted and is `1.2`, in units of the sampling temperature) |
| `boost.boosts` | `null` | The run's last iteration decides how many boosts there are |
| `boost.density_filter` | `false` | Unfiltered, equally allocated mode samples; the real runs use `true` |
| `boost.background`, `boost.region`, `boost.mixing` | `null` | Each extension is off |
| `boost.region_quantile`, `boost.train_domain`, `boost.mode_depth`, `boost.balance_floor` | `null` | Each extension is off |
| Profile `retrain`, `added_points` | `false`, `0` | Use saved network without pre-profile training |
| Profile `chain_length` | `2` | Short deterministic continuation chains |
| Profile `region` | `run` | A network trained on boosts is searched inside the region of its boosts; `null` is the bare network |

The `boost.*` shorthand above means keys inside `acquisition.boost`.

## Inputs and compatibility

The training YAML and profile YAML are different schemas. Put a setting in the command's own configuration; the profile CLI can print resolved settings with `--dry-run`. Both reject settings they do not know, and the run stops before its initial design when a loss, an activation or the number of walkers does not fit the problem.

Resume preserves the settings required to reproduce an existing study. A default does not authorise changing the method or domain of a saved profile study. See [migration notes](_generated/library/UPGRADING.md) and [profile resume rules](_generated/library/docs/create_profiles.md).

## Limits and conditional settings

Detailed allowed values and interactions are maintained beside their workflows:

- [Training, sampling and ordinary acquisition](_generated/library/README.md#configuration)
- [Audit budgets and failure handling](_generated/library/docs/audit_optima.md)
- [Boost settings](_generated/library/docs/boost.md) and [density selection](_generated/library/docs/posterior_density_boost.md)
- [Profiles](_generated/library/docs/create_profiles.md) and [2D restrictions](_generated/library/docs/create_profiles_2d.md)
- [Derived emulator settings](_generated/library/docs/derived_parameters.md)

The generated inventory lists every schema field; it is not a claim that every positive value or every combination is supported. The parser and workflow validation remain authoritative.

## Conditional convergence and phase limits

For `metric: tempered_batch_scatter`, omitting `threshold` gives `1.2`; for `gaussian_posterior_drift`, `threshold` is required. The generated table marks this condition explicitly. `consecutive` must be at least one, and `tempered_batch_scatter` requires a positive threshold. Its units are multiples of `sampling.temperature`, not absolute native log probability.

`acquisition.boost.boosts` limits scheduled boost-phase iterations when no explicit `-i` is supplied. A no-mode fallback still consumes a phase iteration. It does not promise a fixed number of successful posterior batches. The effective end is also limited by the run's iteration ceiling. A saved early start in `boost/start.json` is respected on continuation. See [handover semantics](workflow.md#true-evaluation-convergence-and-early-boost-handover).

The profile `region` setting differs from boost `region`: `run` inherits a saved region, `null` disables it, and a positive number overrides the edge. If the selected network has no saved region, the bare network is used. See [region-aware profiles](inference.md#profiles-inside-a-saved-boost-region).
