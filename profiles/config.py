"""Validated, portable user configuration (no TensorFlow/backend imports)."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import os
import yaml


def check_continuation(continuation):
    """The setting `continuation` (ProfileConfig): {seed: a number, start: best or scan}.
    Returns (seed, start)."""
    c = continuation
    if (not isinstance(c, dict) or set(c) - {"seed", "start"} or "seed" not in c
            or isinstance(c["seed"], bool) or not isinstance(c["seed"], (int, float))
            or not math.isfinite(c["seed"]) or c.get("start", "best") not in ("best", "scan")):
        raise ValueError("continuation must be null or {seed: a number, start: best or scan}")
    return float(c["seed"]), c.get("start", "best")


# How a study is executed, not what it computes: results do not depend on these, so a study
# may be resumed with other values (e.g. on another machine, or with more cores).
EXECUTION_SETTINGS = ("workers",)
SURROGATE_OPTIMIZERS = ("hybrid", "legacy")
# The settings that are real numbers (corrected_tol may also be None).
REAL_SETTINGS = ("broad_fraction", "sigma_range", "initial_lr", "transition_lr", "final_lr", "gtol",
                 "audit_tolerance", "corrected_tol")
# corrected_tol when unset: predicted remaining gain of the residual Newton stage, in native
# log-objective units.
CORRECTED_TOL_DEFAULT = 1e-2


@dataclass
class ProfileConfig:
    dimension: int = 1
    retrain: bool = False
    design_points: int = 7
    iteration: int | None = None
    parameters: list[str] = field(default_factory=list)
    added_points: int = 0
    broad_fraction: float = 0.75
    profile_points: int = 15
    sigma_range: float = 3.5
    ranges: dict = field(default_factory=dict)
    workers: int = 1
    first_epochs: int = 400
    polish_epochs: int = 1600
    initial_lr: float = 1e-5
    transition_lr: float = 1e-6
    final_lr: float = 1e-7
    batch_size: int = 128
    seed: int = 42
    validation_global: int = 512
    validation_nodes: int = 9
    scan_points: int = 41
    gtol: float = 1e-4
    maxfun: int = 150
    total_call_budget: int = 100000
    audit_points: int = 3
    # The audit recomputes the audit_points saved labels with the highest label, where the
    # profiles are. The first rows are design points far from the peak, where chi-squared is
    # ~1e5 and a 1e-5 relative difference between two machines' backends exceeds the absolute
    # tolerance although the rows near the peak agree to 1e-4.
    audit_tolerance: float = 0.01
    # The surrogate stage is the hybrid scan (batched basin hopping + certified projected
    # BFGS, profiles/hybrid.py), or with "legacy" L-BFGS-B with polish and continuation passes
    # (profiles/surrogate.py, profiles/surface.py), which on a rugged 2D network can find better
    # nodes than the hybrid scan. The true stage is residual Newton with measured gradients and a
    # learned curvature correction (profiles/residual_newton.py); results do not depend on
    # `workers`. corrected_tol: its native log-objective gain tolerance; None uses
    # CORRECTED_TOL_DEFAULT. maxfun: its true calls per node.
    surrogate_optimizer: str = "hybrid"
    max_passes: int = 15  # the legacy scan only: the most continuation passes
    hybrid_walkers: int = 8
    # Hops of the hybrid scan at most (profiles/hybrid.py, HybridConfig.hops_max); None, its
    # default of 12 (every study made before 17 October 2026). The scan has converged only
    # after two hops in a row in which no node improved; on a rugged network (several
    # phase maxima at every node) a 2500-node surface can still be settling at hop 12.
    hybrid_hops: int | None = None
    corrected_tol: float | None = None
    backend_module: str | None = None
    backend_sha256: str | None = None
    run_config: str | None = None
    # Parallel structure of the corrected/newton true stage (1D and 2D). chain_length caps
    # how many nodes are solved in sequence from one cold start: k adds a seed after every
    # k chained nodes; None gives one seed per profile (or surface row) with a chain each
    # way (studies made before 30 September 2026); 0 makes every node independent. Shorter
    # chains expose more concurrent work at the cost of warm starts; results depend on it,
    # never on workers or timing. 2 keeps 30-40 workers busy on one or two profiles for
    # 1-3% more calls (docs/create_profiles.md#many-workers).
    chain_length: int | None = 2
    # Where a network trained on boosts is searched (profiles/region.py): "run", inside the
    # region the run saved for it (RUN/boost/region_it_N.npz, acquisition.boost.region); a
    # number, the same region with its edge at that many times the reach of the boost points;
    # None, the bare network over the whole box (every study made before 10 October 2026).
    # A network of the ordinary loop has no region: it is used over the whole box whatever
    # this says. Outside its region a boost network has maxima the truth does not have.
    region: str | float | None = "run"
    # Opt-in (2D, corrected/newton; 18 October 2026): each row's true optima continued along
    # the second parameter. {seed: v, start: best | scan}: a row's one seed is its node nearest
    # v, started from the best training point ("best", the default) or from the network's
    # optimum there ("scan"); every other node starts from the true optimum of the node before
    # it in its chain, not from the network's optimum. For a network far wrong in part of a
    # surface (the axion run's low-omega corner, 120 to 280 log-units too low), where its
    # optima are no start. Needs chain_length: null (one seed a row).
    # None: every node starts from the network's optimum (every study before the key).
    continuation: dict | None = None
    # Opt-in (2D, retrain: false; 7 November 2026): narrower ranges for
    # parameters that are not on the grid, {name: [low, high]} inside the network's box. The
    # scan and the true stage search those parameters only there, and the scan starts from the
    # training points inside. For a free parameter with many separated maxima (the axion
    # run's omega at fixed (A_osc, phi)): the hybrid scan's walkers start at the best point of
    # the box and step locally, so it can miss a maximum far from that point, which a study
    # with a range around it finds. Each node is then the profile over that range only.
    # None: the whole box (every study made before the key).
    search_ranges: dict | None = None

    @classmethod
    def load(cls, path=None, **overrides):
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("profile YAML must contain a mapping")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown profile settings: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        # PyYAML reads a number written without a decimal point, 1e-2, as a string. The run's
        # own settings take it as the number; so do these.
        for name in REAL_SETTINGS:
            if isinstance(data.get(name), str):
                try:
                    data[name] = float(data[name])
                except ValueError:
                    pass                      # validate() names the setting
        # A positive acquisition request is an explicit opt-in, including old YAMLs.
        additions = data.get("added_points", 0)
        if "retrain" not in data and isinstance(additions, int) and additions > 0:
            data["retrain"] = True
        obj = cls(**data)
        obj.validate()
        return obj

    def validate(self):
        ints = dict(
            dimension=1,
            design_points=3,
            added_points=0,
            profile_points=3,
            workers=1,
            first_epochs=2,
            polish_epochs=2,
            batch_size=1,
            validation_global=1,
            validation_nodes=3,
            scan_points=5,
            maxfun=1,
            total_call_budget=1,
            audit_points=1,
            seed=0,
            hybrid_walkers=2,
            max_passes=1,
        )
        for name, minimum in ints.items():
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.dimension not in (1, 2):
            raise ValueError("dimension must be 1 or 2")
        if self.surrogate_optimizer not in SURROGATE_OPTIMIZERS:
            raise ValueError(f"surrogate_optimizer must be one of {SURROGATE_OPTIMIZERS}")
        if not isinstance(self.retrain, bool) or (
            not self.retrain and self.added_points
        ):
            raise ValueError("retrain=false requires added_points=0")
        if self.dimension == 2 and (
            not isinstance(self.parameters, list)
            or len(self.parameters) != 2
            or self.profile_points**2 > 2500
            or self.design_points**2 > 625
        ):
            raise ValueError(
                "2D requires exactly one pair, <=2500 output nodes and <=625 design nodes"
            )
        # Workers are persistent local processes, one backend each: never more than the
        # host's CPUs. Never launch this command under mpirun.
        if self.workers > (os.cpu_count() or 1):
            raise ValueError(f"workers ({self.workers}) exceeds this host's {os.cpu_count()} CPUs")
        if self.region is not None and self.region != "run" and (
            isinstance(self.region, (bool, str))
            or not isinstance(self.region, (int, float))
            or not math.isfinite(self.region)
            or self.region <= 0
        ):
            raise ValueError(
                "region must be run, a positive number of reaches, or null for the bare network")
        # 4 is the scan's hops_min: it hops at least that often whatever the limit.
        if self.hybrid_hops is not None and (
            isinstance(self.hybrid_hops, bool)
            or not isinstance(self.hybrid_hops, int)
            or self.hybrid_hops < 4
        ):
            raise ValueError("hybrid_hops must be null or an integer >= 4")
        if self.chain_length is not None and (
            isinstance(self.chain_length, bool)
            or not isinstance(self.chain_length, int)
            or self.chain_length < 0
        ):
            raise ValueError("chain_length must be null or an integer >= 0")
        if self.continuation is not None:
            check_continuation(self.continuation)
            if self.dimension != 2:
                raise ValueError("continuation requires a 2D study")
            if self.chain_length is not None:
                raise ValueError("continuation requires chain_length: null (one seed a row)")
        if self.iteration is not None and (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 0
        ):
            raise ValueError("invalid iteration")
        if (
            not isinstance(self.parameters, list)
            or not self.parameters
            or any(not isinstance(n, str) or not n for n in self.parameters)
            or len(set(self.parameters)) != len(self.parameters)
        ):
            raise ValueError("parameters must be a nonempty list of unique names")
        for name in REAL_SETTINGS:
            v = getattr(self, name)
            if name == "corrected_tol" and v is None:
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                raise ValueError(f"{name} must be finite")
        if self.corrected_tol is not None and self.corrected_tol <= 0:
            raise ValueError("corrected_tol must be positive")
        if not 0 <= self.broad_fraction <= 1:
            raise ValueError("broad_fraction must be in [0,1]")
        if min(self.sigma_range, self.gtol, self.audit_tolerance) <= 0:
            raise ValueError("range and tolerances must be positive")
        if not 0 < self.final_lr <= self.transition_lr <= self.initial_lr:
            raise ValueError("invalid learning rate schedule")
        if not isinstance(self.ranges, dict) or set(self.ranges) - set(self.parameters):
            raise ValueError("ranges must name selected parameters only")
        for n, r in self.ranges.items():
            if r == "prior":
                continue
            if (
                not isinstance(r, list)
                or len(r) != 2
                or any(
                    not isinstance(x, (float, int)) or not math.isfinite(x) for x in r
                )
                or r[0] >= r[1]
            ):
                raise ValueError(f"invalid range for {n}: use [low, high] or prior")
        if self.search_ranges is not None:
            if not isinstance(self.search_ranges, dict) or not self.search_ranges:
                raise ValueError("search_ranges must be null or a mapping of parameter names to [low, high]")
            for n, r in self.search_ranges.items():
                if not isinstance(n, str) or not n or n in self.parameters:
                    raise ValueError("search_ranges must name parameters that are not on the grid")
                if (not isinstance(r, list) or len(r) != 2
                        or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                               for x in r)
                        or r[0] >= r[1]):
                    raise ValueError(f"invalid search range for {n}: use [low, high]")
            if self.dimension != 2:
                raise ValueError("search_ranges requires a 2D study")
            if self.retrain:
                raise ValueError("search_ranges requires retrain: false")
            # The range scan that chooses omitted ranges searches the whole box.
            if set(self.parameters) - set(self.ranges):
                raise ValueError("search_ranges requires ranges for both grid parameters")
        if self.backend_module is not None and (
            not isinstance(self.backend_module, str) or not self.backend_module
        ):
            raise ValueError("backend_module must be a module name")
        if self.backend_sha256 is not None:
            import re

            if not isinstance(self.backend_sha256, str) or not re.fullmatch(
                "[0-9a-f]{64}", self.backend_sha256
            ):
                raise ValueError(
                    "backend_sha256 must be 64 lowercase hexadecimal characters"
                )
        if self.backend_sha256 and not self.backend_module:
            raise ValueError("backend_sha256 requires backend_module")

    def as_dict(self):
        return asdict(self)

    def corrected_tolerance(self):
        """corrected_tol, or the residual Newton stage's default when unset."""
        if self.corrected_tol is not None:
            return self.corrected_tol
        return CORRECTED_TOL_DEFAULT
