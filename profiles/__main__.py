"""python -m profiles RUN --params omega_b,omega_cdm --output PROFILE_DIR"""

import argparse
import os
from pathlib import Path
import json
from profiles.config import ProfileConfig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create true profiles guided by a saved CLiENT network; augmentation is opt-in."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="profile settings YAML")
    parser.add_argument(
        "--iteration", type=int, help="default latest matching model/data iteration"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--params", help="comma-separated sampled parameter names (1D)")
    group.add_argument(
        "--pair", help="exactly two comma-separated names for one 2D surface"
    )
    for flag, dest in [
        ("add-points", "added_points"),
        ("points", "profile_points"),
        ("workers", "workers"),
    ]:
        parser.add_argument("--" + flag, dest=dest, type=int)
    parser.add_argument("--sigma-range", type=float)
    parser.add_argument("--retrain", action=argparse.BooleanOptionalAction, default=None,
                        help="opt in to profile training; default uses the saved network")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse saved stages and exact successful evaluations; config must match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate files and print plan; zero backend calls and no output directory",
    )
    args = parser.parse_args(argv)
    try:
        if any(
            int(os.environ.get(k, "1")) > 1
            for k in ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"]
        ):
            raise ValueError("Do not run profiles under mpirun; use --workers (up to the number of cores)")
        for k in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        ]:
            os.environ.setdefault(k, "1")
        overrides = {
            k: getattr(args, k)
            for k in [
                "iteration",
                "retrain",
                "added_points",
                "profile_points",
                "workers",
                "sigma_range",
            ]
        }
        if args.pair:
            overrides["dimension"] = 2
        overrides["parameters"] = (
            [n.strip() for n in (args.pair or args.params).split(",")]
            if (args.pair or args.params)
            else None
        )
        if (
            args.resume
            and args.config is None
            and (args.output / "manifest.json").exists()
        ):
            saved = json.loads((args.output / "manifest.json").read_text())[
                "configuration"
            ]
            saved.update({k: v for k, v in overrides.items() if v is not None})
            config = ProfileConfig(**saved)
            config.validate()
        else:
            config = ProfileConfig.load(args.config, **overrides)
        from profiles.workflow import run_workflow

        run_workflow(
            args.run, args.output, config, resume=args.resume, dry_run=args.dry_run
        )
    except (ValueError, FileNotFoundError) as exc:
        # A refusal of the settings or of the study, shown as a usage error. Every ValueError
        # of the workflow ends here, also one from deep inside a stage, so the message says
        # where it was raised; CLIENT_PROFILE_TRACEBACK=1 gives the whole traceback.
        import traceback

        if os.environ.get("CLIENT_PROFILE_TRACEBACK"):
            raise
        frames = traceback.extract_tb(exc.__traceback__)
        origin = f"  [raised in {Path(frames[-1].filename).name}:{frames[-1].lineno}]" if frames else ""
        parser.error(str(exc) + origin)


if __name__ == "__main__":
    main()
