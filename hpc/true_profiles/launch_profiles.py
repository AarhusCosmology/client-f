"""True profiles of a cluster run: `python client.py profiles` with what the cluster's
likelihoods need around it (cluster_backend.py).

    python true_profiles/launch_profiles.py results/<run> --config FILE -o OUTPUT [--resume] [--dry-run]

The arguments are those of `python client.py profiles` (docs/create_profiles.md in CLiENT).
Not under mpirun: the profile code starts its own `workers` backend processes on this node.
submit_profiles.sh is an example job script.

CLIENT_PROFILE_NEWTON (cluster_backend.newton_settings) and CLIENT_CAMB_ACCURACY_TARGET
(cluster_backend.py) are recorded in OUTPUT.newton_settings.json, beside the study, when the
study is made; a resumed study must be given the same.

Where a network trained on boosts is searched is the profile YAML's `region` (CLiENT's
profiles/region.py), not a variable of the job: CLIENT_PROFILE_REGION, which did that in an
earlier launcher, stops this one, and so does a CLiENT without the setting, whose
profile code would search the bare network over the whole box.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("CLIENT_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(HERE))


def output_directory(arguments):
    for k, word in enumerate(arguments):
        if word in ("-o", "--output") and k + 1 < len(arguments):
            return Path(arguments[k + 1])
        if word.startswith("--output="):
            return Path(word.split("=", 1)[1])
    return None


def region_is_the_yaml_s():
    """Stop when the region would not be applied as the user thinks it is."""
    from profiles.config import ProfileConfig

    if "region" not in ProfileConfig.__dataclass_fields__:
        raise SystemExit("this launcher needs a CLiENT whose profile code has the setting `region`: without it a network "
                         "trained on boosts is searched over the whole box. Point CLIENT_REPO at a newer CLiENT")
    if (os.environ.get("CLIENT_PROFILE_REGION") or "").strip():
        raise SystemExit("CLIENT_PROFILE_REGION is not read any more. Put `region: <edge>` in the profile YAML (run, a number "
                         "of reaches, or null; make_config.py --region writes it) and take the variable out of the job's settings")


def main():
    import cluster_backend

    region_is_the_yaml_s()
    cluster_backend.install()
    settings = cluster_backend.newton_settings()
    target = cluster_backend.camb_accuracy_target()
    if target is not None:
        settings = dict(settings, CLIENT_CAMB_ACCURACY_TARGET=target)      # not a Newton field: the objective itself
    output = output_directory(sys.argv[1:])
    if output is not None and "--dry-run" not in sys.argv:
        # Beside the study, not inside it: a new study's folder must be empty.
        record = output.with_name(output.name + ".newton_settings.json")
        if (output / "manifest.json").exists():
            saved = json.loads(record.read_text()) if record.exists() else {}
            earlier = saved.pop("CLIENT_PROFILE_REGION", 0)    # an earlier launcher's own region
            if earlier != 0:
                raise SystemExit(f"{output} was made by a launcher that applied the region itself (edge {earlier}); CLiENT's "
                                 "profile code does that now and cannot continue it. A new study needs a new output directory")
            if saved != settings:
                raise SystemExit(f"{output} was made with the settings {saved or 'unset'} (CLIENT_PROFILE_NEWTON, "
                                 f"CLIENT_CAMB_ACCURACY_TARGET); now {settings or 'unset'}")
        else:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(settings, indent=1) + "\n")
    print(f"[profiles] backend processes build their likelihood {os.environ.get('CLIENT_LOAD_GROUP') or 'all'} at a time; "
          f"settings beyond the YAML: {settings or 'none'}", flush=True)
    from profiles.__main__ import main as profiles_main

    profiles_main()


if __name__ == "__main__":
    main()
