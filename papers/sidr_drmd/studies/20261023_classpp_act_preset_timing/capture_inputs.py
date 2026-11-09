"""Write the exact input that Cobaya hands to classy in the precision test, for every case of it.

    PYTHONPATH=CLASSPP/_install COBAYA_PACKAGES_PATH=PACKAGES python capture_inputs.py

The Cobaya model of results/20261023_act_spt_scoping/precision_test_v2.py (the same likelihoods, cosmologies and
settings sets) is built, and classy.Class.set is wrapped so that the dictionary Cobaya passes is recorded; compute()
is then stopped before CLASS++ runs. Writes inputs/<cosmology>_<set>.ini (the values one per line, as ./class reads
them) and inputs/requirements.json (what each likelihood asks of the theory code).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCOPING = HERE.parent / "20261023_act_spt_scoping"
sys.path.insert(0, str(SCOPING))


class Captured(Exception):
    pass


def main():
    import classy
    from cobaya.model import get_model
    from precision_test import cosmologies
    from precision_test_v2 import LIKES, settings

    recorded = {}

    class Recording(classy.Class):
        """classy.Class that records what it is given and stops before computing."""

        def set(self, *args, **kwargs):
            pars = dict(args[0]) if args else {}
            pars.update(kwargs)
            recorded.update(pars)
            return super().set(*args, **kwargs)

        def compute(self, *args, **kwargs):
            raise Captured()

    classy.Class = Recording                  # Cobaya's wrapper instantiates classy.Class from the imported module

    out = HERE / "inputs"
    out.mkdir(exist_ok=True)
    centre = json.loads((HERE.parents[0] / "20261007_grendel_kit" / "hpc_client" / "results" / "hillik_plksptact_lcdm_centre"
                         / "centre.json").read_text())["parameters"]
    nuisance = {k: v["value"] for k, v in centre.items() if not v.get("fixed")}
    nuisance.update(ACT_radio_EE=0.0, SPT3G_radio_EE=0.0)
    requirements = {}
    for cname, cosmo in cosmologies().items():
        tag = "lcdm" if cname.startswith("LCDM") else "sidr_drmd"
        for sname, s in settings().items():
            info = {"likelihood": {k: None for k in LIKES},
                    "theory": {"classy": {"extra_args": {**cosmo["extra"], **s}, "stop_at_error": True}},
                    "params": {**cosmo["params"], **nuisance}}
            model = get_model(info)
            if not requirements:
                for name, like in model.likelihood.items():
                    requirements[name] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
                                          for k, v in (like.get_requirements() or {}).items()}
            recorded.clear()
            try:
                model.loglikes({})
            except Captured:
                pass
            except Exception as error:          # Cobaya may wrap the exception
                if "Captured" not in repr(error) and not recorded:
                    raise
            short = sname.split()[0]
            lines = [f"# Input Cobaya {__import__('cobaya').__version__} passes to classy (CLASS++ {classy.__file__}):",
                     f"# cosmology '{cname}', settings '{sname}' of precision_test_v2.py.",
                     "# Cobaya's classy wrapper sets output, l_max_scalars, lensing and the like from the likelihoods' needs."]
            lines += [f"{k} = {v}" for k, v in recorded.items()]
            (out / f"{tag}_{short}.ini").write_text("\n".join(lines) + "\n")
            print(f"{tag}_{short}.ini: {len(recorded)} keys")
            del model
    (out / "requirements.json").write_text(json.dumps(requirements, indent=1, default=str))
    print("written:", out)


if __name__ == "__main__":
    main()
