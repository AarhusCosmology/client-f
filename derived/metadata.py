import json
from dataclasses import dataclass
from pathlib import Path

from likelihood.base import ParameterInfo

FILENAME = "derived_parameters.json"


@dataclass(frozen=True)
class DerivedMetadata:
    """Names, labels and scales of the derived parameters, and the inputs they depend on.

    Kept in derived_data/ rather than in metadata.json: that file is written once when a
    run is created and fingerprinted by profile studies, so a run recorded before
    derived parameters existed must not have it rewritten when it is continued.
    """

    parameters: tuple[ParameterInfo, ...]
    inputs: tuple[str, ...]

    @classmethod
    def from_likelihood(cls, likelihood):
        return cls(
            parameters=tuple(
                ParameterInfo(name=name, label=label, scale=scale)
                for name, label, scale in zip(
                    likelihood.derived_names,
                    likelihood.derived_labels,
                    likelihood.derived_scales,
                )
            ),
            inputs=tuple(likelihood.derived_input_names),
        )

    @property
    def names(self):
        return [param.name for param in self.parameters]

    @property
    def labels(self):
        return [param.label for param in self.parameters]

    def save(self, derived_data_dir):
        path = Path(derived_data_dir) / FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "parameters": [
                {"name": p.name, "label": p.label, "scale": p.scale} for p in self.parameters
            ],
            "inputs": list(self.inputs),
        }
        from run.storage import write_json

        write_json(path, data)                     # read at the next start: whole or not at all

    @classmethod
    def load(cls, derived_data_dir):
        data = json.loads((Path(derived_data_dir) / FILENAME).read_text(encoding="utf-8"))
        return cls(
            parameters=tuple(
                ParameterInfo(name=p["name"], label=p["label"], scale=p.get("scale", 1.0))
                for p in data["parameters"]
            ),
            inputs=tuple(data["inputs"]),
        )


def changed_since_recorded(derived_data_dir, names):
    """(added, removed) against the derived parameters a run was recorded with, or None
    when it has no record (a new run, or one recorded before derived parameters)."""
    path = Path(derived_data_dir) / FILENAME
    if not path.exists():
        return None
    recorded = DerivedMetadata.load(derived_data_dir).names
    added = [n for n in names if n not in recorded]
    removed = [n for n in recorded if n not in names]
    return (added, removed) if added or removed else None
