"""How a saved T = 1 chain was sampled, as far as its file says: for the programs that only
read a chain (corner.py, true_profiles/make_config.py) or a true check made from one
(evidence_summary.py). Nothing of CLiENT is imported: they run anywhere the files are.

Since 13 October 2026 a chain records its sampler (`sampling_json`, CLiENT's
sampling/provenance.py). A network whose region has several ellipsoids (a posterior with
several modes) is sampled cell by cell, each cell weighted by the network's evidence in it
(sampling/mixture.py). Until 12 October one ensemble was started on all of them and kept the
shares of its start wherever it did not cross between them. A chain of such a region that
records no sampler may be one of those: analysis/report.py and analysis/true_check.py sample
it again instead of using it, and the readers here must not take it for the posterior either.
"""
import json
import re
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

MIXTURE = "region-cell bridge mixture"          # CLiENT's sampling/provenance.py, sampler_identity
MIN_RELIABLE = 0.1                              # CLiENT's sampling/mixture.py, MIN_RELIABLE_ESS


def ellipsoids(run, iteration, target=None):
    """The number of ellipsoids of the region network `iteration` is sampled in. None when it
    has no region, or when that cannot be read here (the run's boost/ folder is not at hand).
    The file is the one a recorded target names; without a target, RUN/boost/region_it_M.npz
    with the largest M <= iteration (CLiENT's sampling/targets.py, region_file)."""
    if run is None or iteration is None:
        return None
    run = Path(run)
    if isinstance(target, dict) and "region" in target:
        region = target["region"]                               # the resolved entry, or the file's name
        name = region.get("file") if isinstance(region, dict) else region if isinstance(region, str) else None
        if not name:
            return None                                         # recorded: the bare network
        file = run / name if (run / name).exists() else run / "boost" / Path(name).name
    else:
        found = {int(m.group(1)): path for path in (run / "boost").glob("region_it_*.npz")
                 if (m := re.fullmatch(r"region_it_(\d+)\.npz", path.name)) and int(m.group(1)) <= iteration}
        if not found:
            return None
        file = found[max(found)]
    try:
        with np.load(file, allow_pickle=False) as saved:
            return int(len(saved["means"]))
    except (OSError, KeyError, ValueError, TypeError, BadZipFile):
        return None


def recorded(saved, key):
    """The table a file holds under `key`; None when it has none or it cannot be read."""
    if key not in saved.files:
        return None
    try:
        value = json.loads(str(saved[key].item()))
    except (ValueError, TypeError, AttributeError):
        return None
    return value if isinstance(value, dict) else None


def chain_sampling(file, run=None, iteration=None):
    """What a reader must know before it takes the samples of `file` (posterior_it_N.npz, or
    the true_check_it_N.npz made from them) for the network's posterior: (stop, warning), each
    the rest of a sentence about the file, or None.

    stop     the file records no sampler and the network's region has several ellipsoids: the
             shares of the modes may be those of the chain's start. `run` is the run's
             directory, where the region is read; without it this cannot be told, and is not.
    warning  the file records the cell-by-cell sampler and marks cells whose weight is not
             reliable (the efficiency of the bridge estimate below MIN_RELIABLE), or a record
             that cannot be read.

    `iteration` is read from the file's name when it is not given."""
    file = Path(file)
    if iteration is None:
        number = re.search(r"_it_(\d+)(?!\d)", file.name)
        iteration = int(number.group(1)) if number else None
    with np.load(file, allow_pickle=False) as saved:
        has_record = "sampling_json" in saved.files
        target, sampling = recorded(saved, "target_json"), recorded(saved, "sampling_json")
    if not has_record:
        count = ellipsoids(run, iteration, target)
        if count is not None and count > 1:
            return (f"does not record how it was sampled, and the region of network {iteration} has {count} ellipsoids: "
                    "until 12 October 2026 one chain was started on all of them and kept the shares of its start "
                    "wherever it did not cross between them"), None
        return None, None
    if sampling is None or "method" not in sampling:
        return None, "holds a record of how it was sampled that cannot be read"
    if sampling["method"] != MIXTURE:
        return None, None
    try:
        bad = {int(cell) for cell in sampling["unreliable_cells"]}
        bad |= {int(cell["cell"]) for cell in sampling["cells"] if not float(cell["efficiency"]) >= MIN_RELIABLE}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "holds a record of its cells' weights that cannot be read"
    if bad:
        return None, (f"marks the weight of cell(s) {sorted(bad)} of its region as not reliable (efficiency of the bridge "
                      f"estimate below {MIN_RELIABLE:g}): the shares of the modes in these samples are uncertain")
    return None, None
