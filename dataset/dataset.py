from run.storage import atomic_path
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class _WhitenedKNN:
    def __init__(self, points, n_neighbors, transform=None):
        self.ndim = points.shape[1]
        self.n_neighbors = n_neighbors
        if transform is None:
            self._mean = points.mean(axis=0)
            cov = np.cov(points, rowvar=False)

            # Apply small jitter for numerical stability
            # if any parameter has near-zero variance
            cov = cov + 1e-10 * np.eye(self.ndim)

            # Compute the Cholesky factor L where Σ = LL^T
            L = np.linalg.cholesky(cov)

            # For a row vector x, the whitening transform can be written as z = (x - μ) W, where W = L^{-T}
            self._W = np.linalg.solve(L, np.eye(self.ndim)).T
        else:
            # Reuse another index's whitening so that densities estimated over a
            # different point cloud are measured in exactly the same metric.
            self._mean, self._W = transform

        # Build a Euclidean nearest-neighbor index in whitened space
        # (e.g. a tree, or stored points for brute-force search depending on the algorithm)
        self._knn_index = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="auto", metric="euclidean"
        )
        self._knn_index.fit(self.whiten(points))

    @property
    def transform(self):
        return self._mean, self._W

    def whiten(self, points):
        return (points - self._mean) @ self._W

    def query(self, points, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        return self._knn_index.kneighbors(self.whiten(points), n_neighbors=n_neighbors)


class TrainingDataset:
    def __init__(self, inputs, targets, likelihood, n_neighbors, target_temperature,
                 derived=None):
        self.inputs = np.asarray(inputs, dtype=np.float32).copy()
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1).copy()
        self.likelihood = likelihood
        self.param_names = likelihood.param_names
        self.derived_names = list(getattr(likelihood, "derived_names", []))
        self.derived = self._as_derived(derived, len(self.inputs))
        self.n_neighbors = n_neighbors
        self.target_temperature = target_temperature
        self.iteration = None
        self._knn_index = None

    def _as_derived(self, derived, n_rows):
        # Missing derived values are NaN rows, so the array always lines up with
        # inputs. float64: unlike the inputs they never pass through the network as-is.
        if derived is None:
            return np.full((n_rows, len(self.derived_names)), np.nan)
        return np.asarray(derived, dtype=np.float64).reshape(
            n_rows, len(self.derived_names)
        ).copy()

    @property
    def knn_index(self):
        # Built when it is read: an iteration adds to the set up to four times and reads it once.
        if self._knn_index is None:
            self._knn_index = _WhitenedKNN(self.inputs, self.n_neighbors)
        return self._knn_index

    def save(self, path):
        df = pd.DataFrame(self.inputs, columns=self.param_names)
        df["loglkl"] = self.targets[:, 0]
        with atomic_path(path) as temporary:
            df.to_csv(temporary, index=False)

    def save_derived(self, path):
        """Write every point's derived parameters, row-aligned with ``save``.

        A separate file because data_it_<i>.csv is read positionally elsewhere: the
        profile workflow requires its header to be exactly the parameters then
        loglkl. The inputs are repeated, so the file stands on its own and ``load``
        can check that its rows still line up. Nothing is written when the
        likelihood has no derived parameters.
        """
        if not self.derived_names:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.inputs, columns=self.param_names)
        for j, name in enumerate(self.derived_names):
            df[name] = self.derived[:, j]
        with atomic_path(path) as temporary:
            df.to_csv(temporary, index=False)

    def _load_derived(self, path):
        # Leaves NaN wherever the file cannot vouch for a value: runs recorded before
        # derived parameters existed, and derived parameters added to the likelihood
        # since. A file whose rows no longer match the inputs is ignored outright.
        path = Path(path)
        if not self.derived_names or not path.exists():
            return
        df = pd.read_csv(path)
        inputs = df[self.param_names].to_numpy(dtype=np.float32)
        if inputs.shape != self.inputs.shape or not np.array_equal(inputs, self.inputs):
            warnings.warn(
                f"{path.name} does not line up with the training data; "
                "derived parameters of the loaded points are left as NaN"
            )
            return
        for j, name in enumerate(self.derived_names):
            if name in df.columns:
                self.derived[:, j] = df[name].to_numpy(dtype=np.float64)

    @classmethod
    def load(
        cls,
        training_data_dir,
        likelihood,
        n_neighbors,
        target_temperature,
        iteration=None,
        derived_data_dir=None,
    ):
        # If no iteration is specified, load the latest available
        # data_it_<iteration>.csv file.
        if iteration is None:
            iteration = max(
                int(p.stem.rsplit("_", 1)[1]) for p in Path(training_data_dir).iterdir()
            )
        path = Path(training_data_dir) / f"data_it_{iteration}.csv"
        df = pd.read_csv(path)
        param_names = likelihood.param_names
        inputs = df[param_names].to_numpy(dtype=np.float32)
        targets = df[["loglkl"]].to_numpy(dtype=np.float32)
        dataset = cls(inputs, targets, likelihood, n_neighbors, target_temperature)
        dataset.iteration = iteration
        if derived_data_dir is not None:
            dataset._load_derived(Path(derived_data_dir) / f"derived_it_{iteration}.csv")
        return dataset

    def add_data(self, inputs, targets, derived=None):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        derived = self._as_derived(derived, len(inputs))
        self.inputs = np.concatenate([self.inputs, inputs])
        self.targets = np.concatenate([self.targets, targets])
        self.derived = np.concatenate([self.derived, derived])
        self._knn_index = None
