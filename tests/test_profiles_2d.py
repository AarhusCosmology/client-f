"""Multiple fixed-coordinate constraints and one-pair workflow contract."""

import tempfile, unittest
from pathlib import Path
import numpy as np
from profiles.config import ProfileConfig
from profiles.surrogate import ProfileOptimizer


class PairTests(unittest.TestCase):
    def test_pair_contract(self):
        for names in [None, [], ["x"], ["x", "y", "z"]]:
            with self.assertRaises(ValueError):
                ProfileConfig(dimension=2, parameters=names).validate()
        with self.assertRaises(ValueError):
            ProfileConfig(
                dimension=2, parameters=["x", "y"], profile_points=51
            ).validate()
        ProfileConfig(dimension=2, parameters=["x", "y"]).validate()

    def test_joint_conditional_not_product_of_1d(self):
        # f = -.5[(z-x-y)^2 + x^2 + (y-x)^2]; z is the nuisance.
        def obj(u, ids, vals):
            x = np.empty(3)
            mask = np.ones(3, bool)
            if ids is not None:
                mask[ids] = False
                x[ids] = vals
            x[mask] = u * 5
            a, b, z = x
            d = z - a - b
            return (
                0.5 * (d * d + a * a + (b - a) ** 2),
                np.array([-d + 2 * a - b, -d + b - a, d])[mask] * 5,
            )

        optimizer = ProfileOptimizer(obj, [-5] * 3, [5] * 3, gtol=1e-7)
        v, x, d = optimizer([0, 1], [1.0, 2.0], [np.zeros(3)])
        np.testing.assert_allclose(x, [1, 2, 3], atol=1e-6)
        self.assertAlmostEqual(v, -1.0)
        self.assertTrue(d["stationary"])


class PairWorkflowTests(unittest.TestCase):
    def test_saved_network_augmentation_to_surface(self):
        import json, yaml
        import tensorflow as tf
        from likelihood.cobaya import CobayaLikelihood
        from likelihood.surrogate import SurrogateMetadata
        from profiles.workflow import run_workflow
        from profiles.evaluation import aggregate_counts

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            out = Path(temporary) / "study"
            for folder in ["likelihood_input", "training_data", "trained_models"]:
                (run / folder).mkdir(parents=True)
            info = dict(
                likelihood={
                    "gaussian_mixture": dict(
                        means=[[0.0, 0.0, 0.0]],
                        covs=[np.eye(3).tolist()],
                        input_params_prefix="x",
                    )
                },
                params={
                    f"x_{i}": dict(prior=dict(min=-4.0, max=4.0)) for i in range(3)
                },
            )
            path = run / "likelihood_input/toy.yaml"
            path.write_text(yaml.safe_dump(info))
            (run / "run.yaml").write_text(
                yaml.safe_dump(
                    dict(likelihood=dict(wrapper="cobaya", input="toy.yaml"))
                )
            )
            backend = CobayaLikelihood(str(path))
            SurrogateMetadata.from_likelihood(backend).save(run / "metadata.json")
            x = np.random.default_rng(7).uniform(-3, 3, (80, 3)).astype(np.float32)
            y = np.array([backend.loglkl(p) for p in x])
            peak = backend.loglkl([0, 0, 0])
            backend.cobaya_model.close()
            np.savetxt(
                run / "training_data/data_it_0.csv",
                np.column_stack([x, y]),
                delimiter=",",
                header="x_0,x_1,x_2,loglkl",
                comments="",
            )
            inp = tf.keras.Input((3,))
            norm = tf.keras.layers.Normalization(mean=[0.0] * 3, variance=[1.0] * 3)(
                inp
            )
            hidden = tf.keras.layers.Dense(3)(norm)
            square = tf.keras.layers.Dot(axes=1)([hidden, hidden])
            outp = tf.keras.layers.Dense(1)(square)
            model = tf.keras.Model(inp, outp)
            model.layers[2].set_weights(
                [np.eye(3, dtype=np.float32), np.zeros(3, np.float32)]
            )
            model.layers[-1].set_weights(
                [np.array([[-0.5]], np.float32), np.array([peak], np.float32)]
            )
            model.save(run / "trained_models/model_it_0.keras")
            c = ProfileConfig(
                dimension=2,
                parameters=["x_0", "x_1"],
                retrain=True,
                added_points=12,
                ranges={"x_0": [-2.0, 2.0], "x_1": [-2.0, 2.0]},
                design_points=3,
                profile_points=3,
                validation_global=8,
                validation_nodes=3,
                first_epochs=2,
                polish_epochs=2,
                workers=2,
            )
            # The legacy scan feeds the same true stage (no retraining: the saved network).
            legacy_out = Path(temporary) / "legacy"
            run_workflow(run, legacy_out, ProfileConfig(
                dimension=2, parameters=["x_0", "x_1"], surrogate_optimizer="legacy",
                ranges={"x_0": [-2.0, 2.0], "x_1": [-2.0, 2.0]}, profile_points=3, workers=2))
            truth = json.loads((legacy_out / "true_surface.json").read_text())
            self.assertEqual(truth["method"], "joint surrogate-corrected residual Newton")
            xx, yy = np.meshgrid(*truth["axes"], indexing="ij")
            np.testing.assert_allclose(2 * (truth["best_logl"] - np.array(truth["logl"])), xx**2 + yy**2, atol=0.03)
            run_workflow(run, out, c)
            self.assertEqual(len(np.load(out / "augmented_data.npy")), 92)
            truth = json.loads((out / "true_surface.json").read_text())
            self.assertEqual(truth["method"], "joint surrogate-corrected residual Newton")
            xx, yy = np.meshgrid(*truth["axes"], indexing="ij")
            np.testing.assert_allclose(
                2 * (truth["best_logl"] - np.array(truth["logl"])),
                xx**2 + yy**2,
                atol=0.03,
            )
            np.testing.assert_array_equal(np.array(truth["xopt"])[:, :, 0], xx)
            np.testing.assert_array_equal(np.array(truth["xopt"])[:, :, 1], yy)
            val = dict(np.load(out / "validation.npz"))
            data = np.load(out / "augmented_data.npy")
            self.assertFalse(
                set(map(tuple, data[:, :-1]))
                & set(
                    map(
                        tuple,
                        np.vstack(
                            [val["global_data"][:, :-1], val["ridge_data"][:, :-1]]
                        ),
                    )
                )
            )
            self.assertTrue((out / "surface.png").exists())
            self.assertTrue((out / "isocontours.png").exists())
            diagnostics = json.loads((out / "surface_diagnostics.json").read_text())
            np.testing.assert_allclose(
                diagnostics["contour_levels"], [2.2957489289, 6.1800743062], atol=1e-9
            )
            count = aggregate_counts(out / "calls")["backend_attempts"]
            (out / "complete.json").unlink()
            run_workflow(run, out, c, resume=True)
            self.assertEqual(aggregate_counts(out / "calls")["backend_attempts"], count)

            # The same study with part of the box refused: the nuisance draws and held-out
            # points beyond a value of x_2 are left out, and the study still trains and reports.
            # The value lies above the rows the label audit recomputes, which must not be refused.
            limit = float(x[np.argsort(y)[-c.audit_points:], 2].max()) + 0.01
            info["prior"] = dict(refuses=f"lambda x_2: -float('inf') if x_2 > {limit} else 0.0")
            path.write_text(yaml.safe_dump(info))
            refused_out = Path(temporary) / "refused"
            run_workflow(run, refused_out, c)
            listed = json.loads((refused_out / "refused_points.json").read_text())
            refused = [x for points in listed.values() for x in points]
            self.assertGreater(len(refused), 0)
            self.assertTrue(all(point[2] > limit for point in refused), refused)
            data = np.load(refused_out / "augmented_data.npy")
            self.assertTrue(np.isfinite(data).all())
            self.assertEqual(len(data), 92 - len(listed.get("augmentation/pair", [])))
            val = dict(np.load(refused_out / "validation.npz"))
            self.assertTrue(np.isfinite(val["global_data"]).all() and np.isfinite(val["ridge_data"]).all())
            for key in ("profile_pairs", "nuisance_pairs"):
                self.assertGreater(len(val[key]), 0)
                self.assertLess(val[key].max(), len(val["ridge_data"]))
            self.assertTrue((refused_out / "complete.json").exists())
            warnings = json.loads((refused_out / "surface_diagnostics.json").read_text())["warnings"]
            self.assertTrue(any("design or validation point(s)" in w for w in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
