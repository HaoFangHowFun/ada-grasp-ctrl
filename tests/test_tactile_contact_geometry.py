"""First-row contract and independently known geometry transforms; no hardware."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

from ada_grasp_ctrl.tactile_preview.data import LatestFrames, first_contact
from ada_grasp_ctrl.tactile_preview.geometry import (
    CHECKS,
    SDK_LIBRARY,
    GeometryCatalog,
    GeometryMap,
    sdk_hb1_arrays,
    transform_contact,
)
from test_tactile_preview import message


def profile(**overrides: object) -> GeometryMap:
    """Three-by-four map with nonsymmetric coordinates and upward normals."""
    rows, cols = np.indices((3, 4))
    xyz = np.stack((cols, 10 * rows, np.full_like(rows, 2)), axis=-1).astype(float)
    normals = np.zeros_like(xyz)
    normals[:, :, 2] = 2
    options = dict(
        xyz=xyz,
        normals=normals,
        valid_mask=np.ones((3, 4), dtype=bool),
        meters_per_unit=0.001,
        T_E_S=np.eye(4),
        uv_to_col_row=[[1, 0, 0], [0, 1, 0]],
        verified=True,
        evidence={k: "Synthetic known geometry" for k in CHECKS},
        source="test",
        units="mm",
    )
    options.update(overrides)
    return GeometryMap(**options)


class ContactBlockTests(unittest.TestCase):
    def test_first_row_preserves_unknown_columns_and_shape(self) -> None:
        for shape, values, count, row in (
            ([1, 3], [124.8046875, 134.8828125, 11.578125], 1, (124.8046875, 134.8828125, 11.578125)),
            ([2, 4], [1.0, 2.0, 3.0, 4.0, 99.0, 98.0, 97.0, 96.0], 2, (1.0, 2.0, 3.0, 4.0)),
            ([1, 2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, (1.0, 2.0, 3.0)),
            ([3], [1.0, 2.0, 3.0], 1, (1.0, 2.0, 3.0)),
        ):
            result = first_contact(NS(status=1, shape=shape, raw_values=values, error=""))
            self.assertEqual(result["contact_status"], "VALID")
            self.assertEqual(result["contact_shape"], tuple(shape))
            self.assertEqual(result["contact_count"], count)
            self.assertEqual(result["contact_row"], row)

    def test_missing_empty_invalid_malformed_and_nonfinite(self) -> None:
        for status, shape, values, expected in (
            (0, [], [], "MISSING"),
            (2, [0, 3], [], "EMPTY"),
            (1, [0, 3], [], "EMPTY"),
            (3, [], [], "INVALID"),
            (1, [1, 3], [1.0, 2.0], "INVALID"),
            (1, [1, 1], [1.0], "INVALID"),
            (1, [], [], "INVALID"),
            (1, [1, 3], [1.0, float("nan"), 3.0], "INVALID"),
            (1, [2, 2], [1.0, 2.0, 3.0, float("inf")], "INVALID"),
            (2, [1, 3], [1.0, 2.0, 3.0], "INVALID"),
            (0, [], [1.0], "INVALID"),
            (255, [1, 3], [], "INVALID"),
            (1, [-1, 3], [], "INVALID"),
        ):
            with self.subTest(status=status, shape=shape, values=values):
                result = first_contact(NS(status=status, shape=shape, raw_values=values, error="original error"))
                self.assertEqual(result["contact_status"], expected)
                self.assertEqual(result["contact_row"], ())

    def test_atomic_frame_replacement_and_immutable_row(self) -> None:
        cache = LatestFrames()
        msg = message()
        cache.receive(msg, "index", 100.0)
        msg.contact_points.raw_values[0] = 99.0
        saved = cache.snapshot()["frames"]["index"]
        self.assertEqual(saved["contact_row"][0], 124.8046875)
        msg.sdk_frame_id = 500
        msg.f6.raw_values = [5.0] * 6
        msg.contact_points.status = 0
        msg.contact_points.raw_values = []
        cache.receive(msg, "index", 100.0)
        saved = cache.snapshot()["frames"]["index"]
        self.assertEqual((saved["sdk_frame_id"], saved["raw"]), (500, (5.0,) * 6))
        self.assertEqual((saved["contact_status"], saved["contact_row"]), ("MISSING", ()))


class GeometryTests(unittest.TestCase):
    def test_nearest_rounding_axes_edges_and_holes(self) -> None:
        geometry = profile()
        result = geometry.lookup([1.5, 0.5])
        self.assertEqual((result.row, result.col), (1, 2))
        np.testing.assert_allclose(result.p_E, [0.002, 0.01, 0.002])
        np.testing.assert_allclose(result.n_E, [0, 0, 1])
        self.assertEqual((geometry.lookup([3, 2]).row, geometry.lookup([3, 2]).col), (2, 3))
        for uv in ([-0.001, 0], [3.001, 0], [0, 2.01], [np.nan, 1], [1, np.inf]):
            with self.assertRaises(ValueError):
                geometry.lookup(uv)
        mask = np.ones((3, 4), dtype=bool)
        mask[1, 2] = False
        with self.assertRaisesRegex(ValueError, "hole"):
            profile(valid_mask=mask).lookup([1.5, 0.5])
        swapped = profile(uv_to_col_row=[[0, 1, 0], [1, 0, 0]])
        self.assertEqual((swapped.lookup([1, 2]).row, swapped.lookup([1, 2]).col), (1, 2))

    def test_two_nonidentity_rigid_transforms_points_and_vectors(self) -> None:
        t_es = np.array([[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]])
        result = profile(T_E_S=t_es).lookup([2, 1])
        np.testing.assert_allclose(result.p_E, [0.09, 0.202, 0.302])
        t_we = np.array([[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]])
        p, n, f = transform_contact(t_we, result.p_E, result.n_E, [1, -2, 3])
        np.testing.assert_allclose(p, [1.09, 1.698, 3.202])
        np.testing.assert_allclose(n, [0, -1, 0])
        np.testing.assert_allclose(f, [1, -3, -2])
        self.assertAlmostEqual(np.linalg.norm(n), 1)
        self.assertAlmostEqual(np.linalg.norm(f), np.sqrt(14))

    def test_unverified_and_invalid_geometry_refuse(self) -> None:
        unverified = profile(verified=False, evidence={})
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            unverified.lookup([1, 1])
        self.assertEqual(unverified.lookup([1, 1], True).verification, "UNVERIFIED DIAGNOSTIC")
        for kwargs in (
            dict(evidence={}),
            dict(meters_per_unit=0),
            dict(T_E_S=np.eye(4) * 2),
            dict(valid_mask=np.ones((1, 2))),
            dict(uv_to_col_row=[[0, 0, 0], [0, 0, 0]]),
        ):
            with self.assertRaises(ValueError):
                profile(**kwargs)
        normals = np.zeros((3, 4, 3))
        with self.assertRaises(ValueError):
            profile(normals=normals).lookup([1, 1])
        xyz = np.zeros((3, 4, 3))
        xyz[1, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            profile(xyz=xyz).lookup([1, 1])
        with self.assertRaises(ValueError):
            transform_contact(np.eye(4), [1, 2, 3], [0, 0, 0], [1, 2, 3])

    def test_explicit_config_and_independent_missing_finger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = profile()
            for name, array in (("xyz", p.xyz), ("normals", p.normals), ("valid_mask", p.mask)):
                np.save(root / f"{name}.npy", array)
            config = dict(
                schema_version=1,
                fingers={"index": "test"},
                profiles={
                    "test": dict(
                        xyz="xyz.npy",
                        normals="normals.npy",
                        valid_mask="valid_mask.npy",
                        meters_per_unit=0.001,
                        T_E_S=np.eye(4).tolist(),
                        uv_to_col_row=[[1, 0, 0], [0, 1, 0]],
                        verified=True,
                        evidence=p.evidence,
                        source="synthetic",
                        units="mm",
                    )
                },
            )
            path = root / "geometry.json"
            path.write_text(json.dumps(config))
            catalog = GeometryCatalog.from_config(path)
            self.assertIn("index", catalog.maps)
            self.assertIn("thumb", catalog.errors)
            with self.assertRaises(ValueError):
                catalog.lookup("thumb", [1, 1], True)
            self.assertEqual(catalog.lookup("index", [1, 1], False).row, 1)
            del config["profiles"]["test"]["T_E_S"]
            path.write_text(json.dumps(config))
            self.assertFalse(GeometryCatalog.from_config(path).maps)

    def test_missing_or_different_library_never_guesses_offsets(self) -> None:
        self.assertFalse(GeometryCatalog.installed_candidate(Path("/nonexistent/library.so")).maps)
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"not the audited SDK")
            stream.flush()
            self.assertFalse(GeometryCatalog.installed_candidate(Path(stream.name)).maps)

    @unittest.skipUnless(SDK_LIBRARY.is_file(), "Local SDK static asset not installed")
    def test_audited_static_maps_and_finger_types(self) -> None:
        maps = sdk_hb1_arrays(SDK_LIBRARY)
        self.assertEqual(set(maps), {"general_point", "general_normal", "thumb_point", "thumb_normal"})
        self.assertTrue(all(a.shape == (240, 240, 3) for a in maps.values()))
        catalog = GeometryCatalog.installed_candidate()
        self.assertIs(catalog.maps["index"], catalog.maps["pinky"])
        self.assertIsNot(catalog.maps["index"], catalog.maps["thumb"])
        self.assertFalse(catalog.maps["thumb"].verified)
        for f in ("index", "thumb"):
            with self.assertRaises(ValueError):
                catalog.lookup(f, [120, 120], False)
            self.assertTrue(np.isfinite(catalog.lookup(f, [120, 120], True).p_E).all())


if __name__ == "__main__":
    unittest.main()
