import unittest

import numpy as np

from get_shredded.data import gappy_pod_reconstruct, qr_place, qr_place_probes, qrpod_reconstruct


def _data(m: int = 60, n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, 6)) @ rng.normal(size=(6, n)) + 0.01 * rng.normal(size=(m, n))


class GappyPODTest(unittest.TestCase):
    def test_no_missing_matches_qrpod(self) -> None:
        X = _data()
        locs, U_r = qr_place(X, 5)
        y = X.T[:, locs]
        np.testing.assert_allclose(
            gappy_pod_reconstruct(y, None, locs, U_r), qrpod_reconstruct(y, locs, U_r, X.shape[0]),
            rtol=1e-6, atol=1e-8,
        )

    def test_missing_values_are_ignored_and_rank_reduced(self) -> None:
        X = _data()
        locs, U_r = qr_place(X, 5)
        y = X.T[:3, locs].copy()
        missing = np.zeros_like(y, dtype=bool)
        missing[0, [1, 4]] = True
        missing[2, 1:] = True  # one measurement left -> rank 1
        base = gappy_pod_reconstruct(y, missing, locs, U_r)
        y[missing] = 1e6
        np.testing.assert_allclose(gappy_pod_reconstruct(y, missing, locs, U_r), base)
        expected = U_r[:, :1] * (y[2, 0] / U_r[locs[0], 0])
        np.testing.assert_allclose(base[2], expected.ravel(), rtol=1e-6)
        self.assertTrue(np.isfinite(base).all())


class QRProbePlacementTest(unittest.TestCase):
    def test_single_channel_matches_qr_place(self) -> None:
        X = _data()
        locs, U_r = qr_place_probes(X, 5, rank=5, n_points=X.shape[0])
        ref_locs, ref_U = qr_place(X, 5)
        np.testing.assert_array_equal(locs, ref_locs)
        np.testing.assert_allclose(U_r, ref_U)

    def test_two_channel_probes_are_distinct_points(self) -> None:
        X = _data(m=80)  # two channel blocks of 40 points
        locs, U_r = qr_place_probes(X, 6, rank=12, n_points=40)
        self.assertEqual(len(set(locs.tolist())), 6)
        self.assertTrue((locs >= 0).all() and (locs < 40).all())
        self.assertEqual(U_r.shape, (80, 12))


if __name__ == "__main__":
    unittest.main()
