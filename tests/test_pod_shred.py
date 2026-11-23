import unittest

import torch

from get_shredded.model import PODSHRED, TimeSeriesDataset, fit


def _snapshots(n=30, points=12, rank=3):
    torch.manual_seed(0)
    u = torch.randn(n, rank) @ torch.randn(rank, points) + 2.0
    v = torch.randn(n, rank) @ torch.randn(rank, points) - 1.0
    return torch.cat([u, v], dim=1)  # [u block; v block], each of rank `rank` plus a mean


def _model(X, rank, scale=None, offset=None):
    m = X.shape[1]
    scale = torch.ones(m) if scale is None else scale
    offset = torch.zeros(m) if offset is None else offset
    return PODSHRED(4, X, 2, rank, scale, offset, hidden_size=8, hidden_layers=1, l1=6, l2=6)


class PODSHREDTest(unittest.TestCase):
    def test_exact_rank_reconstructs_training_snapshots(self):
        X = _snapshots()
        model = _model(X, rank=3)
        torch.testing.assert_close(model.decode(model.coefficient_targets(X)), X, atol=1e-4, rtol=1e-4)

    def test_lower_rank_is_a_projection_with_scaled_coefficients(self):
        X = _snapshots()
        a = _model(X, rank=2).coefficient_targets(X)
        self.assertEqual(a.shape, (30, 4))
        self.assertAlmostEqual(a.min().item(), 0.0, places=5)  # MinMax per coefficient on the training rows
        self.assertAlmostEqual(a.max().item(), 1.0, places=5)

    def test_forward_is_scaled_field(self):
        X = _snapshots()
        scale, offset = torch.full((24,), 2.0), torch.full((24,), 0.5)
        model = _model(X, 2, scale, offset)
        x = torch.randn(5, 7, 4)
        out = model(x)
        self.assertEqual(out.shape, (5, 24))
        torch.testing.assert_close(out, model.decode(model.shred(x)))
        unscaled = _model(X, 2).decode(model.shred(x))
        torch.testing.assert_close(out, unscaled * 2.0 + 0.5)
        self.assertEqual(model.latent(x).shape, (5, 8))

    def test_rank_too_large_raises(self):
        with self.assertRaises(ValueError):
            _model(_snapshots(n=5), rank=6)

    def test_coefficient_training_runs(self):
        X = _snapshots()
        model = _model(X, 2)
        x = torch.randn(30, 7, 4)
        y = model.coefficient_targets(X)
        fit(model.shred, TimeSeriesDataset(x, y), TimeSeriesDataset(x, y), num_epochs=2, batch_size=8)


if __name__ == "__main__":
    unittest.main()
