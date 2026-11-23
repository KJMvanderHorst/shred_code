import unittest

import torch

from get_shredded.model import PODTrunkSHRED, fit, TimeSeriesDataset


def _grid_data(n=40):
    xs, ys = torch.meshgrid(torch.linspace(0.0, 2.0, 8), torch.linspace(-1.0, 1.0, 6), indexing="ij")
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (48, 2)
    x, y = coords[:, 0], coords[:, 1]
    modes = torch.stack([torch.sin(torch.pi * x / 2) * torch.cos(torch.pi * y / 2), torch.cos(torch.pi * x / 2) * y])
    torch.manual_seed(0)
    u = 1.0 + torch.randn(n, 2) @ modes
    v = 0.2 * x + torch.randn(n, 2) @ modes.flip(0)
    return torch.cat([u, v], dim=1), coords


def _model(steps=1500, physics_weight=0.0, train_trunk=False):
    X, coords = _grid_data()
    torch.manual_seed(0)
    return PODTrunkSHRED(4, X, 2, 2, torch.ones(96), torch.zeros(96), coords, trunk_width=64, num_frequencies=3,
                         fit_steps=steps, fit_lr=3e-3, physics_weight=physics_weight, n_colloc=5, train_trunk=train_trunk,
                         hidden_size=8, hidden_layers=1, l1=6, l2=6)


class PODTrunkSHREDTest(unittest.TestCase):
    def test_trunk_fits_smooth_basis_and_is_frozen(self):
        model = _model()
        report = model.fit_report
        self.assertEqual(len(report["mode_error"]), 2)
        self.assertLess(max(max(e) for e in report["mode_error"]), 2e-2)
        self.assertLess(report["mean_error"], 2e-2)
        self.assertLess(report["gram_max_dev"], 5e-2)
        self.assertTrue(all(not p.requires_grad for p in model.trunk.parameters()))
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(trainable, {id(p) for p in model.shred.parameters()})

    def test_decode_matches_trunk_at_the_grid(self):
        model = _model(steps=50)
        x = torch.randn(5, 7, 4)
        s = model.shred(x)
        out = model(x)
        self.assertEqual(out.shape, (5, 96))
        uv = model.field_at(s, model.coords.expand(5, -1, -1))  # (5, 48, 2)
        torch.testing.assert_close(out, uv.permute(0, 2, 1).reshape(5, 96), atol=1e-5, rtol=1e-5)

    def test_aux_loss_and_divergence_report(self):
        x = torch.randn(5, 7, 4)
        self.assertEqual(_model(steps=20).aux_loss(x).item(), 0.0)
        model = _model(steps=20, physics_weight=1.0)
        loss = model.aux_loss(x)
        self.assertTrue(torch.isfinite(loss) and loss.item() > 0)
        loss.backward()
        self.assertIsNotNone(model.shred.linear3.weight.grad)
        rel = model.divergence_rel(x, torch.rand(9, 2))
        self.assertEqual(set(rel), {"held_out", "grid"})
        self.assertTrue(all(v >= 0 for v in rel.values()))

    def test_trained_trunk_gets_gradients_and_decode_follows_it(self):
        model = _model(steps=20, physics_weight=1.0, train_trunk=True)
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertTrue({id(p) for p in model.trunk.parameters()} <= trainable)
        x = torch.randn(5, 7, 4)
        (model(x).square().mean() + model.aux_loss(x)).backward()
        self.assertTrue(all(p.grad is not None for p in model.trunk.parameters()))
        with torch.no_grad():
            model.trunk[-1].bias.add_(0.1)  # a trunk update must reach decode (no stale grid cache)
            out, uv = model(x), model.field_at(model.shred(x), model.coords.expand(5, -1, -1))
        torch.testing.assert_close(out, uv.permute(0, 2, 1).reshape(5, 96), atol=1e-5, rtol=1e-5)

    def test_coefficient_training_runs(self):
        model = _model(steps=20)
        X, _ = _grid_data()
        x = torch.randn(40, 7, 4)
        y = model.coefficient_targets(X)
        fit(model.shred, TimeSeriesDataset(x, y), TimeSeriesDataset(x, y), num_epochs=2, batch_size=8)


if __name__ == "__main__":
    unittest.main()
