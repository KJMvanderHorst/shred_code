import math
import unittest

import torch

from get_shredded.model import TimeSeriesDataset, TrunkSHRED, fit, wave_residual


def _model(physics_weight=1.0, probe_weight=0.0):
    torch.manual_seed(0)
    x_grid = torch.linspace(0.0, 2.0, 11)
    return TrunkSHRED(3, x_grid, torch.full((11,), 2.0), torch.full((11,), 0.5), c=2.0, L=2.0, dt=0.004,
                      lags=4, hidden_size=8, hidden_layers=1, p=5, trunk_width=8,
                      physics_weight=physics_weight, n_colloc=6, num_frequencies=3,
                      probe_idx=[2, 5, 9], probe_weight=probe_weight)


class TrunkSHREDTest(unittest.TestCase):
    def test_wave_residual_exact_mode(self):
        x = torch.rand(50, dtype=torch.float64, requires_grad=True) * 2.0
        t = torch.rand(50, dtype=torch.float64, requires_grad=True)
        k, c, L = 3, 2.0, 2.0
        solution = torch.sin(k * math.pi * x / L) * torch.cos(k * math.pi * c / L * t)
        self.assertLess(wave_residual(solution, x, t, c).abs().max().item(), 1e-9)
        static = torch.sin(k * math.pi * x / L) + 0 * t**3  # u_tt = 0 but u_xx != 0: not a solution
        self.assertGreater(wave_residual(static, x, t, c).abs().max().item(), 1.0)

    def test_forward_is_scaled_field_at_tau_zero(self):
        model = _model()
        x = torch.randn(7, 4, 3)
        out = model(x)
        self.assertEqual(out.shape, (7, 11))
        u = model.field(model.coefficients(x), model.x_grid, torch.zeros(11))
        torch.testing.assert_close(out, u * 2.0 + 0.5)
        self.assertEqual(model.latent(x).shape, (7, 8))

    def test_aux_loss(self):
        x = torch.randn(7, 4, 3)
        self.assertEqual(_model(0.0).aux_loss(x).item(), 0.0)
        model = _model(1.0)
        loss = model.aux_loss(x)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.trunk.parameters()))

    def test_probe_field_alignment(self):
        model = _model()
        x = torch.randn(7, 4, 3)
        probes = model.probe_field(model.coefficients(x))
        self.assertEqual(probes.shape, (7, 4, 3))
        torch.testing.assert_close(probes[:, -1, :], model(x)[:, [2, 5, 9]])  # last lag is tau = 0

    def test_probe_loss(self):
        x = torch.randn(7, 4, 3)
        model = _model(0.0, probe_weight=1.0)
        loss = model.aux_loss(x)
        expected = (model.probe_field(model.coefficients(x)) - x).square().mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.trunk.parameters()))

    def test_fit_runs(self):
        model = _model()
        ds = TimeSeriesDataset(torch.randn(16, 4, 3), torch.rand(16, 11))
        history = fit(model, ds, ds, batch_size=8, num_epochs=2)
        self.assertTrue(torch.isfinite(history).all())


if __name__ == "__main__":
    unittest.main()
