import unittest

import numpy as np
import torch

from get_shredded.model import SHRED, TimeSeriesDataset, fit
from get_shredded.robust_shred import ResidualSHREDv1


def _model(**kw):
    torch.manual_seed(0)
    # 3 probes, 2 channels, 5 output points per channel; probe j of channel c sits at output c*5 + [0, 2, 4][j]
    index = [0, 2, 4, 5, 7, 9]
    return ResidualSHREDv1(3, 10, torch.rand(3, 2), index, input_channels=2, hidden_size=8, num_heads=2,
                           embed_dim=8, num_frequencies=2, l1=10, l2=9, null_token=True, **kw)


class ResidualV1Test(unittest.TestCase):
    def test_pass1_is_network_with_zero_residual(self):
        model = _model().eval()
        x = torch.randn(4, 5, 6)
        with torch.no_grad():
            np.testing.assert_array_equal(model.pass1(x).numpy(),
                                          model.net(torch.cat([x, torch.zeros_like(x)], -1)).numpy())

    def test_pass2_uses_residual_at_probe(self):
        model = _model().eval()
        x = torch.randn(4, 5, 6)
        with torch.no_grad():
            first = model.pass1(x)
            r = x[:, -1, :] - first[:, [0, 2, 4, 5, 7, 9]]
            second = model.net(torch.cat([x, r[:, None, :].expand(-1, 5, -1)], -1))
            np.testing.assert_allclose(model(x).numpy(), second.numpy(), atol=1e-6)

    def test_no_gradient_through_residual(self):
        model = _model().train()
        x = torch.randn(4, 5, 6, requires_grad=True)
        out = model(x)
        # d out / d x through r is blocked: equals the gradient with r treated as a constant
        r = model._second_input(x, model.pass1(x))[..., 6:].detach()
        ref = model.net(torch.cat([x, r], -1))
        g1 = torch.autograd.grad(out.sum(), x)[0]
        g2 = torch.autograd.grad(ref.sum(), x)[0]
        np.testing.assert_allclose(g1.numpy(), g2.numpy(), atol=1e-5)

    def test_pass1_loss_only_in_training(self):
        model = _model(pass1_weight=2.0)
        x, y = torch.randn(4, 5, 6), torch.randn(4, 10)
        model.train()
        model(x)
        expected = 2.0 * torch.mean((model.pass1(x) - y) ** 2)
        self.assertAlmostEqual(model.target_aux_loss(y).item(), expected.item(), places=5)
        model.eval()
        model(x)
        self.assertEqual(float(model.target_aux_loss(y)), 0.0)

    def test_fit_trains_and_other_models_have_no_hook(self):
        self.assertFalse(hasattr(SHRED(6, 10), "target_aux_loss"))
        model = _model()
        x, y = torch.randn(32, 5, 6), torch.rand(32, 10)
        history = fit(model, TimeSeriesDataset(x, y), TimeSeriesDataset(x[:8], y[:8]), batch_size=16, num_epochs=40)
        self.assertGreater(len(history), 0)
        self.assertLess(float(history[-1]), float(history[0]))


if __name__ == "__main__":
    unittest.main()
