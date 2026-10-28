import unittest

import torch

from get_shredded.model import SHRED
from get_shredded.robust_shred import RobustSHREDv1, RobustSHREDv2
from get_shredded.senseiver import SenseiverSDN

NUM_SENSORS = 6
CHANNELS = 2
LAGS = 4


class _MaskedModel(torch.nn.Module):
    """Uniform (x, mask) interface: SenseiverSDN only sees the last snapshot."""

    def __init__(self, model: torch.nn.Module, coords: torch.Tensor | None = None) -> None:
        super().__init__()
        self.model = model
        self.coords = coords

    def forward(self, x, mask=None):
        if self.coords is None:
            return self.model(x, mask)
        values = x[:, -1, :].reshape(x.shape[0], CHANNELS, NUM_SENSORS).transpose(1, 2)
        return self.model(values, self.coords, mask)

    def latent(self, x, mask=None):
        if self.coords is None:
            return self.model.latent(x, mask)
        values = x[:, -1, :].reshape(x.shape[0], CHANNELS, NUM_SENSORS).transpose(1, 2)
        return self.model.latent(values, self.coords, mask)


def _models(null_token: bool = True) -> dict[str, _MaskedModel]:
    torch.manual_seed(0)
    coords = torch.rand(NUM_SENSORS, 2)
    common = dict(
        num_sensors=NUM_SENSORS, output_size=11, sensor_coordinates=coords, hidden_size=8,
        num_heads=2, num_frequencies=2, l1=10, l2=9, input_channels=CHANNELS, null_token=null_token,
    )
    models = {
        "v1": _MaskedModel(RobustSHREDv1(embed_dim=8, **common)),
        "v2": _MaskedModel(RobustSHREDv2(num_latents=3, latent_dim=4, hidden_layers=1, **common)),
        "senseiver_sdn": _MaskedModel(
            SenseiverSDN(11, input_channels=CHANNELS, num_latents=3, latent_dim=8, num_frequencies=2,
                         num_heads=2, l1=10, l2=9, null_token=null_token),
            coords,
        ),
    }
    for model in models.values():
        model.eval()
    return models


class MaskedAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.x = torch.randn(5, LAGS, CHANNELS * NUM_SENSORS)
        self.mask = torch.zeros(5, NUM_SENSORS, dtype=torch.bool)
        self.mask[0, [0, 3]] = True
        self.mask[1, 1:] = True  # only sensor 0 remains
        self.mask[2, 5] = True

    def test_nothing_masked_equals_unmasked(self) -> None:
        empty = torch.zeros(5, NUM_SENSORS, dtype=torch.bool)
        for name, model in _models().items():
            with torch.no_grad():
                torch.testing.assert_close(model(self.x, empty), model(self.x), msg=name)

    def test_masked_sensor_values_have_no_effect(self) -> None:
        changed = self.x.clone()
        full = self.mask.repeat(1, CHANNELS)  # both channels of a failed sensor
        changed[full.unsqueeze(1).expand_as(changed)] = 1e3
        for name, model in _models().items():
            with torch.no_grad():
                torch.testing.assert_close(model(changed, self.mask), model(self.x, self.mask), msg=name)
                self.assertFalse(torch.allclose(model(changed), model(self.x)), msg=name)

    def test_single_remaining_sensor_gives_finite_output(self) -> None:
        for name, model in _models().items():
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(self.x, self.mask)).all(), msg=name)

    def test_mask_without_null_token_still_excludes_sensors(self) -> None:
        changed = self.x.clone()
        changed[0, :, 0] = 1e3
        mask = torch.zeros(5, NUM_SENSORS, dtype=torch.bool)
        mask[0, 0] = True
        for name, model in _models(null_token=False).items():
            with torch.no_grad():
                torch.testing.assert_close(model(changed, mask), model(self.x, mask), msg=name)

    def test_gradients_reach_null_token(self) -> None:
        for name, model in _models().items():
            model.train()
            model(self.x, self.mask).square().mean().backward()
            grads = [p.grad for n, p in model.named_parameters() if "null" in n]
            self.assertTrue(grads and all(g is not None for g in grads), msg=name)


class LatentReadoutTest(unittest.TestCase):
    def test_forward_is_decoder_of_latent(self) -> None:
        x = torch.randn(3, LAGS, CHANNELS * NUM_SENSORS)
        for name, model in _models().items():
            with torch.no_grad():
                latent = model.latent(x)
                torch.testing.assert_close(model.model.decoder(latent), model(x), msg=name)
        self.assertEqual(_models()["v1"].latent(x).shape, (3, 8))
        self.assertEqual(_models()["v2"].latent(x).shape, (3, 8))
        self.assertEqual(_models()["senseiver_sdn"].latent(x).shape, (3, 3 * 8))

    def test_shred_latent_matches_forward(self) -> None:
        torch.manual_seed(0)
        model = SHRED(4, 7, hidden_size=8, hidden_layers=2, l1=10, l2=9).eval()
        x = torch.randn(3, LAGS, 4)
        with torch.no_grad():
            h = model.latent(x)
            out = model.linear3(torch.relu(model.linear2(torch.relu(model.linear1(h)))))
            torch.testing.assert_close(out, model(x))
        self.assertEqual(h.shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
