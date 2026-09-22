import unittest

import torch

from get_shredded.senseiver import Senseiver, SenseiverSDN


class SenseiverSDNTest(unittest.TestCase):
    def _make_model(self) -> SenseiverSDN:
        return SenseiverSDN(
            output_size=15,
            num_latents=3,
            latent_dim=8,
            num_frequencies=2,
            num_heads=2,
            l1=12,
            l2=10,
        )

    def test_forward_supports_arbitrary_sensor_counts(self) -> None:
        model = self._make_model()
        for num_sensors in (4, 8, 16):
            values = torch.randn(2, num_sensors, 1)
            coords = torch.rand(num_sensors, 2)
            self.assertEqual(model(values, coords).shape, (2, 15))

    def test_gradients_reach_encoder_and_decoder(self) -> None:
        model = self._make_model()
        output = model(torch.randn(3, 8, 1), torch.rand(8, 2))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.decoder.parameters()))

    def test_encoder_configuration_matches_senseiver(self) -> None:
        senseiver = Senseiver(num_latents=3, latent_dim=8, num_frequencies=2, num_heads=2)
        hybrid = self._make_model()
        self.assertEqual(
            (senseiver.encoder.num_latents, senseiver.encoder.latent_dim, senseiver.encoder.num_frequencies),
            (hybrid.encoder.num_latents, hybrid.encoder.latent_dim, hybrid.encoder.num_frequencies),
        )

    def test_decoder_is_not_coordinate_conditioned(self) -> None:
        model = self._make_model()
        self.assertFalse(hasattr(model, "decoder_cross_attention"))
        self.assertEqual(model.encoder(torch.randn(1, 4, 1), torch.rand(4, 2)).shape, (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
