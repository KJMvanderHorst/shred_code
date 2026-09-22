import unittest

import torch

from get_shredded.model import TimeSeriesDataset, fit
from get_shredded.robust_shred import RobustSHREDv1, RobustSHREDv2


class RobustSHREDTest(unittest.TestCase):
    def _models(self):
        coords = torch.rand(5, 2)
        common = dict(
            num_sensors=5,
            output_size=7,
            sensor_coordinates=coords,
            hidden_size=8,
            num_heads=2,
            num_frequencies=2,
            l1=10,
            l2=9,
        )
        return (
            RobustSHREDv1(embed_dim=8, **common),
            RobustSHREDv2(num_latents=3, latent_dim=4, hidden_layers=1, **common),
        )

    def test_forward_and_gradients(self):
        for model in self._models():
            output = model(torch.randn(3, 4, 5))
            self.assertEqual(output.shape, (3, 7))
            output.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_v2_encoder_configuration(self):
        model = self._models()[1]
        self.assertEqual(model.encoder.num_latents, 3)
        self.assertEqual(model.encoder.latent_dim, 4)
        self.assertEqual(model.encoder.num_frequencies, 2)

    def test_fit_integration(self):
        for model in self._models():
            dataset = TimeSeriesDataset(torch.randn(6, 3, 5), torch.randn(6, 7))
            history = fit(model, dataset, dataset, batch_size=3, num_epochs=1, patience=2)
            self.assertEqual(history.ndim, 1)


if __name__ == "__main__":
    unittest.main()
