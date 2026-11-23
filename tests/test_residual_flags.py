import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from get_shredded.residual_flags import detection_metrics, fit_thresholds, joint_score, probe_residuals
from get_shredded.robust_shred import RobustSHREDv1


class ResidualTest(unittest.TestCase):
    def test_residual_in_physical_units(self):
        # 3 grid points, 2 channels, probes at points 0 and 2
        X = np.array([[0.0, 1.0, 2.0, 10.0, 11.0, 12.0], [4.0, 5.0, 6.0, 30.0, 31.0, 32.0]])
        scaler = MinMaxScaler().fit(X)
        state = np.array([0, 2, 3, 5])
        sd = SimpleNamespace(channels=2, num_sensors=2, state_indices=state, scaler=scaler,
                             sensor_range=(scaler.data_max_ - scaler.data_min_)[state])
        truth = np.array([[2.0, 4.0, 20.0, 25.0]])          # physical readings
        x = scaler.transform(np.array([[2.0, 0, 4.0, 20.0, 0, 25.0]]))[:, state]
        x = np.concatenate([np.zeros_like(x), x], axis=0)[None]  # (1, lags=2, C*N), last lag = readings
        pred = np.array([[1.0, 0, 4.5, 21.0, 0, 25.0]])
        r = probe_residuals(pred, x, sd)
        np.testing.assert_allclose(r, (truth - pred[:, state]).reshape(1, 2, 2), atol=1e-5)

    def test_thresholds_are_quantiles_of_joint_score(self):
        r = np.random.default_rng(0).standard_normal((500, 2, 3)) * np.array([1.0, 3.0])[None, :, None]
        s, taus = fit_thresholds(r, [0.95, 0.99])
        np.testing.assert_allclose(s, r.std(axis=0))
        d = np.sqrt(((r / s) ** 2).sum(axis=1))
        np.testing.assert_allclose(joint_score(r, s), d)
        np.testing.assert_allclose(taus[0.99], np.quantile(d, 0.99, axis=0))
        self.assertAlmostEqual((d > taus[0.95]).mean(), 0.05, delta=0.01)

    def test_detection_metrics(self):
        truth = np.array([[1, 0, 0], [1, 1, 0]], bool)
        flags = np.array([[1, 1, 0], [0, 1, 0]], bool)
        m = detection_metrics(flags, truth)
        self.assertEqual(m["precision"], 2 / 3)
        self.assertEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertEqual(m["false_alarm"], 1 / 3)
        clean = detection_metrics(np.zeros((2, 3), bool), np.zeros((2, 3), bool))
        self.assertIsNone(clean["recall"])
        self.assertEqual(clean["false_alarm"], 0.0)


class AttentionBiasTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = RobustSHREDv1(5, 7, torch.rand(5, 2), hidden_size=8, num_heads=2, embed_dim=8,
                                   num_frequencies=2, l1=10, l2=9, input_channels=2, null_token=True).eval()
        self.x = torch.randn(3, 4, 10)

    def test_zero_bias_is_unmasked(self):
        with torch.no_grad():
            np.testing.assert_array_equal(self.model.latent(self.x, None, bias=torch.zeros(3, 5)).numpy(),
                                          self.model.latent(self.x).numpy())

    def test_floor_bias_matches_hard_exclusion(self):
        mask = torch.tensor([[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        bias = torch.where(mask, torch.tensor(math.log(1e-6)), torch.tensor(0.0))
        with torch.no_grad():
            soft, hard = self.model.latent(self.x, None, bias=bias), self.model.latent(self.x, mask)
        np.testing.assert_allclose(soft.numpy(), hard.numpy(), atol=1e-4)


if __name__ == "__main__":
    unittest.main()


class FlagFracTest(unittest.TestCase):
    def test_default_keeps_fingerprint_dict(self):
        from get_shredded.corruption import CorruptionSpec, spec_dict
        self.assertNotIn("flag_frac", spec_dict(CorruptionSpec()))
        self.assertEqual(spec_dict(CorruptionSpec(fault="garbage", p=0.3, flag_frac=0.5))["flag_frac"], 0.5)
        with self.assertRaises(ValueError):
            CorruptionSpec(flag_frac=1.5)

    def test_augmenter_flags_a_fraction_of_faults(self):
        from get_shredded.corruption import CorruptionSpec, make_augmenter, split_masked_input
        n, c = 8, 2
        x = torch.zeros(4000, 3, c * n)
        for frac, lo, hi in [(1.0, 1.0, 1.0), (0.5, 0.45, 0.55), (0.0, 0.0, 0.0)]:
            torch.manual_seed(0)
            spec = CorruptionSpec(noise=False, fault="garbage", flags="oracle", p=0.3, flag_frac=frac)
            out = make_augmenter(spec, np.zeros(c * n), n, channels=c, mask_input=True)(x)
            values, flags = split_masked_input(out, c * n)
            faulty = (values[:, 0, :n] != 0)          # garbage replaced the zeros (prob. 1)
            self.assertTrue(bool((flags <= faulty).all()))   # only faulty probes are ever flagged
            frac_flagged = flags.sum().item() / faulty.sum().item()
            self.assertGreaterEqual(frac_flagged, lo)
            self.assertLessEqual(frac_flagged, hi)


class TopKTest(unittest.TestCase):
    def test_top_k_flags(self):
        from get_shredded.residual_flags import top_k_flags
        d = np.array([[0.1, 5.0, 0.3, 2.0], [1.0, 0.0, 0.0, 0.0], [3.0, 1.0, 2.0, 0.5]])
        flags = top_k_flags(d, np.array([2, 0, 1]))
        np.testing.assert_array_equal(flags, [[0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
