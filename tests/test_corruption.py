import unittest

import numpy as np
import torch

from get_shredded.corruption import (
    corrupt_numpy,
    make_corruption_augmenter,
    noise_std_from_training,
    sample_dropout_mask,
    sample_dropout_mask_torch,
    split_masked_input,
)


class DropoutMaskTest(unittest.TestCase):
    def test_numpy_mask_rate_and_at_least_one_sensor(self) -> None:
        mask = sample_dropout_mask(20000, 8, 0.3, np.random.default_rng(0))
        self.assertEqual(mask.shape, (20000, 8))
        self.assertEqual(mask.dtype, bool)
        self.assertFalse(mask.all(axis=1).any())
        self.assertAlmostEqual(mask.mean(), 0.3, delta=0.01)
        self.assertGreater(len({row.tobytes() for row in mask}), 100)

    def test_all_failed_rows_are_resampled(self) -> None:
        mask = sample_dropout_mask(5000, 2, 0.9, np.random.default_rng(1))
        self.assertFalse(mask.all(axis=1).any())
        tmask = sample_dropout_mask_torch(5000, 2, 0.9)
        self.assertFalse(bool(tmask.all(dim=1).any()))

    def test_p_zero_is_no_dropout(self) -> None:
        self.assertFalse(sample_dropout_mask(10, 4, 0.0, np.random.default_rng(0)).any())

    def test_reproducible_with_seed(self) -> None:
        a = sample_dropout_mask(50, 8, 0.3, np.random.default_rng(3))
        b = sample_dropout_mask(50, 8, 0.3, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)


class CorruptionTest(unittest.TestCase):
    def test_noise_std_is_fraction_of_training_std(self) -> None:
        train = np.random.default_rng(0).normal(scale=[1.0, 4.0], size=(10000, 2))
        np.testing.assert_allclose(noise_std_from_training(train, 0.05), [0.05, 0.2], rtol=0.05)

    def test_corrupt_numpy_fills_both_channels_for_whole_window(self) -> None:
        x = np.ones((2, 5, 6), dtype=np.float32)  # 2 channels x 3 sensors
        mask = np.array([[True, False, False], [False, False, True]])
        out = corrupt_numpy(x, None, mask, channels=2)
        np.testing.assert_array_equal(out[0, :, [0, 3]], 0.0)
        np.testing.assert_array_equal(out[1, :, [2, 5]], 0.0)
        self.assertEqual(out.sum(), 2 * 5 * 6 - 2 * 5 * 2)

    def test_augmenter_masks_constant_over_lags_and_per_sample(self) -> None:
        torch.manual_seed(0)
        aug = make_corruption_augmenter(np.zeros(16), 0.3, 8, channels=2, append_mask=True)
        out = aug(torch.ones(256, 10, 16))
        values, mask = split_masked_input(out, 16)
        self.assertEqual(values.shape, (256, 10, 16))
        self.assertEqual(mask.shape, (256, 8))
        zero = values == 0
        self.assertTrue(torch.equal(zero, zero[:, :1, :].expand_as(zero)))
        self.assertTrue(torch.equal(zero[:, 0, :8], mask))
        self.assertTrue(torch.equal(zero[:, 0, 8:], mask))
        self.assertGreater(len({tuple(row.tolist()) for row in mask}), 20)

    def test_augmenter_noise_level(self) -> None:
        torch.manual_seed(0)
        aug = make_corruption_augmenter(np.array([0.1, 0.5]), 0.0, 2)
        out = aug(torch.zeros(20000, 2))
        np.testing.assert_allclose(out.std(dim=0).numpy(), [0.1, 0.5], rtol=0.05)


if __name__ == "__main__":
    unittest.main()
