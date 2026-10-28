import unittest

import numpy as np
import pytest
import torch

from get_shredded.corruption import (
    CorruptionSpec,
    apply_fault,
    make_augmenter,
    noise_std_from_training,
    sample_dropout_mask,
    sample_dropout_mask_torch,
    split_masked_input,
)

MISSING = CorruptionSpec(fault="missing", p=0.3)
GARBAGE = CorruptionSpec(fault="garbage", p=0.3)


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

    def test_missing_fills_both_channels_for_whole_window(self) -> None:
        x = np.ones((2, 5, 6), dtype=np.float32)  # 2 channels x 3 sensors
        mask = np.array([[True, False, False], [False, False, True]])
        out = apply_fault(x, mask, 2, MISSING)
        np.testing.assert_array_equal(out[0, :, [0, 3]], 0.0)
        np.testing.assert_array_equal(out[1, :, [2, 5]], 0.0)
        self.assertEqual(out.sum(), 2 * 5 * 6 - 2 * 5 * 2)

    def test_garbage_replaces_only_faulty_probes_reproducibly(self) -> None:
        channels, n, lags = 2, 4, 3
        x = np.random.default_rng(0).random((5, lags, channels * n)).astype(np.float32)
        mask = np.zeros((5, n), dtype=bool)
        mask[0, 1] = mask[3, [0, 2]] = True
        out = apply_fault(x, mask, channels, GARBAGE, np.random.default_rng(1))
        again = apply_fault(x, mask, channels, GARBAGE, np.random.default_rng(1))
        self.assertEqual(out.dtype, x.dtype)
        np.testing.assert_array_equal(out, again)
        expected = np.tile(mask, (1, channels))
        np.testing.assert_array_equal((out != x).all(axis=1), expected)  # changed at every lag
        np.testing.assert_array_equal((out == x).all(axis=1), ~expected)

    def test_augmenter_masks_constant_over_lags_and_per_sample(self) -> None:
        torch.manual_seed(0)
        aug = make_augmenter(MISSING, np.zeros(16), 8, channels=2, mask_input=True)
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
        aug = make_augmenter(CorruptionSpec(), np.array([0.1, 0.5]), 2)
        out = aug(torch.zeros(20000, 2))
        np.testing.assert_allclose(out.std(dim=0).numpy(), [0.1, 0.5], rtol=0.05)

    def test_clean_spec_adds_no_noise(self) -> None:
        aug = make_augmenter(CorruptionSpec(noise=False), np.array([0.1, 0.5]), 2)
        self.assertTrue(torch.equal(aug(torch.zeros(10, 2)), torch.zeros(10, 2)))


def test_garbage_augmenter_is_silent_and_fails_whole_probes() -> None:
    torch.manual_seed(0)
    spec = CorruptionSpec(fault="garbage", flags="silent", p=0.3)
    out = make_augmenter(spec, np.zeros(8, dtype=np.float32), 4, channels=2, mask_input=True)(torch.full((256, 3, 8), 0.25))
    assert out.shape == (256, 3, 12)
    assert torch.all(out[..., 8:] == 0)
    changed = (out[..., :8] != 0.25).all(dim=1)  # failed columns differ at every lag
    untouched = (out[..., :8] == 0.25).all(dim=1)
    assert torch.all(changed | untouched)
    assert torch.equal(changed[:, :4], changed[:, 4:])  # both channels of a probe fail together
    assert 0.2 < changed[:, :4].float().mean() < 0.4


@pytest.mark.parametrize("kwargs", [{"fault": "stuck"}, {"flags": "maybe"}, {"fault": "none", "p": 0.3}])
def test_spec_rejects_invalid_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        CorruptionSpec(**kwargs)
