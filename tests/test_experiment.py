import numpy as np
from unittest import TestCase

from get_shredded.experiment import split_window_indices


class TestSplitWindowIndices(TestCase):
    def test_is_interleaved_reproducible_and_complete(self) -> None:
        train, valid, test = split_window_indices(
            100,
            train_percentage=80,
            val_percentage=10,
            test_percentage=10,
            seed=42,
        )

        self.assertEqual((len(train), len(valid), len(test)), (80, 10, 10))
        self.assertTrue(np.array_equal(np.sort(np.concatenate([train, valid, test])), np.arange(100)))
        self.assertTrue(set(train).isdisjoint(valid))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(valid).isdisjoint(test))
        self.assertFalse(np.array_equal(test, np.arange(90, 100)))

        repeated = split_window_indices(
            100,
            train_percentage=80,
            val_percentage=10,
            test_percentage=10,
            seed=42,
        )
        self.assertTrue(all(np.array_equal(first, second) for first, second in zip((train, valid, test), repeated)))

    def test_rejects_invalid_percentages(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 100"):
            split_window_indices(
                100,
                train_percentage=70,
                val_percentage=10,
                test_percentage=10,
                seed=42,
            )
