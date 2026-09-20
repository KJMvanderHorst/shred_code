import numpy as np

from get_shredded.experiment import _period_split_window_indices


def test_period_split_window_count_matches_sensor_windows():
    n_frames = 150
    lags = 10
    period_length = 30
    train_idx, valid_idx, test_idx = _period_split_window_indices(
        n_frames=n_frames,
        lags=lags,
        period_length=period_length,
        test_pct=10.0,
        val_pct=20.0,
    )

    total_windows = len(train_idx) + len(valid_idx) + len(test_idx)
    assert total_windows == n_frames - lags
    assert test_idx.max() < n_frames - lags
    assert valid_idx.max() < n_frames - lags
