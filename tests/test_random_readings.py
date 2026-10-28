import numpy as np

from get_shredded.random_readings import garbage_windows


def test_garbage_windows_replaces_only_failed_probes() -> None:
    channels, n, lags = 2, 4, 3
    windows = np.random.default_rng(0).random((5, lags, channels * n)).astype(np.float32)
    mask = np.zeros((5, n), dtype=bool)
    mask[0, 1] = mask[3, [0, 2]] = True
    out = garbage_windows(windows, mask, channels, 0.5, 0.5, np.random.default_rng(1))
    again = garbage_windows(windows, mask, channels, 0.5, 0.5, np.random.default_rng(1))

    assert out.shape == windows.shape and out.dtype == windows.dtype
    np.testing.assert_array_equal(out, again)
    changed = (out != windows).all(axis=1)  # (B, C*N): changed at every lag
    expected = np.tile(mask, (1, channels))
    np.testing.assert_array_equal(changed, expected)
    np.testing.assert_array_equal(out[~np.broadcast_to(expected[:, None, :], out.shape)],
                                  windows[~np.broadcast_to(expected[:, None, :], out.shape)])
