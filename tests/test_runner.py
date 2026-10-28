import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from get_shredded.runner import load_arrays, load_config, load_results, masked_invariance_check, output_root, run_all
from get_shredded.wave_data import WaveGeneratorConfig, generate_wave_dataset

RUNS = [
    ("shred", "noise"), ("shred", "missing"), ("shred_mask", "missing"), ("sdn", "missing"),
    ("senseiver_sdn", "missing"), ("robust_shred_v1", "missing"), ("robust_shred_v2", "garbage"),
    ("gappy_pod", "clean"),
]
CONDITIONS = ("[{corruption:missing,flags:oracle,ps:[0.0,0.3]},{corruption:garbage,flags:silent,ps:[0.3]},"
              "{corruption:missing,flags:oracle,ks:all}]")
SMALL = ["data.lags=3", "train.epochs=2", "sensors.num=3", "sensors.rank=6", "eval.seeds=[0]",
         "eval.draws=2", "eval.panel_windows=2", "verbose=true"]


def _write_random_nektar(path: Path, nx: int = 6, ny: int = 4, nt: int = 40) -> None:
    xs = np.linspace(1.0, 2.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([gx.ravel(order="F"), gy.ravel(order="F")], axis=1)
    rng = np.random.default_rng(0)
    savemat(str(path), {
        "X_star": coords,
        "t": np.arange(nt, dtype=float)[:, None],
        "U_star": rng.normal(size=(coords.shape[0], 2, nt)),
        "p_star": rng.normal(size=(coords.shape[0], nt)),
    })


def _cfg(tmp_path: Path, runs, extra=()):
    runs_arg = "[" + ",".join(f"{{method:{m},train:{t}}}" for m, t in runs) + "]"
    return load_config([f"outputs_root={tmp_path / 'out'}", f"runs={runs_arg}", f"eval.conditions={CONDITIONS}",
                        *SMALL, *extra])


def test_run_all_writes_artifacts_and_skips_completed(tmp_path: Path, capsys) -> None:
    data = tmp_path / "nektar.mat"
    _write_random_nektar(data)
    cfg = _cfg(tmp_path, RUNS, [f"data.file={data}"])
    records = run_all(cfg)
    assert [r["status"] for r in records] == ["completed"] * len(RUNS)
    root = output_root(cfg)
    results = load_results(root)
    names = [f"{m}__{t}" for m, t in RUNS]
    assert set(results) == {(n, 0) for n in names}
    for (method, _), name in zip(RUNS, names):
        metrics = results[(name, 0)]["metrics"]
        for cond in ("missing_oracle_p0", "missing_oracle_p0.3", "garbage_silent_p0.3"):
            assert np.isfinite(metrics[cond]["err"])
        arrays = load_arrays(root, name, 0)
        assert arrays["missing_oracle_p0.3_err"].shape == (2 * 4,)  # draws x test windows
        for k in range(3):  # ks: all with sensors.num=3
            assert (arrays[f"missing_oracle_k{k}_k"] == k).all()
        assert ("missing_oracle_p0.3_d_latent" in arrays) == (method not in ("sdn", "gappy_pod"))
    summary = json.loads((root / "summary.json").read_text())
    assert summary["missing"] == []
    capsys.readouterr()
    again = run_all(cfg)
    assert [r["status"] for r in again] == ["skipped"] * len(RUNS)


def test_masked_invariance_check_on_trained_models(tmp_path: Path) -> None:
    data = tmp_path / "nektar.mat"
    _write_random_nektar(data)
    runs = [("senseiver_sdn", "missing"), ("robust_shred_v1", "missing"), ("robust_shred_v2", "missing")]
    cfg = _cfg(tmp_path, runs, [f"data.file={data}"])
    run_all(cfg)
    for run in cfg.runs:
        assert masked_invariance_check(cfg, run, 0) < 1e-4


def test_runs_on_single_channel_1d_wave(tmp_path: Path) -> None:
    data = tmp_path / "wave.npz"
    generate_wave_dataset(WaveGeneratorConfig(num_trajectories=1, nx=32, nt=40, num_modes=4), output_path=data)
    runs = [("shred", "missing"), ("senseiver_sdn", "missing"), ("robust_shred_v1", "garbage"), ("gappy_pod", "clean")]
    cfg = _cfg(tmp_path, runs, ["data=wave", f"data.file={data}"])
    records = run_all(cfg)
    assert [r["status"] for r in records] == ["completed"] * len(runs)
    assert all(np.isfinite(r["garbage_silent_p0.3"]) for r in records)
