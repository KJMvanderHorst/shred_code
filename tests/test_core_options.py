"""eval.split, train_options, sensor placement, and methods defined inside an experiment file."""
import json
from pathlib import Path

import numpy as np
import pytest

from get_shredded import runner
from get_shredded.runner import (
    _stable_config_fingerprint, load_config, output_root, prepare_seed, run_all, run_config,
)
from get_shredded.wave_data import WaveGeneratorConfig, generate_wave_dataset

SMALL = ["data=wave", "data.lags=3", "train.epochs=2", "sensors.num=3", "sensors.rank=6", "eval.seeds=[0]",
         "eval.draws=2", "eval.panel_windows=2", "verbose=true", "jobs=1", "device=cpu",
         "eval.conditions=[{corruption:missing,flags:oracle,ks:all},{corruption:garbage,flags:silent,ps:[0.3]}]"]
RUNS = "runs=[{method:shred,train:missing},{method:sdn,train:noise},{method:gappy_pod,train:clean}]"


@pytest.fixture
def wave(tmp_path: Path) -> Path:
    path = tmp_path / "wave.npz"
    generate_wave_dataset(WaveGeneratorConfig(num_trajectories=1, nx=32, nt=40, num_modes=4), output_path=path)
    return path


def _cfg(tmp_path: Path, wave: Path, *extra):
    return load_config([f"outputs_root={tmp_path / 'out'}", f"data.file={wave}", RUNS, *SMALL, *extra])


def test_default_fingerprint_unchanged() -> None:
    # Recorded on main before eval.split / placement / train_options existed: cached results stay valid.
    cfg = load_config(["+experiment=_template", "data=wave"])
    assert _stable_config_fingerprint(run_config(cfg, cfg.runs[0], 0)) == "e595afba7e399321"


def test_valid_split_never_evaluates_test_windows(tmp_path: Path, wave: Path, monkeypatch) -> None:
    real = runner.prepare_seed

    def poisoned(cfg, seed, input="probes"):
        sd = real(cfg, seed, input)
        sd.windows[sd.test_idx] = np.nan  # any test window through a model (or gappy POD) gives NaN
        for X in (sd.load_X, sd.transformed_X):
            X[sd.test_idx + sd.lags - 1] = np.nan
        return sd

    monkeypatch.setattr(runner, "prepare_seed", poisoned)
    cfg = _cfg(tmp_path / "valid", wave, "eval.split=valid")
    run_all(cfg)
    results = runner.load_results(output_root(cfg))
    assert len(results) == 3
    for payload in results.values():
        assert payload["eval_split"] == "valid" and payload["config"]["eval"]["split"] == "valid"
        values = [v["err"] for v in payload["metrics"].values() if isinstance(v, dict)]
        assert values and np.all(np.isfinite(values))
    summary = json.loads((output_root(cfg) / "summary.json").read_text())
    assert all(entry["eval_split"] == "valid" for entry in summary["runs"].values())

    # The poisoning itself works: the test split is NaN.
    cfg = _cfg(tmp_path / "test", wave)
    run_all(cfg)
    payload = next(iter(runner.load_results(output_root(cfg)).values()))
    assert "eval_split" not in payload and "split" not in payload["config"]["eval"]
    assert np.isnan(payload["metrics"]["clean"]["err"])


def test_valid_split_has_own_stream(tmp_path: Path, wave: Path) -> None:
    test = prepare_seed(_cfg(tmp_path, wave), 0)
    valid = prepare_seed(_cfg(tmp_path, wave, "eval.split=valid"), 0)
    again = prepare_seed(_cfg(tmp_path, wave, "eval.split=valid"), 0)
    # Existing streams untouched: test draws from [seed, 2], validation noise from [seed, 1].
    rng = np.random.default_rng([0, 2])
    expected = (rng.standard_normal(test.test_noise.shape) * test.sigma).astype(np.float32)
    np.testing.assert_array_equal(test.test_noise, expected)
    np.testing.assert_array_equal(test.valid_noise, valid.valid_noise)
    assert test.valid_rng_state == valid.valid_rng_state
    np.testing.assert_array_equal(valid.eval_idx, valid.valid_idx)
    np.testing.assert_array_equal(valid.truth, valid.load_X[valid.valid_idx + valid.lags - 1])
    np.testing.assert_array_equal(valid.test_noise, again.test_noise)
    for k in valid.test_mask:
        np.testing.assert_array_equal(valid.test_mask[k], again.test_mask[k])
    assert valid.test_noise.shape[1] == len(valid.valid_idx)
    assert not np.array_equal(valid.test_noise[:, :1], test.test_noise[:, :1])
    with pytest.raises(ValueError, match="eval.split"):
        prepare_seed(_cfg(tmp_path, wave, "eval.split=train"), 0)


def test_train_options_passthrough(tmp_path: Path, wave: Path, monkeypatch) -> None:
    seen = {}
    real_fit = runner.fit

    def fit(*args, foo: int = 0, **kwargs):
        seen["foo"] = foo
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(runner, "fit", fit)
    cfg = load_config([f"outputs_root={tmp_path / 'out'}", f"data.file={wave}", "runs=[{method:shred,train:clean}]",
                       *SMALL, "+methods.shred.train_options={foo:3}"])
    run_all(cfg)
    assert seen["foo"] == 3
    assert run_config(cfg, cfg.runs[0], 0)["method"]["train_options"] == {"foo": 3}
    for bad, match in (("{bar:1}", "unknown train_options"), ("{lr:0.1}", "set by the runner")):
        cfg = load_config([f"data.file={wave}", "runs=[{method:shred,train:clean}]", *SMALL,
                           f"+methods.shred.train_options={bad}"])
        with pytest.raises(ValueError, match=match):
            run_all(cfg)


def test_sensor_placement(tmp_path: Path, wave: Path) -> None:
    qr = prepare_seed(_cfg(tmp_path, wave), 0)
    assert run_config(_cfg(tmp_path, wave), _cfg(tmp_path, wave).runs[0], 0)["placement"] == "QR-probe-block"
    r0 = prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=random"), 0)
    r0b = prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=random"), 0)
    r1 = prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=random"), 1)
    np.testing.assert_array_equal(r0.locations, r0b.locations)
    assert not np.array_equal(r0.locations, r1.locations)
    np.testing.assert_array_equal(r0.U_r, qr.U_r)  # same POD basis for gappy POD
    ex = prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=explicit", "+sensors.locations=[3,10,20]"), 0)
    np.testing.assert_array_equal(ex.locations, [3, 10, 20])
    cfg = _cfg(tmp_path, wave, "+sensors.placement=explicit", "+sensors.locations=[3,10,20]")
    assert run_config(cfg, cfg.runs[0], 0)["placement"] == "explicit"
    with pytest.raises(ValueError, match="sensors.locations"):
        prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=explicit", "+sensors.locations=[3,3,20]"), 0)
    with pytest.raises(ValueError, match="sensors.placement"):
        prepare_seed(_cfg(tmp_path, wave, "+sensors.placement=grid"), 0)


def test_method_defined_in_experiment_file(tmp_path: Path) -> None:
    exp = tmp_path / "experiment"
    exp.mkdir()
    (exp / "auto_demo.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /method@methods.auto_lr: flow_trunk_shred_w256_cos_lr3\n"
        "name: auto_demo\n"
        "methods:\n"
        "  auto_lr: {lr: 1.0e-3}\n"
        "  auto_inline: {kind: shred, mask_input: false, recurrent_cell: gru, hidden_size: 32, hidden_layers: 1,"
        " l1: 50, l2: 60, dropout: 0.0}\n"
        "runs:\n"
        "  - {method: auto_lr, train: clean}\n"
        "  - {method: auto_inline, train: clean}\n"
    )
    cfg = load_config([f"hydra.searchpath=[file://{tmp_path}]", "+experiment=auto_demo"])
    assert cfg.methods.auto_lr.kind == "flow_trunk_shred" and cfg.methods.auto_lr.trunk_width == 256
    assert cfg.methods.auto_lr.lr == 1.0e-3 and cfg.methods.auto_lr.lr_schedule == "cosine"
    assert cfg.methods.auto_inline.hidden_size == 32
    assert "shred" in cfg.methods  # base methods still loaded
