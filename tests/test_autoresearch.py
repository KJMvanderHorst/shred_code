"""scripts/autoresearch.py: guard, experiment-file validator, schema, gate, records, and the batch worker pool."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from get_shredded.runner import load_config, run_all
from get_shredded.wave_data import WaveGeneratorConfig, generate_wave_dataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("autoresearch", ROOT / "scripts/autoresearch.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

HEADER = "# @package _global_\n"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    """A fake worktree with the real method/corruption configs and one experiment file."""
    for group in ("method", "corruption"):
        d = tmp_path / "configs" / group
        d.mkdir(parents=True, exist_ok=True)
        for src in (ROOT / "configs" / group).glob("*.yaml"):
            (d / src.name).write_text(src.read_text())
    path = tmp_path / "configs/experiment" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_allowed_paths() -> None:
    assert ar.allowed("configs/experiment/auto_lr_t1.yaml")
    assert ar.allowed("tests/test_auto_wd.py") and ar.allowed("src/get_shredded/model.py")
    assert ar.allowed("auto/journal.md") and not ar.allowed("auto/results.tsv")
    assert not ar.allowed("configs/experiment/dropout_baseline.yaml")
    assert not ar.allowed("configs/experiment/sub/auto_x.yaml")
    assert not ar.allowed("src/get_shredded/runner.py") and not ar.allowed("configs/config.yaml")


def test_changed_paths_and_revert(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
                   check=True)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "new dir").mkdir()
    (tmp_path / "new dir" / "b.txt").write_text("b")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "keep.txt").write_text("k")
    assert sorted(ar.changed_paths(tmp_path)) == ["a.txt", "new dir/b.txt"]
    ar.revert(tmp_path)
    assert ar.changed_paths(tmp_path) == [] and (tmp_path / "a.txt").read_text() == "a"
    assert (tmp_path / "ignored" / "keep.txt").exists()


@pytest.mark.parametrize("body", [
    "defaults:\n  - /method@methods.auto_x_t1: flow_trunk_shred_w256_cos_lr3\nname: auto_x_t1\n"
    "methods:\n  auto_x_t1: {lr: 1.0e-3}\nruns:\n  - {method: auto_x_t1, train: clean}\n",
    "name: auto_x_t1\nmethods:\n  auto_w: {kind: shred, hidden_size: 8}\ncorruptions:\n  auto_m: {fault: missing, p: 0.5}\n"
    "runs:\n  - {method: auto_w, train: auto_m, early_stopping: false}\ntrain: {epochs: 2000, lr: 3.0e-3}\n"
    "sensors: {num: 6, placement: random}\nnoise_frac: 0.1\ndata: {lags: 20}\n"
    "eval:\n  conditions:\n    - {corruption: garbage, flags: silent, ks: all}\n",
    "name: auto_x_t1\nruns:\n  - {method: shred, train: missing}\n",
])
def test_experiment_validator_accepts(tmp_path: Path, body: str) -> None:
    assert ar.validate_experiment_file(tmp_path, _write(tmp_path, "auto_x_t1", HEADER + body)) == []


@pytest.mark.parametrize("body, match", [
    ("name: auto_x_t1\neval: {split: test}\nruns:\n  - {method: shred, train: missing}\n", "eval.split"),
    ("name: auto_x_t1\ndata: {split: [0.5, 0.25, 0.25]}\nruns:\n  - {method: shred, train: missing}\n", "data.split"),
    ("name: auto_x_t1\ndevice: mps\nruns:\n  - {method: shred, train: missing}\n", "device"),
    ("name: auto_x_t1\neval: {seeds: [5]}\nruns:\n  - {method: shred, train: missing}\n", "eval.seeds"),
    ("name: auto_x_t1\noutputs_root: outputs\nruns:\n  - {method: shred, train: missing}\n", "outputs_root"),
    ("defaults:\n  - override /data: wave\nname: auto_x_t1\nruns:\n  - {method: shred, train: missing}\n", "defaults"),
    ("name: auto_x_t1\nmethods:\n  shred: {hidden_size: 8}\nruns:\n  - {method: shred, train: missing}\n", "auto_"),
    ("name: auto_x_t1\nruns:\n  - {method: shred, train: missing}\n  - {method: sdn, train: missing}\n", "one entry"),
    ("name: auto_x_t1\nruns:\n  - {method: shred, train: missing, input: random_input}\n", "input"),
    ("name: other\nruns:\n  - {method: shred, train: missing}\n", "name"),
])
def test_experiment_validator_rejects(tmp_path: Path, body: str, match: str) -> None:
    errs = ar.validate_experiment_file(tmp_path, _write(tmp_path, "auto_x_t1", HEADER + body))
    assert errs and any(match in e for e in errs), errs
    assert ar.validate_experiment_file(tmp_path, _write(tmp_path, "auto_x_t1", body))  # header required


def test_step_schema() -> None:
    good = {"action": "run", "commit_message": "m", "code_changed": False,
            "decisions_on_previous": [{"experiment": "auto_a_t1", "decision": "tie", "reason": "r"}],
            "trials": [{"idea_id": "a", "trial_id": "t2", "experiment": "auto_a_t2",
                        "objective": {"keys": ["clean_err"]}, "seeds": [0, 1, 2], "estimated_minutes": 3}]}
    assert ar.check_schema(ar.STEP_SCHEMA, good) == []
    bad = json.loads(json.dumps(good))
    bad["trials"][0]["seeds"] = [0, 0, 7]
    bad["action"] = "go"
    bad["extra"] = 1
    errs = ar.check_schema(ar.STEP_SCHEMA, bad)
    assert any("unexpected extra" in e for e in errs) and any("'go'" in e for e in errs)
    assert any("duplicate" in e for e in errs) and any("7" in e for e in errs)


def _row(exp, mean, std, per_seed, seeds="0,1,2", pid="p", obj="mean(clean_err)"):
    return {"experiment": exp, "status": "completed", "seeds": seeds, "problem_id": pid, "objective": obj,
            "value_mean": str(mean), "value_std": str(std), "per_seed": ",".join(map(str, per_seed))}


def test_gate() -> None:
    gate = ar.Night.gate
    inc = _row("inc", 1.0, 0.05, [0.95, 1.0, 1.05])
    assert gate(None, _row("c", 0.8, 0.02, [0.78, 0.8, 0.82]), inc) is None
    assert "std" in gate(None, _row("c", 0.97, 0.02, [0.95, 0.97, 0.99]), inc)
    assert "every seed" in gate(None, _row("c", 0.8, 0.0, [0.5, 0.9, 1.0]), _row("i", 1.0, 0.0, [0.4, 1.3, 1.3]))
    assert "3-seed" in gate(None, _row("c", 0.5, 0.0, [0.5], seeds="0"), inc)
    assert "problem_id" in gate(None, _row("c", 0.5, 0.0, [0.5, 0.5, 0.5], pid="q"), inc)
    assert gate(None, _row("c", 0.5, 0.0, [0.5] * 3), None) == "incumbent row not found"


def test_objective_value() -> None:
    metrics = {"clean": {"err": 0.1}, "missing_oracle_k2": {"err": 0.3, "err_u": 0.2, "err_v": None,
                                                          "d_latent_mean": 0.5}, "train": {"err": 0.05}}
    assert ar.Night.objective_value(metrics, {"keys": ["clean_err", "missing_oracle_k2_err"]}) == pytest.approx(0.2)
    assert ar.Night.objective_value(metrics, {"keys": ["clean_err", "train_err"], "agg": "max"}) == 0.1
    assert ar.Night.objective_value(metrics, {"keys": ["missing_oracle_k2_d_latent_mean"]}) == 0.5
    with pytest.raises(ValueError):
        ar.Night.objective_value(metrics, {"keys": ["missing_oracle_k2_err_v"]})


def test_settings_paths_are_absolute(tmp_path: Path) -> None:
    s = ar.settings_json(Path("/w/auto-wt"), Path("/w/main"))["permissions"]
    assert "Edit(//w/auto-wt/configs/experiment/auto_*.yaml)" in s["allow"]
    assert "Read(//w/main/outputs/**)" in s["deny"] and "Bash" in s["deny"]
    assert all(r.startswith(("Edit(//", "Read(//")) for r in s["allow"] + s["deny"] if "(" in r)


def test_batch_pool_matches_sequential(tmp_path: Path) -> None:
    """2 trials x 3 seeds at once (one run.py per trial) == each trial run sequentially; no file collisions."""
    wave = tmp_path / "wave.npz"
    generate_wave_dataset(WaveGeneratorConfig(num_trajectories=1, nx=32, nt=40, num_modes=4), output_path=wave)
    small = [f"data.file={wave}", "data.lags=3", "sensors.num=3", "sensors.rank=6", "eval.draws=2",
             "eval.panel_windows=2"]
    trials = {"auto_p_t1": "{method:shred,train:missing}", "auto_p_t2": "{method:sdn,train:noise}"}
    procs = []
    for name, run in trials.items():
        cmd = ["python", str(ROOT / "scripts/run.py"), f"name={name}", f"runs=[{run}]",
               *ar.forced_overrides([0, 1, 2], dry_run=True), f"outputs_root={tmp_path / 'pool'}", *small]
        procs.append(subprocess.Popen([str(ROOT / ".venv/bin/python"), *cmd[1:]], cwd=ROOT,
                                      env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p in procs:
        out, _ = p.communicate(timeout=600)
        assert p.returncode == 0, out
    for name, run in trials.items():
        cfg = load_config([f"name={name}", f"runs=[{run}]", *ar.forced_overrides([0, 1, 2], dry_run=True),
                           f"outputs_root={tmp_path / 'seq'}", *small, "jobs=1"])
        run_all(cfg)
        pool = json.loads((tmp_path / "pool" / name / "wave" / "summary.json").read_text())
        seq = json.loads((tmp_path / "seq" / name / "wave" / "summary.json").read_text())
        for entry in (pool, seq):
            for r in entry["runs"].values():
                r.pop("wall_time_s")
        assert pool == seq and pool["missing"] == []
        assert all(r["eval_split"] == "valid" for r in pool["runs"].values())


def test_until_and_estimates() -> None:
    night = ar.Night.__new__(ar.Night)
    night.state = {"measured": {"shred": [10.0, 30.0, 20.0]}}
    assert night.s100("shred") == 20.0 and night.s100("flow_trunk_shred") == 12.0
    assert night.s100("robust_v2") == ar.DEFAULT_S100
    assert ar._until("00:00", False) > ar.dt.datetime.now()
    with pytest.raises(SystemExit):
        ar._until(None, False)
