"""Overnight autoresearch driver: stateless Claude planning steps alternating with plain run.py batches.

The agent's rules are in AUTORESEARCH.md; the protocol side in docs/protocol.md (Overnight mode).
  uv run python scripts/autoresearch.py init [--date YYYY-MM-DD] [--ideas PATH] [--base main] [--dry-run]
  cd ../get-shredded-auto
  uv run python scripts/autoresearch.py run --until 07:00 [--max-steps N] [--max-cost-usd 20] [--dry-run]
  uv run python scripts/autoresearch.py report | status

The agent only reads files and edits the allowed ones; this script does all execution and all git operations.
Every trial runs with eval.split=valid, device=cpu and outputs_root=outputs_auto, forced on the command line.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MIN_VERSION = (2, 1, 259)  # --permission-prompts
ALLOWED = ("configs/experiment/auto_*.yaml", "auto/plan.md", "auto/journal.md", "auto/report.md",
           "src/get_shredded/model.py", "tests/test_auto_*.py")
CODE = ("src/get_shredded/model.py", "tests/test_auto_*.py")
# gitignored, so git status can't see agent edits there (results come from here). state.json needs no watch:
# the driver keeps state in memory and overwrites the file on every save.
IGNORED_WATCH = ("outputs_auto",)
TSV_COLUMNS = ("time", "idea_id", "trial_id", "experiment", "run", "seeds", "objective", "value_mean", "value_std",
               "per_seed", "wall_time_s", "epochs_used", "parameters", "device", "status", "git_commit", "problem_id")
DECISION_COLUMNS = ("time", "experiment", "vs_experiment", "agent_decision", "decision", "reason")
METRIC_SUFFIXES = ("d_latent_mean", "d_recon_mean", "err_u", "err_v", "err")
# Seconds per run per 100 epochs under 6-way CPU load (outputs/speed_benchmark/cpu_6x1, r1 jobs 1-6, includes
# one clean eval condition). sdn is unmeasured and borrows shred's; unknown kinds use the slowest measured one.
PRIORS_S100 = {"shred": 8.9, "sdn": 8.9, "flow_trunk_shred": 12.0, "flow_trunk_shred+physics": 27.6, "pod": 0.0}
DEFAULT_S100 = 27.6
STARTUP_S = 20.0      # process start + data load + evaluation slack per job
MAKESPAN_FACTOR = 1.2
REPORT_RESERVE_MIN = 15
# Fixed default runs whose 20-epoch metrics must not change during the night (catches an edit to model.py
# that alters existing behaviour). Passed on the command line so no agent-writable file defines them.
CANARY = {
    "wave": "[{method:shred,train:missing},{method:sdn,train:noise},{method:shred_mask,train:missing},"
            "{method:robust_shred_v1,train:missing},{method:robust_shred_v2,train:garbage},"
            "{method:trunk_shred_pinn,train:clean},{method:gappy_pod,train:clean}]",
    "nektar_uv": "[{method:shred,train:missing},{method:flow_trunk_shred_w256,train:clean},"
                 "{method:flow_trunk_shred_div,train:clean}]",
}
DRY_IDEAS = """# Ideas (dry run)

## 1. Wider SHRED
What to try: SHRED with hidden_size 128 instead of 64.
Why: wiring test of the overnight loop.
Objective: missing_oracle_k2_err (mean).
Priority: high

## 2. Lower learning rate for SHRED
What to try: train.lr 5e-4.
Why: wiring test.
Objective: clean_err.
Priority: medium
"""

# ---------------------------------------------------------------- schemas and prompts

STEP_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "commit_message", "code_changed", "decisions_on_previous", "trials"],
    "properties": {
        "action": {"enum": ["run", "done", "skip_idea"]},
        "commit_message": {"type": "string", "maxLength": 200},
        "code_changed": {"type": "boolean"},
        "decisions_on_previous": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["experiment", "decision", "reason"],
            "properties": {"experiment": {"type": "string"}, "decision": {"enum": ["keep", "discard", "tie"]},
                           "vs_experiment": {"type": "string"}, "reason": {"type": "string", "maxLength": 300}}}},
        "trials": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["idea_id", "trial_id", "experiment", "objective", "seeds", "estimated_minutes"],
            "properties": {
                "idea_id": {"type": "string", "pattern": "^[a-z0-9]+$"},
                "trial_id": {"type": "string", "pattern": "^t[0-9]+$"},
                "experiment": {"type": "string", "pattern": "^auto_[a-z0-9]+_t[0-9]+$"},
                "objective": {"type": "object", "additionalProperties": False, "required": ["keys"],
                              "properties": {"keys": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                                             "agg": {"enum": ["mean", "max"]}}},
                "seeds": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": [0, 1, 2]}},
                "estimated_minutes": {"type": "number"}}}},
    },
}
INTAKE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["ideas"],
    "properties": {"ideas": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["idea_id", "title", "status"],
        "properties": {"idea_id": {"type": "string", "pattern": "^[a-z0-9]+$"}, "title": {"type": "string"},
                       "status": {"enum": ["planned", "daytime_only"]}}}}},
}
REPORT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["written"],
                 "properties": {"written": {"type": "boolean"}}}

RULES = "Before anything else, Read AUTORESEARCH.md (your rules; they override this prompt where they differ) and CLAUDE.md."
INTAKE_PROMPT = """You are the overnight tuning agent for get-shredded. This is the intake call of the night.
{rules}
Read auto/ideas.md and follow AUTORESEARCH.md "Intake": write auto/plan.md. The job slots per batch are {workers};
the night runs until {until}. Edit only auto/plan.md. Reply with the JSON object only."""
STEP_PROMPT = """You are the overnight tuning agent for get-shredded. Step {step}; now {now}; the driver stops planning at
{plan_deadline} ({minutes_left} min left, report time already reserved).
{rules}

State from the driver (facts; trust these over your memory):
- Job slots for this batch: {workers}. One job = one trial x one seed. Fill them: e.g. {workers} seed-0
  screens, or 2 configs x 3 seeds for confirmations. Unused slots are idle cores.
- Trials finished since your last step (rows in auto/results.tsv): {finished}
- Driver notes since your last step: {notes}
- Driver time estimate per job: {startup:.0f} s + train.epochs/100 x s_per_100_epochs, under {workers}-way load:
  {estimates}. A batch takes as long as its longest job (x{factor}) and must fit in {minutes_left} min.

Do, in order:
1. Read auto/plan.md, the last 5 entries of auto/journal.md, auto/results.tsv, auto/decisions.tsv.
2. For each finished trial: append a journal entry (AUTORESEARCH.md format) and add it to decisions_on_previous
   (field `experiment`; for keep/tie also `vs_experiment`, the incumbent you compared with).
3. Then exactly one of:
   - write the next batch as configs/experiment/auto_<idea>_<trial>.yaml files (one run each) -> action "run";
   - close the current idea -> action "skip_idea" (with no trials);
   - nothing left in the plan that fits the time -> action "done".
   Update auto/plan.md status lines as you go.
You cannot run commands; the driver trains, evaluates, commits and checks everything.
Reply with the JSON object only."""
REPORT_PROMPT = """You are the overnight tuning agent for get-shredded. This is the last call of the night: the report.
{rules}
Read auto/plan.md, auto/journal.md, auto/results.tsv and auto/decisions.tsv, then write auto/report.md following
AUTORESEARCH.md "Report". Edit only auto/report.md (and append a closing note to auto/journal.md if useful).
Driver facts for the report (copy them, do not recompute):
- Steps: {steps}; trials recorded: {trials}; stop reason: {reason}
- Cost of Claude calls: ${cost:.2f}
- Guard violations: {violations}
- Failed code checks / failed trials / refused batches: {failures}
- Gate overrides (a `keep` downgraded by the driver): {overrides}
- Throughput per hour (runs, actual/estimated wall time): {throughput}
Reply with the JSON object only."""


class StepError(Exception):
    """A step that must be reverted (guard violation, malformed output, failed check)."""


# ---------------------------------------------------------------- small helpers

def sh(cmd: list[str], cwd: Path, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True, **kw)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def check_schema(schema: dict, value: Any, path: str = "$") -> list[str]:
    """Errors for the subset of JSON Schema used above (no jsonschema dependency)."""
    errs: list[str] = []
    types = {"object": dict, "array": list, "string": str, "boolean": bool}
    t = schema.get("type")
    if t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path}: expected number"]
    elif t and not isinstance(value, types[t]):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if t == "string":
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if t == "object":
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in value]
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errs += check_schema(sub, value[k], f"{path}.{k}")
    if t == "array":
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(v) for v in value}) != len(value):
            errs.append(f"{path}: duplicate items")
        for i, item in enumerate(value):
            errs += check_schema(schema.get("items", {}), item, f"{path}[{i}]")
    return errs


def allowed(path: str, patterns=ALLOWED) -> bool:
    return any(fnmatch.fnmatchcase(path, p) and path.count("/") == p.count("/") for p in patterns)


def changed_paths(wt: Path) -> list[str]:
    out = sh(["git", "status", "--porcelain", "-z", "--untracked-files=all"], wt).stdout.split("\0")
    paths, skip = [], False
    for entry in out:
        if skip or not entry:
            skip = False
            continue
        paths.append(entry[3:])
        skip = entry[0] in "RC"  # rename/copy: the next field is the old path
    return paths


def snapshot(wt: Path) -> dict[str, tuple[float, int]]:
    """(mtime, size) of watched gitignored files; the agent may not touch them."""
    out = {}
    for name in IGNORED_WATCH:
        base = wt / name
        files = [base] if base.is_file() else (p for p in base.rglob("*") if p.is_file()) if base.exists() else []
        for p in files:
            st = p.stat()
            out[str(p.relative_to(wt))] = (st.st_mtime, st.st_size)
    return out


def revert(wt: Path) -> None:
    """Drop every uncommitted change to tracked and untracked (not ignored) files."""
    sh(["git", "checkout", "--", "."], wt)
    sh(["git", "clean", "-fdq"], wt)


def commit(wt: Path, message: str) -> None:
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], wt).stdout.strip()
    if not branch.startswith("auto/"):
        raise SystemExit(f"refusing to commit on branch {branch!r}: overnight commits go to auto/* only")
    sh(["git", "add", "-A"], wt)
    if sh(["git", "diff", "--cached", "--quiet"], wt, check=False).returncode == 0:
        return
    message = "auto: " + " ".join(message.split())[:200]
    sh(["git", "commit", "-q", "-m", message], wt)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_tsv(path: Path, columns, rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, columns, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- experiment files

TOP_KEYS = {"name", "defaults", "methods", "corruptions", "runs", "train", "eval", "sensors", "noise_frac", "data"}
SUB_KEYS = {"train": {"epochs", "batch_size", "lr", "patience"}, "eval": {"conditions"},
            "sensors": {"num", "rank", "placement", "locations"}, "data": {"lags"}}
DEFAULT_RE = re.compile(r"^/(method|corruption)@(methods|corruptions)\.(auto_[a-z0-9_]+)$")


def validate_experiment_file(wt: Path, path: Path) -> list[str]:
    """Static checks on an agent-written experiment file (before composing it)."""
    from omegaconf import OmegaConf

    text = path.read_text()
    if not text.startswith("# @package _global_"):
        return [f"{path.name}: first line must be '# @package _global_'"]
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path))
    except Exception as exc:
        return [f"{path.name}: not valid YAML ({exc})"]
    if not isinstance(raw, dict):
        return [f"{path.name}: not a mapping"]
    errs = [f"{path.name}: key {k!r} not allowed" for k in raw if k not in TOP_KEYS]
    for k, subs in SUB_KEYS.items():
        if k in raw:
            if not isinstance(raw[k], dict):
                errs.append(f"{path.name}: {k} must be a mapping")
            else:
                errs += [f"{path.name}: {k}.{s} not allowed" for s in raw[k] if s not in subs]
    if raw.get("name") != path.stem:
        errs.append(f"{path.name}: name must be {path.stem!r}")
    base_methods = {p.stem for p in (wt / "configs/method").glob("*.yaml")}
    base_corruptions = {p.stem for p in (wt / "configs/corruption").glob("*.yaml")}
    new = {"methods": set(), "corruptions": set()}
    for entry in raw.get("defaults") or []:
        if entry == "_self_":
            continue
        if not isinstance(entry, dict) or len(entry) != 1:
            errs.append(f"{path.name}: defaults entry {entry!r} not allowed")
            continue
        (key, value), = entry.items()
        m = DEFAULT_RE.match(str(key))
        if not m or (m.group(1) + "s") != m.group(2):
            errs.append(f"{path.name}: defaults entry {key!r} not allowed (only /method@methods.auto_* "
                        "or /corruption@corruptions.auto_*)")
            continue
        if str(value) not in (base_methods if m.group(1) == "method" else base_corruptions):
            errs.append(f"{path.name}: unknown {m.group(1)} config {value!r}")
        new[m.group(2)].add(m.group(3))
    for group, base in (("methods", base_methods), ("corruptions", base_corruptions)):
        for k in (raw.get(group) or {}):
            if not str(k).startswith("auto_") or k in base:
                errs.append(f"{path.name}: {group}.{k}: only new auto_* entries may be defined")
            new[group].add(k)
    runs = raw.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        errs.append(f"{path.name}: runs must have exactly one entry")
    else:
        run = runs[0]
        errs += [f"{path.name}: runs[0].{k} not allowed" for k in run if k not in {"method", "train", "early_stopping"}]
        if run.get("method") not in base_methods | new["methods"]:
            errs.append(f"{path.name}: unknown method {run.get('method')!r}")
        if run.get("train") not in base_corruptions | new["corruptions"]:
            errs.append(f"{path.name}: unknown train corruption {run.get('train')!r}")
    return errs


def forced_overrides(seeds: list[int], dry_run: bool) -> list[str]:
    forced = ["eval.split=valid", "outputs_root=outputs_auto", "device=cpu", "threads=1", f"jobs={len(seeds)}",
              f"eval.seeds=[{','.join(str(s) for s in seeds)}]"]
    return forced + (["data=wave", "train.epochs=20"] if dry_run else [])


def compose_trial(experiment: str, seeds: list[int], dry_run: bool):
    from get_shredded.runner import load_config

    return load_config([f"+experiment={experiment}", *forced_overrides(seeds, dry_run)])


def problem_id(cfg) -> str:
    """Hash of everything that defines the reconstruction problem; trials compare only within one problem."""
    from omegaconf import OmegaConf
    from get_shredded.runner import eval_conditions

    spec = {"data": OmegaConf.to_container(cfg.data), "sensors": OmegaConf.to_container(cfg.sensors),
            "noise_frac": float(cfg.noise_frac), "draws": int(cfg.eval.draws),
            "conditions": {k: asdict(s) for k, (s, _) in eval_conditions(cfg).items()}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:10]


def objective_keys(cfg) -> set[str]:
    from get_shredded.runner import eval_conditions

    return {"clean_err", "train_err"} | {f"{tag}_{m}" for tag in eval_conditions(cfg) for m in METRIC_SUFFIXES}


def time_key(cfg, run) -> str:
    method = cfg.methods[run.method]
    physics = float(method.get("physics_weight", 0.0)) > 0 or float(method.get("probe_weight", 0.0)) > 0
    return str(method.kind) + ("+physics" if physics else "")


# ---------------------------------------------------------------- night state

class Night:
    def __init__(self, wt: Path, args) -> None:
        self.wt, self.args = wt, args
        self.auto = wt / "auto"
        self.logs = self.auto / "logs"
        self.state_path = self.auto / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "step": 0, "cost_usd": 0.0, "violation_streak": 0, "refusal_streak": 0, "notes": [],
            "unreviewed": [], "pending": None, "measured": {}, "hours": {}, "violations": [], "failures": [],
            "overrides": [], "stop_reason": None,
        }
        self.procs: list[subprocess.Popen] = []
        self.stop = False

    def save(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.state_path)

    @property
    def python(self) -> str:
        return str(self.wt / ".venv/bin/python")

    def note(self, text: str, kind: str | None = None) -> None:
        self.state["notes"].append(text)
        if kind:
            self.state[kind].append(f"{now()} {text}")
        self.save()

    # ------------------------------------------------------------ claude

    def claude(self, label: str, prompt: str, schema: dict) -> dict:
        cmd = [self.args.claude, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(schema),
               "--settings", str(self.auto / "claude-settings.json"), "--permission-mode", "dontAsk",
               "--permission-prompts", "none", "--restricted", "--tools", "Read,Glob,Grep,Edit,Write",
               "--strict-mcp-config", "--no-session-persistence", "--max-budget-usd", str(self.args.max_call_usd)]
        if self.args.model:
            cmd += ["--model", self.args.model]
        for attempt in (1, 2):
            try:
                proc = sh(cmd, self.wt, check=False, timeout=1200, start_new_session=True)  # Ctrl-C: driver only
            except subprocess.TimeoutExpired:
                proc = None
            raw = proc.stdout if proc else ""
            (self.logs / f"{label}.claude.json").write_text(raw or "(timeout)")
            if proc and proc.stderr:
                (self.logs / f"{label}.claude.stderr").write_text(proc.stderr)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if data:
                self.state["cost_usd"] += float(data.get("total_cost_usd") or 0.0)
                self.save()
                out = data.get("structured_output")
                if out is None and isinstance(data.get("result"), str):
                    try:
                        out = json.loads(data["result"])
                    except json.JSONDecodeError:
                        out = None
                if not data.get("is_error") and isinstance(out, dict):
                    return out
            if attempt == 1:
                time.sleep(30)
        raise RuntimeError(f"claude call {label} failed twice (see auto/logs/{label}.claude.json)")

    # ------------------------------------------------------------ guard and checks

    def guard(self, before: dict, patterns=ALLOWED) -> list[str]:
        errs = [f"edited {p}" for p in changed_paths(self.wt) if not allowed(p, patterns)]
        after = snapshot(self.wt)
        errs += [f"touched ignored file {p}" for p in set(before) | set(after) if before.get(p) != after.get(p)]
        return errs

    def canary(self, label: str) -> dict:
        """{data/run: metrics} of the fixed default runs (20 epochs, seed 0, eval.split=valid)."""
        out_root = self.wt / "outputs_auto" / "_canary" / label
        if out_root.exists():
            shutil.rmtree(out_root)
        metrics = {}
        for data, runs in CANARY.items():
            cmd = [self.python, "scripts/run.py", "+experiment=_template", "name=_canary", f"data={data}",
                   f"runs={runs}", "train.epochs=20", "eval.seeds=[0]", "eval.split=valid", "device=cpu",
                   f"jobs={self.args.workers}", "threads=1", f"outputs_root={out_root}"]
            proc = sh(cmd, self.wt, check=False, env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=1800)
            (self.logs / f"canary_{label}_{data}.log").write_text(proc.stdout + proc.stderr)
            for path in sorted((out_root / "_canary" / data).glob("seed0/*/result.json")):
                payload = json.loads(path.read_text())
                metrics[f"{data}/{payload['run']}"] = payload.get("metrics", payload.get("error_summary"))
        return metrics

    def code_checks(self, label: str) -> str | None:
        """None if pytest and the canary pass, else a two-line summary (full output in auto/logs)."""
        proc = sh([self.python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], self.wt, check=False,
                  env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=1800)
        (self.logs / f"{label}_pytest.txt").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()][-2:]
            return "pytest failed: " + " | ".join(tail)
        expected = json.loads((self.auto / "canary.json").read_text())
        got = self.canary(label)
        diff = [k for k in expected if got.get(k) != expected[k]] + [k for k in got if k not in expected]
        if diff:
            return f"canary changed (existing default behaviour altered): {', '.join(diff[:4])}"
        return None

    # ------------------------------------------------------------ time budget

    def s100(self, key: str) -> float:
        measured = self.state["measured"].get(key, [])[-10:]
        if measured:
            return sorted(measured)[len(measured) // 2]
        return PRIORS_S100.get(key, DEFAULT_S100)

    def estimates_text(self) -> str:
        keys = sorted(set(PRIORS_S100) | set(self.state["measured"]))
        parts = [f"{k} {self.s100(k):.1f}" + (" (measured)" if self.state["measured"].get(k) else " (prior)")
                 for k in keys]
        return ", ".join(parts) + f"; any other kind {DEFAULT_S100} (prior)"

    def plan_deadline(self) -> dt.datetime:
        return self.args.until_dt - dt.timedelta(minutes=REPORT_RESERVE_MIN)

    def minutes_left(self) -> float:
        return (self.plan_deadline() - dt.datetime.now()).total_seconds() / 60

    # ------------------------------------------------------------ steps

    def intake(self) -> None:
        before = snapshot(self.wt)
        prompt = INTAKE_PROMPT.format(rules=RULES, workers=self.args.workers, until=self.args.until_dt.strftime("%H:%M"))
        out = self.claude("intake", prompt, INTAKE_SCHEMA)
        errs = check_schema(INTAKE_SCHEMA, out) + self.guard(before, ("auto/plan.md",))
        if not (self.auto / "plan.md").exists():
            errs.append("auto/plan.md was not written")
        if errs:
            revert(self.wt)
            raise SystemExit("intake failed: " + "; ".join(errs))
        commit(self.wt, f"intake: {len(out['ideas'])} ideas")

    def validate_step(self, out: dict) -> list[dict]:
        """Checks beyond the schema; returns per-trial info for execution."""
        errs = check_schema(STEP_SCHEMA, out)
        if errs:
            raise StepError("malformed step output: " + "; ".join(errs[:5]))
        trials = out["trials"]
        if (out["action"] == "run") != bool(trials):
            raise StepError("action 'run' needs trials, and only 'run' may have trials")
        if sum(len(t["seeds"]) for t in trials) > self.args.workers:
            raise StepError(f"batch has more than {self.args.workers} (trial, seed) jobs")
        done = {r["experiment"] for r in read_tsv(self.auto / "results.tsv")}
        for d in out["decisions_on_previous"]:
            if d["decision"] == "keep" and not d.get("vs_experiment"):
                raise StepError(f"decision keep on {d['experiment']} needs vs_experiment")
        infos, names = [], set()
        for t in trials:
            exp = t["experiment"]
            if exp != f"auto_{t['idea_id']}_{t['trial_id']}" or exp in done or exp in names:
                raise StepError(f"{exp}: must be auto_<idea_id>_<trial_id> and new")
            names.add(exp)
            path = self.wt / "configs/experiment" / f"{exp}.yaml"
            if not path.exists():
                raise StepError(f"{exp}: configs/experiment/{exp}.yaml not written")
            errs = validate_experiment_file(self.wt, path)
            if errs:
                raise StepError("; ".join(errs[:5]))
            seeds = sorted(t["seeds"])
            try:
                cfg = compose_trial(exp, seeds, self.args.dry_run)
            except Exception as exc:
                raise StepError(f"{exp}: does not compose ({type(exc).__name__}: {str(exc)[:200]})")
            bad = [k for k in t["objective"]["keys"] if k not in objective_keys(cfg)]
            if bad:
                raise StepError(f"{exp}: objective keys {bad} are not summary keys of its conditions")
            run = cfg.runs[0]
            key = time_key(cfg, run)
            infos.append({**t, "seeds": seeds, "problem_id": problem_id(cfg), "time_key": key,
                          "epochs": int(cfg.train.epochs),
                          "est_s": STARTUP_S + int(cfg.train.epochs) / 100 * self.s100(key)})
        return infos

    def gate(self, cand: dict, inc: dict | None) -> str | None:
        """None if `cand` passes the gate against `inc`, else why not."""
        if inc is None:
            return "incumbent row not found"
        for row in (cand, inc):
            if row["status"] != "completed" or row["seeds"] != "0,1,2":
                return f"{row['experiment']} is not a completed 3-seed row"
        if cand["problem_id"] != inc["problem_id"] or cand["objective"] != inc["objective"]:
            return "different problem_id or objective"
        mc, mi = float(cand["value_mean"]), float(inc["value_mean"])
        if not mc < mi - max(float(cand["value_std"]), float(inc["value_std"])):
            return "mean difference within the larger seed std"
        pc, pi = ([float(v) for v in r["per_seed"].split(",")] for r in (cand, inc))
        if not all(c < i for c, i in zip(pc, pi)):
            return "not lower on every seed"
        return None

    def record_decisions(self, out: dict) -> None:
        rows = {r["experiment"]: r for r in read_tsv(self.auto / "results.tsv")}
        out_rows = []
        for d in out["decisions_on_previous"]:
            final = d["decision"]
            if final == "keep":
                why = self.gate(rows.get(d["experiment"], {"status": "missing", "experiment": d["experiment"],
                                                           "seeds": ""}), rows.get(d.get("vs_experiment", "")))
                if why:
                    final = "tie"
                    self.note(f"gate override: {d['experiment']} keep -> tie ({why})", "overrides")
            out_rows.append({"time": now(), "experiment": d["experiment"], "vs_experiment": d.get("vs_experiment", ""),
                             "agent_decision": d["decision"], "decision": final, "reason": d["reason"]})
            if d["experiment"] in self.state["unreviewed"]:
                self.state["unreviewed"].remove(d["experiment"])
        append_tsv(self.auto / "decisions.tsv", DECISION_COLUMNS, out_rows)
        self.save()

    def execute(self, trials: list[dict]) -> None:
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        self.procs = []
        for t in trials:
            log = (self.logs / f"step{self.state['step']}_{t['experiment']}.log").open("a")
            cmd = [self.python, "scripts/run.py", f"+experiment={t['experiment']}",
                   *forced_overrides(t["seeds"], self.args.dry_run)]
            # own session: a Ctrl-C in the terminal reaches the driver only, which lets the batch finish
            self.procs.append(subprocess.Popen(cmd, cwd=self.wt, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        for p in self.procs:
            p.wait()
        self.procs = []

    def record(self, trials: list[dict]) -> None:
        data_name = "wave" if self.args.dry_run else None
        rows = []
        for t in trials:
            exp_root = self.wt / "outputs_auto" / t["experiment"]
            roots = [exp_root / data_name] if data_name else sorted(p for p in exp_root.glob("*") if p.is_dir())
            results = {}
            for root in roots:
                for s in t["seeds"]:
                    for path in root.glob(f"seed{s}/*/result.json"):
                        payload = json.loads(path.read_text())
                        if payload.get("status") == "completed" and payload.get("eval_split") == "valid":
                            results[s] = payload
            row = {"time": now(), "idea_id": t["idea_id"], "trial_id": t["trial_id"], "experiment": t["experiment"],
                   "seeds": ",".join(map(str, t["seeds"])), "problem_id": t["problem_id"],
                   "objective": f"{t['objective'].get('agg', 'mean')}({','.join(t['objective']['keys'])})"}
            try:
                if len(results) != len(t["seeds"]):
                    raise ValueError(f"{len(t['seeds']) - len(results)} seed(s) failed or missing")
                per_seed = [self.objective_value(results[s]["metrics"], t["objective"]) for s in t["seeds"]]
                first = results[t["seeds"][0]]
                mean = sum(per_seed) / len(per_seed)
                std = (sum((v - mean) ** 2 for v in per_seed) / len(per_seed)) ** 0.5  # population std, as summary.json
                walls = [results[s]["wall_time_s"] for s in t["seeds"]]
                checks = [int(results[s]["metrics"]["epochs_used"]) for s in t["seeds"]]
                row.update(run=first["run"], value_mean=f"{mean:.6g}", value_std=f"{std:.6g}",
                           per_seed=",".join(f"{v:.6g}" for v in per_seed),
                           wall_time_s=f"{sum(walls) / len(walls):.1f}", epochs_used=",".join(map(str, checks)),
                           parameters=first["metrics"].get("parameters", ""), device=first["device"],
                           status="completed", git_commit=first.get("git_commit") or "")
                full = 1 + t["epochs"] // 20  # validation checks when training is not stopped early
                for wall, c in zip(walls, checks):
                    epochs = t["epochs"] if c >= full or c <= 1 else 20 * (c - 1)
                    if epochs > 0:
                        self.state["measured"].setdefault(t["time_key"], []).append(round(wall / (epochs / 100), 2))
                hour = dt.datetime.now().strftime("%Y-%m-%dT%H")
                h = self.state["hours"].setdefault(hour, {"runs": 0, "ratio_sum": 0.0})
                h["runs"] += len(walls)
                prior = STARTUP_S + t["epochs"] / 100 * PRIORS_S100.get(t["time_key"], DEFAULT_S100)
                h["ratio_sum"] += sum(w / max(prior, 1.0) for w in walls)
            except (KeyError, ValueError, TypeError) as exc:
                row.update(status="failed", run="", device="cpu")
                self.note(f"{t['experiment']} failed: {exc} (log: auto/logs/step{self.state['step']}_"
                          f"{t['experiment']}.log)", "failures")
            rows.append(row)
            self.state["unreviewed"].append(t["experiment"])
        append_tsv(self.auto / "results.tsv", TSV_COLUMNS, rows)
        self.state["pending"] = None
        self.save()

    @staticmethod
    def objective_value(metrics: dict, objective: dict) -> float:
        vals = []
        for key in objective["keys"]:
            suffix = next(s for s in METRIC_SUFFIXES if key.endswith("_" + s))
            value = metrics[key[: -len(suffix) - 1]][suffix]
            if value is None:
                raise ValueError(f"{key} is not available for this run")
            vals.append(float(value))
        return max(vals) if objective.get("agg") == "max" else sum(vals) / len(vals)

    def run_batch(self, trials: list[dict], message: str) -> None:
        self.state["pending"] = {"trials": trials, "message": message}
        self.save()
        self.execute(trials)
        self.record(trials)
        commit(self.wt, f"record {', '.join(t['experiment'] for t in trials)}")

    def inject(self, n: int) -> None:
        """Dry run only: deliberate faults the guard / code checks must catch."""
        if n == 2:
            with (self.wt / "configs/config.yaml").open("a") as f:
                f.write("# injected by --dry-run: the guard must revert this\n")
        if n == 3:
            (self.wt / "tests/test_auto_injected.py").write_text(
                "def test_injected_failure():\n    assert False, 'injected by --dry-run'\n")

    def step(self, n_this_run: int) -> str | None:
        """One planning step plus its batch; returns a stop reason or None."""
        self.state["step"] += 1
        n = self.state["step"]
        label = f"step{n}"
        notes, self.state["notes"] = self.state["notes"], []
        self.save()
        prompt = STEP_PROMPT.format(
            step=n, now=dt.datetime.now().strftime("%H:%M"), plan_deadline=self.plan_deadline().strftime("%H:%M"),
            minutes_left=int(self.minutes_left()), rules=RULES, workers=self.args.workers,
            finished=", ".join(self.state["unreviewed"]) or "none", notes=" | ".join(notes) or "none",
            startup=STARTUP_S, estimates=self.estimates_text(), factor=MAKESPAN_FACTOR)
        before = snapshot(self.wt)
        try:
            out = self.claude(label, prompt, STEP_SCHEMA)
        except RuntimeError as exc:
            revert(self.wt)
            self.note(str(exc), "failures")
            return "claude call failed"
        (self.logs / f"{label}.step.json").write_text(json.dumps(out, indent=1))
        if self.stop:
            revert(self.wt)
            return "interrupted"
        if self.args.dry_run:
            self.inject(n_this_run)
        try:
            violations = self.guard(before)
            if violations:
                self.state["violation_streak"] += 1
                raise StepError("guard: " + "; ".join(violations[:5]))
            trials = self.validate_step(out)
        except StepError as exc:
            revert(self.wt)
            kind = "violations" if str(exc).startswith("guard") else "failures"
            if kind == "failures":
                self.state["violation_streak"] += 1  # malformed output counts towards the stop rule too
            self.note(f"step {n} reverted: {exc}", kind)
            return "3 violations in a row" if self.state["violation_streak"] >= 3 else None
        self.state["violation_streak"] = 0
        if trials:
            makespan_min = max(t["est_s"] for t in trials) * MAKESPAN_FACTOR / 60
            if makespan_min > self.minutes_left():
                revert(self.wt)
                self.state["refusal_streak"] += 1
                self.note(f"step {n} refused: batch needs ~{makespan_min:.0f} min, {self.minutes_left():.0f} left",
                          "failures")
                return "no batch fits before the deadline" if self.state["refusal_streak"] >= 2 else None
        self.state["refusal_streak"] = 0
        if any(allowed(p, CODE) for p in changed_paths(self.wt)):
            failure = self.code_checks(label)
            if failure:
                revert(self.wt)
                self.note(f"step {n} reverted, code check failed: {failure} (auto/logs/{label}_pytest.txt)",
                          "failures")
                return None
        self.record_decisions(out)
        commit(self.wt, f"step {n}: {out['commit_message']}")
        if out["action"] == "done":
            return "agent: plan done"
        if trials:
            self.run_batch(trials, out["commit_message"])
        return None

    def report(self, reason: str) -> None:
        self.state["stop_reason"] = reason
        self.save()
        hours = ", ".join(f"{h[11:]}h: {v['runs']} runs, x{v['ratio_sum'] / max(v['runs'], 1):.2f}"
                          for h, v in sorted(self.state["hours"].items())) or "none"
        prompt = REPORT_PROMPT.format(
            rules=RULES, steps=self.state["step"], trials=len(read_tsv(self.auto / "results.tsv")), reason=reason,
            cost=self.state["cost_usd"], violations=" | ".join(self.state["violations"]) or "none",
            failures=" | ".join(self.state["failures"]) or "none",
            overrides=" | ".join(self.state["overrides"]) or "none", throughput=hours)
        before = snapshot(self.wt)
        try:
            self.claude("report", prompt, REPORT_SCHEMA)
        except RuntimeError as exc:
            print(f"report call failed: {exc}", file=sys.stderr)
            return
        errs = self.guard(before, ("auto/report.md", "auto/journal.md"))
        if errs:
            revert(self.wt)
            print("report reverted: " + "; ".join(errs), file=sys.stderr)
            return
        commit(self.wt, f"report ({reason})")


# ---------------------------------------------------------------- commands

def claude_version(claude: str) -> tuple[int, ...]:
    out = sh([claude, "--version"], ROOT).stdout
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        raise SystemExit(f"cannot parse `claude --version`: {out!r}")
    return tuple(int(x) for x in m.groups())


def settings_json(wt: Path, main: Path) -> dict:
    W, M = f"/{wt}", f"/{main}"  # `//abs` anchors at the filesystem root; `/x` would be relative to auto/
    locked_src = ("runner", "data", "corruption", "wave_data", "robust_shred", "senseiver", "plots", "__init__")
    deny = ["Bash", "WebFetch", "WebSearch", "NotebookEdit", "Agent",
            f"Read({M}/outputs/**)", f"Read({W}/outputs/**)", f"Read({W}/data/**)", f"Read({W}/.venv/**)"]
    deny += [f"Edit({W}/src/get_shredded/{name}.py)" for name in locked_src]
    deny += [f"Edit({W}/{p})" for p in (
        "configs/config.yaml", "configs/data/**", "configs/corruption/**", "configs/method/**", "docs/**",
        "CLAUDE.md", "AUTORESEARCH.md", "scripts/**", "auto/ideas.md", "auto/results.tsv", "auto/decisions.tsv",
        "auto/state.json", "auto/canary.json", "auto/claude-settings.json", "auto/logs/**", ".git/**", ".gitignore",
        "pyproject.toml", "uv.lock", ".claude/**", "outputs_auto/**")]
    return {"permissions": {"defaultMode": "dontAsk",
                            "allow": ["Read", "Glob", "Grep"] + [f"Edit({W}/{p})" for p in ALLOWED],
                            "deny": deny}}


def cmd_init(args) -> None:
    main = ROOT
    date = args.date or dt.date.today().isoformat()
    branch = f"auto/dry-{date}" if args.dry_run else f"auto/{date}"
    wt = Path(args.worktree or (main.parent / ("get-shredded-auto-dry" if args.dry_run else "get-shredded-auto")))
    wt = wt.resolve()
    has_branch = sh(["git", "rev-parse", "--verify", "--quiet", branch], main, check=False).returncode == 0
    if (wt.exists() or has_branch) and not args.resume:
        raise SystemExit(f"{wt} or branch {branch} already exists (use --resume to continue it)")
    if not wt.exists():
        sh(["git", "worktree", "add", "-q", *([] if has_branch else ["-b"]), *([str(wt), branch] if has_branch
                                                                              else [branch, str(wt), args.base])], main)
    if not (wt / "data").exists():
        (wt / "data").symlink_to(main / "data")
    print("uv sync ...", flush=True)
    sh(["uv", "sync", "-q"], wt)
    auto = wt / "auto"
    (auto / "logs").mkdir(parents=True, exist_ok=True)
    ideas = auto / "ideas.md"
    if not ideas.exists():
        if args.dry_run:
            ideas.write_text(DRY_IDEAS)
        else:
            shutil.copy(Path(args.ideas) if args.ideas else main / "docs/ideas_template.md", ideas)
    (auto / "claude-settings.json").write_text(json.dumps(settings_json(wt, main), indent=1) + "\n")
    if not (auto / "journal.md").exists():
        (auto / "journal.md").write_text("# Journal\n")
    if not (auto / "canary.json").exists():
        print("canary (fixed default runs, 20 epochs) ...", flush=True)
        night = Night(wt, argparse.Namespace(workers=args.workers, dry_run=args.dry_run))
        (auto / "canary.json").write_text(json.dumps(night.canary("init"), indent=1, sort_keys=True))
    commit(wt, f"init {branch}")
    print(f"worktree {wt} on {branch}. Edit auto/ideas.md, then:\n"
          f"  cd {wt} && uv run python scripts/autoresearch.py run --until 07:00" + (" --dry-run" if args.dry_run else ""))


def _until(text: str | None, dry_run: bool) -> dt.datetime:
    if text is None:
        if not dry_run:
            raise SystemExit("--until HH:MM is required")
        return dt.datetime.now() + dt.timedelta(minutes=60)
    h, m = (int(x) for x in text.split(":"))
    t = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > dt.datetime.now() else t + dt.timedelta(days=1)


def _worktree() -> Path:
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], ROOT).stdout.strip()
    if not branch.startswith("auto/"):
        raise SystemExit(f"run, report and status work inside the auto/* worktree (this checkout is on {branch!r})")
    return ROOT


def _lock(wt: Path) -> None:
    lock = wt / "auto/driver.lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            raise SystemExit(f"another driver is running (pid {lock.read_text()})")
        except (ProcessLookupError, ValueError):
            pass  # stale lock
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))


def cmd_run(args) -> None:
    wt = _worktree()
    if sys.platform == "darwin" and not os.environ.get("AUTORESEARCH_CAFFEINATED") and shutil.which("caffeinate"):
        os.environ["AUTORESEARCH_CAFFEINATED"] = "1"
        os.execvp("caffeinate", ["caffeinate", "-i", sys.executable, *sys.argv])
    if claude_version(args.claude) < CLAUDE_MIN_VERSION:
        raise SystemExit(f"claude >= {'.'.join(map(str, CLAUDE_MIN_VERSION))} is needed for --permission-prompts")
    _lock(wt)
    args.until_dt = _until(args.until, args.dry_run)
    night = Night(wt, args)
    night.state["until"] = args.until_dt.isoformat(timespec="minutes")
    night.save()

    def on_sigint(signum, frame):
        if night.stop:  # second Ctrl-C: stop the training runs too
            for p in night.procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            raise SystemExit(130)
        night.stop = True
        print("stopping after the current batch, then writing the report (Ctrl-C again to abort)", flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    dirty = changed_paths(wt)
    if dirty == ["auto/ideas.md"]:
        commit(wt, "ideas for tonight")
    elif dirty:
        raise SystemExit(f"worktree has uncommitted changes ({', '.join(dirty[:5])}); commit or revert them first")
    pending = night.state.get("pending")
    if pending:  # crashed mid-batch: run.py skips the seeds that finished
        print(f"resuming batch: {[t['experiment'] for t in pending['trials']]}", flush=True)
        night.run_batch(pending["trials"], pending["message"])
    if not (wt / "auto/plan.md").exists():
        night.intake()
    reason, n = None, 0
    while reason is None:
        if night.stop:
            reason = "interrupted"
        elif args.max_steps is not None and n >= args.max_steps:
            reason = f"max steps ({args.max_steps})"
        elif night.state["cost_usd"] >= args.max_cost_usd:
            reason = f"cost limit (${args.max_cost_usd})"
        elif night.minutes_left() <= 0:
            reason = "deadline"
        else:
            n += 1
            reason = night.step(n)
        print(json.dumps({"step": night.state["step"], "cost_usd": round(night.state["cost_usd"], 2),
                          "trials": len(read_tsv(wt / "auto/results.tsv")), "stop": reason}), flush=True)
    night.report(reason)


def cmd_report(args) -> None:
    wt = _worktree()
    _lock(wt)
    args.until_dt = dt.datetime.now()
    night = Night(wt, args)
    night.report(night.state.get("stop_reason") or "report run on its own")


def cmd_status(args) -> None:
    wt = _worktree()
    state_path = wt / "auto/state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rows = read_tsv(wt / "auto/results.tsv")
    print(f"steps {state.get('step', 0)} · trials {len(rows)} ({sum(r['status'] == 'failed' for r in rows)} failed)"
          f" · cost ${state.get('cost_usd', 0):.2f} · until {state.get('until', '?')} · stop: {state.get('stop_reason')}")
    if state.get("until"):
        left = (dt.datetime.fromisoformat(state["until"]) - dt.datetime.now()).total_seconds() / 60
        print(f"time left {max(left, 0):.0f} min")
    best: dict[tuple, dict] = {}
    for r in rows:
        if r["status"] == "completed" and r["seeds"] == "0,1,2":
            k = (r["problem_id"], r["objective"])
            if k not in best or float(r["value_mean"]) < float(best[k]["value_mean"]):
                best[k] = r
    print("best confirmed (3-seed) per problem / objective:")
    for (pid, obj), r in sorted(best.items()):
        print(f"  {pid} {obj}: {r['experiment']} {float(r['value_mean']):.4g} ± {float(r['value_std']):.2g}")
    print("throughput per hour (runs, actual / prior wall time):")
    for h, v in sorted(state.get("hours", {}).items()):
        print(f"  {h[11:]}h  {v['runs']:4d} runs  x{v['ratio_sum'] / max(v['runs'], 1):.2f}")
    for kind in ("violations", "failures", "overrides"):
        for item in state.get(kind, [])[-3:]:
            print(f"{kind[:-1]}: {item}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=6)
    common.add_argument("--dry-run", action="store_true", help="wave data, 20 epochs, canned ideas, injected faults")
    common.add_argument("--claude", default="claude")
    common.add_argument("--model", default=None)
    common.add_argument("--max-call-usd", type=float, default=2.0)
    i = sub.add_parser("init", parents=[common])
    i.add_argument("--date")
    i.add_argument("--ideas")
    i.add_argument("--base", default="main")
    i.add_argument("--worktree")
    i.add_argument("--resume", action="store_true")
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--until")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--max-cost-usd", type=float, default=20.0)
    sub.add_parser("report", parents=[common])
    sub.add_parser("status", parents=[common])
    args = p.parse_args(argv)
    {"init": cmd_init, "run": cmd_run, "report": cmd_report, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
