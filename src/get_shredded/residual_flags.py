"""Residual-based fault flags, test time only (experiments/residual_flags).

A faulty reading should disagree with the field the model predicts at that probe's own location. Pass 1 runs a
trained masked model with no flags; the last-lag residual r = reading - prediction (physical units) per probe and
channel gives a joint score d_j = sqrt(sum_c (r_cj / s_cj)^2), s = residual std on clean (noisy) validation
windows. A probe is flagged for the whole window if d_j > tau_j (quantile q of d_j on validation), and pass 2
reruns the model with those flags. Faults are the runner's own (whole window, both channels), so the last lag of
the window is the only residual a window has. Checkpoints and every draw come from the source experiment.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch
from omegaconf import OmegaConf

from .corruption import CorruptionSpec
from .runner import (
    _abs,
    _build_model,
    _device,
    _evaluate,
    _format_inputs,
    _predict,
    _stable_config_fingerprint,
    eval_conditions,
    prepare_seed,
    require_matching_result_summary,
    run_config,
    write_result_summary,
)

FLAG_MODES = ("oracle", "silent", "residual", "soft", "topk")
SCORED = ("residual", "soft", "topk")  # modes that run pass 1; they also save scores to arrays.npz


def probe_residuals(pred_phys: np.ndarray, x_scaled: np.ndarray, sd) -> np.ndarray:
    """Last-lag reading minus prediction at the probe's own grid point, physical units, shape (B, C, N).

    pred_phys: (B, m) unscaled prediction; x_scaled: (B, lags, >= C*N) network input (scaled readings first).
    """
    cn = sd.channels * sd.num_sensors
    readings = x_scaled[:, -1, :cn] * sd.sensor_range + sd.scaler.data_min_[sd.state_indices]
    return (readings - pred_phys[:, sd.state_indices]).reshape(-1, sd.channels, sd.num_sensors)


def joint_score(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(B, C, N) residuals, (C, N) scales -> (B, N) normalised residual norm over channels."""
    return np.sqrt(((r / s) ** 2).sum(axis=1))


def fit_thresholds(r_valid: np.ndarray, qs) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Residual std s (C, N) and, per q, tau (N,) = quantile q of the joint score on the clean validation residuals."""
    s = r_valid.std(axis=0)
    d = joint_score(r_valid, s)
    return s, {float(q): np.quantile(d, float(q), axis=0) for q in qs}


def top_k_flags(d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Flag the k[b] highest scores of row b of d (B, N); upper bound for any rule that ranks probes."""
    return np.argsort(-d, axis=1).argsort(axis=1) < np.asarray(k)[:, None]


def detection_metrics(flags: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    """Per (window, probe) detection against the true fault mask; None where undefined (no positives)."""
    tp = int((flags & truth).sum())
    fp = int((flags & ~truth).sum())
    fn = int((~flags & truth).sum())
    neg = int((~truth).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if tp + fn else None)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / neg if neg else None}


def _inner_predict(inner, x: np.ndarray, cn: int, bias: np.ndarray, device: str, batch: int = 512):
    """RobustSHREDv1 with an attention-logit bias (B, N); flags columns of x are ignored."""
    recon, lat = [], []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch):
            xb = torch.tensor(x[start:start + batch, :, :cn], device=device)
            h = inner.latent(xb, None, bias=torch.tensor(bias[start:start + batch], device=device))
            recon.append(inner.decoder(h).cpu().numpy())
            lat.append(h.cpu().numpy())
    return np.concatenate(recon), np.concatenate(lat)


def self_influence(model, method, sd, deltas, device: str) -> np.ndarray:
    """Step 0: response of the prediction at every probe to an offset of one reading at the last lag only.

    Clean test windows, all probes in. delta = factor x training std of the offset probe and channel. Returns
    (len(deltas), C*N offset, C*N measured), mean over windows, in std units: (du_hat(x_j) / std_j) / (delta_i / std_i).
    The diagonal is s_j = du_hat(x_j) / delta (near 0: the model does not copy the reading); off-diagonal entries
    show how far one bad reading moves the prediction at the other probes.
    """
    cn = sd.channels * sd.num_sensors
    x = _format_inputs(sd, sd.windows[sd.test_idx], None, None, CorruptionSpec(noise=False), method)
    base = sd.scaler.inverse_transform(_predict(model, x, device, False)[0])[:, sd.state_indices]
    std = sd.load_X[sd.train_idx][:, sd.state_indices].std(axis=0)
    out = np.zeros((len(deltas), cn, cn))
    for i, factor in enumerate(deltas):
        for col in range(cn):
            shifted = x.copy()
            shifted[:, -1, col] += float(factor) * std[col] / sd.sensor_range[col]
            pred = sd.scaler.inverse_transform(_predict(model, shifted, device, False)[0])[:, sd.state_indices]
            out[i, col] = ((pred - base) / std).mean(axis=0) / float(factor)
    return out


def _source_cfg(cfg, name: str | None = None):
    """A source experiment's config: same data, sensors, conditions and draws, its name for the fingerprint."""
    src = cfg.copy()
    src.name = str(name or cfg.residual_flags.source)
    return src


def _run_source(rf, run_name: str) -> str:
    """Experiment holding a run's checkpoint: residual_flags.sources[run], else residual_flags.source."""
    return str((rf.get("sources") or {}).get(run_name, rf.source))


def _load(src, sd, run_name: str, seed: int, device: str):
    method_key, train = run_name.split("__")
    run = OmegaConf.create({"method": method_key, "train": train})
    root = _abs(src.outputs_root) / str(src.name) / str(src.data.name) / f"seed{seed}" / run_name
    if require_matching_result_summary(root / "result.json", _stable_config_fingerprint(run_config(src, run, seed))) is None:
        raise FileNotFoundError(f"no completed checkpoint at {root}")
    method = src.methods[method_key]
    model = _build_model(sd, method, device, src.data)
    model.load_state_dict(torch.load(root / "model.pt", map_location=device))
    model.eval()
    return model, method


def variant_name(v) -> str:
    name = f"{v.run}__{v.flags}"
    return name + (f"_q{float(v.q):g}" if v.flags == "residual" else "")  # topk: no threshold


def _evaluate_variant(cfg, sd, model, method, v, s, taus, device: str):
    if v.flags not in FLAG_MODES:
        raise ValueError(f"flags must be one of {FLAG_MODES}, got {v.flags!r}")
    if v.flags == "soft" and method.kind != "robust_v1":
        raise ValueError("soft flags need robust_v1 (attention-logit bias)")
    cn = sd.channels * sd.num_sensors
    test_windows = sd.windows[sd.eval_idx]
    draws = sd.test_noise.shape[0]
    calls = []  # (true mask, flags, pass-1 score or None) per recon_fn call, in _evaluate's condition order

    def recon_fn(noise, mask, spec, rng):
        base = test_windows if noise is None else np.tile(test_windows, (draws, 1, 1))
        x = _format_inputs(sd, base, noise, mask, replace(spec, flags="silent"), method, rng)  # true mask drives the fault
        truth = np.zeros((x.shape[0], sd.num_sensors), bool) if mask is None else mask
        if v.flags in ("oracle", "silent"):
            flags = truth if v.flags == "oracle" else np.zeros_like(truth)
        else:  # pass 1: no flags, all probes in
            d = joint_score(probe_residuals(sd.scaler.inverse_transform(_predict(model, x, device, False)[0]), x, sd), s)
            if v.flags == "soft":
                bias = np.log(np.maximum(np.exp(-0.5 * d ** 2), 1e-6)).astype(np.float32)
                calls.append((truth, d > taus[float(cfg.residual_flags.soft_q)], d))
                pred, latent = _inner_predict(model.model, x, cn, bias, device)
                return sd.scaler.inverse_transform(pred), latent
            # topk: the true fault count is given (oracle knowledge of k only, not of which probes)
            flags = top_k_flags(d, truth.sum(axis=1)) if v.flags == "topk" else d > taus[float(v.q)]
        calls.append((truth, flags, None if v.flags in ("oracle", "silent") else d))
        x[..., cn:] = flags[:, None, :].astype(np.float32)
        pred, latent = _predict(model, x, device, True)
        return sd.scaler.inverse_transform(pred), latent

    garbage_cfg = cfg.copy()  # evaluate only the source's garbage conditions; sd's mask draws are unchanged
    garbage_cfg.eval.conditions = [c for c in cfg.eval.conditions if c.corruption == "garbage" and c.flags == "silent"]
    metrics, ev_arrays = _evaluate(garbage_cfg, sd, recon_fn, np.arange(0))
    metrics = {k.replace("garbage_silent_", "garbage_"): val for k, val in metrics.items()}
    tags = ["clean"] + [k.replace("garbage_silent_", "garbage_") for k in eval_conditions(garbage_cfg)]
    arrays = {}
    for tag, (truth, flags, d) in zip(tags, calls):
        metrics[tag]["detection"] = detection_metrics(flags, truth)
        if d is not None:  # rows: (draw, test window) flattened draw-major, as the per-snapshot errors
            src_tag = tag.replace("garbage_", "garbage_silent_")
            arrays.update({f"{tag}_score": d.astype(np.float32), f"{tag}_truth": truth, f"{tag}_flags": flags,
                           f"{tag}_err": ev_arrays[f"{src_tag}_err"] if tag != "clean" else ev_arrays["clean_err"]})
    return metrics, arrays


def run_residual_flags(cfg) -> None:
    rf = cfg.residual_flags
    src = _source_cfg(cfg)
    device = _device(str(cfg.device))
    root = _abs(cfg.outputs_root) / str(cfg.name) / str(cfg.data.name)
    qs = sorted({float(q) for q in rf.qs})
    runs = sorted({str(v.run) for v in rf.variants} | {str(r) for r in rf.self_influence_runs})
    for seed in [int(s) for s in cfg.eval.seeds]:
        sd = prepare_seed(src, seed)
        for run_name in runs:
            model, method = _load(_source_cfg(cfg, _run_source(rf, run_name)), sd, run_name, seed, device)
            out = root / f"seed{seed}" / f"self_influence__{run_name}.json"
            if run_name in rf.self_influence_runs and not out.exists():
                si = self_influence(model, method, sd, list(rf.self_influence_deltas), device)
                diag = np.stack([np.diag(a) for a in si]).reshape(len(si), sd.channels, sd.num_sensors)
                write_result_summary(out, {"run": run_name, "seed": seed, "deltas": list(rf.self_influence_deltas),
                                           "s": diag.tolist(), "matrix": si.tolist(), "device": device})
                print(json.dumps({"self_influence": run_name, "seed": seed,
                                  "mean_diag_by_delta": [round(float(a.mean()), 4) for a in diag]}), flush=True)
            # calibration: clean validation windows with the fixed validation noise, no faults, no flags
            valid_x = _format_inputs(sd, sd.windows[sd.valid_idx], sd.valid_noise, None, CorruptionSpec(), method)
            r_valid = probe_residuals(sd.scaler.inverse_transform(_predict(model, valid_x, device, False)[0]),
                                      valid_x, sd)
            s, taus = fit_thresholds(r_valid, qs)
            for v in [v for v in rf.variants if str(v.run) == run_name]:
                path = root / f"seed{seed}" / variant_name(v) / "result.json"
                if path.exists() and (v.flags not in SCORED or (path.parent / "arrays.npz").exists()):
                    print(json.dumps({"status": "skipped", "run": variant_name(v), "seed": seed}))
                    continue
                metrics, arrays = _evaluate_variant(cfg, sd, model, method, v, s, taus, device)
                if arrays:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path.parent / "arrays.npz", **arrays)
                write_result_summary(path, {
                    "status": "completed", "run": variant_name(v), "seed": seed, "device": device,
                    "source": _run_source(rf, run_name), "variant": OmegaConf.to_container(v), "metrics": metrics,
                    "residual_std": s.tolist(), "tau": {f"{q:g}": t.tolist() for q, t in taus.items()},
                })
                print(json.dumps({"status": "completed", "run": variant_name(v), "seed": seed,
                                  "garbage_k2": round(metrics["garbage_k2"]["err"], 5)}), flush=True)
    write_result_summary(root / "summary.json", build_residual_summary(root, [variant_name(v) for v in rf.variants],
                                                                       [int(s) for s in cfg.eval.seeds]))


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return [float(np.mean(vals)), float(np.std(vals))] if vals else None


def build_residual_summary(root, variants: list[str], seeds: list[int]) -> dict:
    """Mean/std over seeds of err and detection per variant and condition, plus self-influence per run."""
    summary: dict = {"runs": {}, "self_influence": {}, "missing": [], "seeds": seeds}
    for name in variants:
        done = []
        for seed in seeds:
            path = root / f"seed{seed}" / name / "result.json"
            if path.exists():
                done.append(json.loads(path.read_text()))
            else:
                summary["missing"].append(f"{name}/seed{seed}")
        if not done:
            continue
        entry = {"n_seeds": len(done)}
        for cond in done[0]["metrics"]:
            entry[f"{cond}_err"] = _mean_std([r["metrics"][cond]["err"] for r in done])
            for key in ("precision", "recall", "f1", "false_alarm"):
                entry[f"{cond}_{key}"] = _mean_std([r["metrics"][cond]["detection"][key] for r in done])
        summary["runs"][name] = entry
    for path in sorted(root.glob("seed*/self_influence__*.json")):
        payload = json.loads(path.read_text())
        for key, store in (("s", "self_influence"), ("matrix", "influence_matrix")):
            summary.setdefault(store, {}).setdefault(payload["run"], {})[f"seed{payload['seed']}"] = payload[key]
    for store in ("self_influence", "influence_matrix"):
        for run, by_seed in summary.get(store, {}).items():
            arr = np.array([by_seed[k] for k in sorted(by_seed)])  # (seeds, deltas, ...)
            by_seed["mean_over_seeds"] = arr.mean(axis=0).tolist()
            by_seed["std_over_seeds"] = arr.std(axis=0).tolist()
    return summary
