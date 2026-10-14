"""Aggregate plots and tables for the meeting notebooks (reads meeting_runner caches)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .meeting_runner import LATENT_METHODS, _abs, load_arrays, load_results, load_seed_info

# Fixed categorical order (validated default palette); colour follows the method.
METHOD_COLORS = {
    "SHRED-noise": "#2a78d6",
    "SHRED-dropout": "#eb6834",
    "SDN-noise": "#1baf7a",
    "SDN-dropout": "#eda100",
    "SenseiverSDN": "#e87ba4",
    "RobustSHREDv1": "#008300",
    "RobustSHREDv2": "#6250d6",
    "SHRED-dropout+mask": "#e34948",
    "GappyPOD": "#7a7974",
}
CONDITION_COLORS = {"p0": "#2a78d6", "p0.3": "#eb6834"}
CONDITION_LABELS = {"p0": "p = 0 (noise only)", "p0.3": "p = 0.3 (noise + dropout)"}
TEXT = "#52514e"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bdbcb6",
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "axes.grid": True,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 110,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def available_methods(results: dict, methods: list[str]) -> list[str]:
    present = {m for m, _ in results}
    return [m for m in methods if m in present]


def _seeds(results: dict, method: str) -> list[int]:
    return sorted(s for m, s in results if m == method)


def metric_over_seeds(results: dict, method: str, cond: str, key: str = "err") -> tuple[float, float, int]:
    vals = [results[(method, s)]["metrics"][cond][key] for s in _seeds(results, method)]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


def summary_table(results: dict, methods: list[str]) -> str:
    """Markdown table: mean ± std over seeds of relative L2 error at p=0 and p=0.3."""
    rows = ["| Method | Params | Seeds | Clean | p = 0 | p = 0.3 | p = 0.3 (u) | p = 0.3 (v) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in available_methods(results, methods):
        first = results[(m, _seeds(results, m)[0])]["metrics"]
        params = f"{first['parameters']:,}" if "parameters" in first else "–"
        cells = []
        for cond, key in [("clean", "err"), ("p0", "err"), ("p0.3", "err"), ("p0.3", "err_u"), ("p0.3", "err_v")]:
            mean, std, n = metric_over_seeds(results, m, cond, key)
            cells.append(f"{mean:.4f} ± {std:.4f}")
        rows.append(f"| {m} | {params} | {len(_seeds(results, m))} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def plot_error_bars(results: dict, methods: list[str], ax=None):
    """Grouped bars: error at p=0 vs p=0.3 per method (mean ± std over seeds)."""
    methods = available_methods(results, methods)
    ax = ax or plt.subplots(figsize=(1.1 * len(methods) + 2, 4))[1]
    x = np.arange(len(methods))
    width = 0.38
    for i, cond in enumerate(("p0", "p0.3")):
        stats_ = [metric_over_seeds(results, m, cond) for m in methods]
        ax.bar(x + (i - 0.5) * (width + 0.02), [s[0] for s in stats_], width, yerr=[s[1] for s in stats_],
               color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond], capsize=3,
               error_kw={"elinewidth": 1, "ecolor": TEXT})
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylabel("relative L2 error (test, unscaled)")
    ax.legend(frameon=False)
    ax.set_title("Reconstruction error without and with sensor dropout", loc="left")
    return ax


def per_sample(root, results: dict, method: str, key: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-seed arrays of (values, k) for a per-sample key such as 'p0.3_err'."""
    cond = key.split("_")[0]
    values, ks = [], []
    for s in _seeds(results, method):
        arrays = load_arrays(root, method, s)
        if key in arrays:
            values.append(arrays[key])
            ks.append(arrays[f"{cond}_k"])
    return values, ks


def by_k(values: list[np.ndarray], ks: list[np.ndarray]) -> dict[int, tuple[float, float, int]]:
    """{k: (mean over seeds of per-seed mean, std over seeds, total sample count)}."""
    out = {}
    for k in sorted(set(np.concatenate(ks).tolist())):
        per_seed = [v[kk == k].mean() for v, kk in zip(values, ks) if (kk == k).any()]
        out[int(k)] = (float(np.mean(per_seed)), float(np.std(per_seed)), int(sum((kk == k).sum() for kk in ks)))
    return out


def k_counts_table(root, results: dict, method: str, cond: str = "p0.3") -> str:
    values, ks = per_sample(root, results, method, f"{cond}_err")
    counts = by_k(values, ks)
    return ("| k failed | " + " | ".join(str(k) for k in counts) + " |\n|---|" + "---:|" * len(counts) +
            "\n| samples (all seeds) | " + " | ".join(str(c[2]) for c in counts.values()) + " |")


def plot_by_k(root, results: dict, methods: list[str], key: str, ylabel: str, title: str,
              min_count: int = 10, ax=None, logy: bool = False):
    """Mean per-sample metric vs number of failed sensors k (bins with < min_count samples hidden)."""
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    for m in available_methods(results, methods):
        values, ks = per_sample(root, results, m, key)
        if not values:
            continue
        table = {k: v for k, v in by_k(values, ks).items() if v[2] >= min_count}
        k = np.array(list(table))
        mean = np.array([v[0] for v in table.values()])
        std = np.array([v[1] for v in table.values()])
        ax.plot(k, mean, marker="o", color=METHOD_COLORS[m], label=m)
        ax.fill_between(k, mean - std, mean + std, color=METHOD_COLORS[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("number of failed sensors k")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return ax


def _field(vec: np.ndarray, info: dict, channel: int) -> np.ndarray:
    n_points = int(info["n_points"])
    nx = int(info["nx"])
    return vec[channel * n_points:(channel + 1) * n_points].reshape(nx, n_points // nx, order="F")


def _extent(info: dict) -> list[float]:
    pc = info["point_coordinates"]
    return [pc[:, 0].min(), pc[:, 0].max(), pc[:, 1].min(), pc[:, 1].max()]


def plot_sensor_layout(info: dict, snapshot: np.ndarray, title: str = ""):
    """u and v snapshot with the probe locations."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), constrained_layout=True)
    probes = info["point_coordinates"][info["locations"]]
    for c, (ax, name) in enumerate(zip(axes, ("u", "v"))):
        field = _field(snapshot, info, c)
        lim = np.abs(field - (field.mean() if c == 0 else 0)).max()
        center = field.mean() if c == 0 else 0
        im = ax.imshow(field, origin="lower", extent=_extent(info), cmap="RdBu_r", vmin=center - lim, vmax=center + lim)
        ax.scatter(probes[:, 0], probes[:, 1], s=40, facecolor="#0b0b0b", edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(f"{name} {title}", loc="left")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    return fig


def plot_reconstruction_panel(root, results: dict, methods: list[str], seed: int, window: int = 0,
                              cond: str = "p0.3", channel: int = 1):
    """Truth (probes; failed ones marked x) vs each method's reconstruction, draw 0 of one test window."""
    info = load_seed_info(root, seed)
    methods = [m for m in available_methods(results, methods) if (m, seed) in results]
    truth = info["panel_truth"][window]
    mask = info[f"test_mask_{cond}"][0, window]
    fields = [("truth", truth)] + [(m, load_arrays(root, m, seed)[f"{cond}_panel_recon"][window]) for m in methods]
    ncols = 3
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), constrained_layout=True, squeeze=False)
    truth_field = _field(truth, info, channel)
    center = truth_field.mean() if channel == 0 else 0.0
    lim = np.abs(truth_field - center).max()
    probes = info["point_coordinates"][info["locations"]]
    for ax, (name, vec) in zip(axes.ravel(), fields):
        im = ax.imshow(_field(vec, info, channel), origin="lower", extent=_extent(info), cmap="RdBu_r",
                       vmin=center - lim, vmax=center + lim)
        if name == "truth":
            ax.scatter(*probes[~mask].T, s=36, facecolor="#0b0b0b", edgecolor="white", linewidth=1.5, zorder=3)
            ax.scatter(*probes[mask].T, s=60, marker="x", color="#d03b3b", linewidth=2.5, zorder=3)
            ax.set_title(f"truth ({int(mask.sum())} failed: ×)", loc="left", fontsize=9)
        else:
            err = np.linalg.norm(vec - truth) / np.linalg.norm(truth)
            ax.set_title(f"{name}  (err {err:.3f})", loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[len(fields):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(f"{'uv'[channel]} at {CONDITION_LABELS[cond]}, seed {seed}, test window {window}", x=0.01, ha="left")
    return fig


def latent_correlations(root, results: dict, methods: list[str], cond: str) -> dict[str, dict[str, float]]:
    out = {}
    for m in [m for m in available_methods(results, methods) if m in LATENT_METHODS]:
        dl, _ = per_sample(root, results, m, f"{cond}_d_latent")
        dr, _ = per_sample(root, results, m, f"{cond}_d_recon")
        if not dl:
            continue
        a, b = np.concatenate(dl), np.concatenate(dr)
        out[m] = {"pearson": float(stats.pearsonr(a, b)[0]), "spearman": float(stats.spearmanr(a, b)[0]),
                  "d_latent_mean": float(a.mean()), "d_recon_mean": float(b.mean()), "n": int(a.size)}
    return out


def plot_latent_scatter(root, results: dict, methods: list[str], cond: str = "p0.3"):
    """Per model: D_latent vs D_recon per corrupted test sample (all seeds), with correlations."""
    methods = [m for m in available_methods(results, methods) if m in LATENT_METHODS]
    corr = latent_correlations(root, results, methods, cond)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.2), constrained_layout=True,
                             squeeze=False)
    for ax, m in zip(axes.ravel(), methods):
        dl, ks = per_sample(root, results, m, f"{cond}_d_latent")
        dr, _ = per_sample(root, results, m, f"{cond}_d_recon")
        a, b, k = np.concatenate(dl), np.concatenate(dr), np.concatenate(ks)
        sc = ax.scatter(a, b, c=k, cmap="Blues", vmin=-1, vmax=max(k.max(), 1), s=10,
                        edgecolor=METHOD_COLORS[m], linewidth=0.4)
        c = corr[m]
        ax.set_title(f"{m}\nPearson {c['pearson']:.2f} · Spearman {c['spearman']:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("D_latent")
    axes.ravel()[0].set_ylabel("D_recon")
    if cond != "p0":
        fig.colorbar(sc, ax=axes, shrink=0.8, label="failed sensors k")
    fig.suptitle(f"Latent vs reconstruction change, {CONDITION_LABELS[cond]}", x=0.01, ha="left")
    return fig


def save(fig, root, name: str) -> Path:
    path = _abs(root) / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


__all__ = [
    "METHOD_COLORS", "available_methods", "metric_over_seeds", "summary_table", "plot_error_bars", "per_sample",
    "by_k", "k_counts_table", "plot_by_k", "plot_sensor_layout", "plot_reconstruction_panel",
    "latent_correlations", "plot_latent_scatter", "save", "load_results",
]
