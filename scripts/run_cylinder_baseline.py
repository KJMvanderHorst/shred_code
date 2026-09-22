"""Train SHRED + SDN, compute QR/POD baseline, save metrics + plots + GIF."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from get_shredded.experiment import run_experiment
from get_shredded.plotting import (
    animate_reconstructions,
    plot_per_snapshot_error,
    plot_reconstruction_panel,
    plot_training_curves,
)


@hydra.main(version_base=None, config_path="../configs", config_name="cylinder_baseline")
def main(cfg: DictConfig) -> None:
    result = run_experiment(
        mat_path=Path(to_absolute_path(cfg.data.mat)),
        num_sensors=int(cfg.model.num_sensors),
        lags=int(cfg.model.lags),
        placement=str(cfg.model.placement),
        test_size=int(cfg.data.test_size),
        val_size=int(cfg.data.val_size),
        hidden_size=int(cfg.model.hidden_size),
        hidden_layers=int(cfg.model.hidden_layers),
        l1=int(cfg.model.l1),
        l2=int(cfg.model.l2),
        dropout=float(cfg.model.dropout),
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.train.lr),
        patience=int(cfg.train.patience),
        seed=int(cfg.seed),
        noise_enabled=bool(cfg.noise.enabled),
        noise_modes=[str(mode) for mode in cfg.noise.modes],
        noise_white_std=float(cfg.noise.white_std),
        noise_none_fill_value=float(cfg.noise.none_fill_value),
        noise_auto_extend=bool(cfg.noise.auto_extend),
        noise_default_mode=str(cfg.noise.default_mode),
        noise_seed=int(cfg.noise.seed) if cfg.noise.seed is not None else None,
        senseiver_enabled=bool(cfg.senseiver.enabled),
        senseiver_num_latents=int(cfg.senseiver.num_latents),
        senseiver_latent_dim=int(cfg.senseiver.latent_dim),
        senseiver_num_frequencies=int(cfg.senseiver.num_frequencies),
        senseiver_num_heads=int(cfg.senseiver.num_heads),
        senseiver_dropout=float(cfg.senseiver.dropout),
        senseiver_sdn_enabled=bool(cfg.senseiver.sdn_enabled),
        senseiver_sdn_l1=int(cfg.senseiver.sdn_l1),
        senseiver_sdn_l2=int(cfg.senseiver.sdn_l2),
        robust_shred_v1_enabled=bool(cfg.robust_shred.v1_enabled),
        robust_shred_v2_enabled=bool(cfg.robust_shred.v2_enabled),
        robust_shred_hidden_size=int(cfg.robust_shred.hidden_size),
        robust_shred_hidden_layers=int(cfg.robust_shred.hidden_layers),
        robust_shred_num_heads=int(cfg.robust_shred.num_heads),
        robust_shred_embed_dim=int(cfg.robust_shred.embed_dim),
        robust_shred_num_latents=int(cfg.robust_shred.num_latents),
        robust_shred_latent_dim=int(cfg.robust_shred.latent_dim),
        robust_shred_num_frequencies=int(cfg.robust_shred.num_frequencies),
        robust_shred_l1=int(cfg.robust_shred.l1),
        robust_shred_l2=int(cfg.robust_shred.l2),
        robust_shred_dropout=float(cfg.robust_shred.dropout),
        verbose=True,
    )

    print(f"\nNum sensors: {result.num_sensors} | Placement: {result.placement} | Lags: {result.lags}")
    print(f"SHRED     relative L2 error: {result.shred_err:.6f}")
    print(f"SDN       relative L2 error: {result.sdn_err:.6f}")
    if result.senseiver_err is not None:
        print(f"Senseiver relative L2 error: {result.senseiver_err:.6f}")
    if result.senseiver_sdn_err is not None:
        print(f"SenseiverSDN relative L2 error: {result.senseiver_sdn_err:.6f}")
    if result.robust_shred_v1_err is not None:
        print(f"RobustSHREDv1 relative L2 error: {result.robust_shred_v1_err:.6f}")
    if result.robust_shred_v2_err is not None:
        print(f"RobustSHREDv2 relative L2 error: {result.robust_shred_v2_err:.6f}")
    print(f"QR/POD    relative L2 error: {result.qrpod_err:.6f}")
    for model_name, counts in result.parameter_counts.items():
        details = ", ".join(f"{name}={value}" for name, value in counts.items())
        print(f"{model_name} parameters: {details}")

    outputs_root = Path(to_absolute_path(cfg.outputs.root))
    recon_dir = outputs_root / "reconstructions"
    curves_dir = outputs_root / "curves"

    n_test = result.truth.shape[0]
    snap_indices = sorted({0, n_test // 2, n_test - 1})
    plot_reconstruction_panel(result, snap_indices, recon_dir / "panel.png")
    animate_reconstructions(result, recon_dir / "comparison.gif", fps=int(cfg.outputs.gif_fps))
    plot_per_snapshot_error(result, curves_dir / "per_snapshot_error.png")
    plot_training_curves(result, curves_dir / "training_curves.png")
    print(f"Saved plots under {outputs_root}/")

    if cfg.checkpoint.enabled:
        ckpt_dir = Path(to_absolute_path(cfg.checkpoint.dir))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / str(cfg.checkpoint.name)

        torch.save(
            {
                "sensor_locations": result.sensor_locations,
                "shred_state_dict": result.shred_state_dict,
                "sdn_state_dict": result.sdn_state_dict,
                "senseiver_state_dict": result.senseiver_state_dict,
                "senseiver_sdn_state_dict": result.senseiver_sdn_state_dict,
                "robust_shred_v1_state_dict": result.robust_shred_v1_state_dict,
                "robust_shred_v2_state_dict": result.robust_shred_v2_state_dict,
                "shred_val_history": result.shred_val_history,
                "sdn_val_history": result.sdn_val_history,
                "senseiver_val_history": result.senseiver_val_history,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "metrics": {
                    "shred_err": result.shred_err,
                    "sdn_err": result.sdn_err,
                    "senseiver_err": result.senseiver_err,
                    "senseiver_sdn_err": result.senseiver_sdn_err,
                    "robust_shred_v1_err": result.robust_shred_v1_err,
                    "robust_shred_v2_err": result.robust_shred_v2_err,
                    "qrpod_err": result.qrpod_err,
                },
            },
            ckpt,
        )
        print(f"Saved metrics: {ckpt}")


if __name__ == "__main__":
    main()
