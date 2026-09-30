from __future__ import annotations

from pathlib import Path

from get_shredded.wave_data import WaveGeneratorConfig, generate_wave_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "wave_1d.npz"


def main() -> None:
    config = WaveGeneratorConfig(
        L=2.0,
        c=2.0,
        num_modes=10,
        num_trajectories=1000,
        nx=256,
        nt=500,
        t_start=0.0,
        t_end=2.0,
        endpoint=True,
        seed=42,
    )
    print(f"Generating {config.num_trajectories} trajectories...")
    dataset = generate_wave_dataset(config, output_path=DATA_PATH)
    field = dataset["field"]
    print(f"Field shape: {field.shape}")
    print(f"Saving to {DATA_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
