from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wave_1d.npz"


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as payload:
        field = np.asarray(payload["field"], dtype=np.float32)
        x = np.asarray(payload["x"], dtype=np.float64)
        t = np.asarray(payload["t"], dtype=np.float64)
        metadata = {key: np.asarray(value) for key, value in payload.items() if key not in {"field", "x", "t"}}
    return field, x, t, metadata


def main() -> None:
    data_path = DEFAULT_DATA_PATH
    field, x, t, metadata = _load_dataset(data_path)
    num_trajectories, num_times, num_space = field.shape

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[3, 1])
    ax_field = fig.add_subplot(gs[0, 0])
    ax_space_time = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    traj_index = 0
    time_index = 0
    playing = False

    def draw_state() -> None:
        nonlocal traj_index, time_index
        current = field[traj_index]
        current_field = current[time_index]

        ax_field.clear()
        ax_field.plot(x, current_field, color="tab:blue", linewidth=2)
        ax_field.set_title(f"u(x, t) — trajectory {traj_index}, time {time_index}")
        ax_field.set_xlabel("x")
        ax_field.set_ylabel("u")
        ax_field.set_xlim(x.min(), x.max())
        ax_field.grid(alpha=0.3)

        ax_space_time.clear()
        space_time = current
        image = ax_space_time.imshow(
            space_time.T,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], x[0], x[-1]],
            interpolation="nearest",
        )
        ax_space_time.set_title("Space-time field")
        ax_space_time.set_xlabel("t")
        ax_space_time.set_ylabel("x")
        ax_space_time.axvline(t[time_index], color="white", linewidth=1.5)
        fig.colorbar(image, ax=ax_space_time, pad=0.02, label="u")

        ax_info.clear()
        ax_info.text(
            0.01,
            0.55,
            (
                f"trajectory: {traj_index} | time: {t[time_index]:.4f} | frame index: {time_index} | "
                f"field shape: ({num_trajectories}, {num_times}, {num_space})"
            ),
            va="center",
            fontsize=10,
        )
        ax_info.set_axis_off()
        fig.canvas.draw_idle()

    def on_traj_change(val):
        nonlocal traj_index
        traj_index = int(val)
        draw_state()

    def on_time_change(val):
        nonlocal time_index
        time_index = int(val)
        draw_state()

    def toggle_play(_event):
        nonlocal playing
        playing = not playing
        play_button.label.set_text("Pause" if playing else "Play")

    def animate(_):
        if not playing:
            return
        nonlocal time_index
        time_index = (time_index + 1) % num_times
        time_slider.set_val(time_index)

    traj_slider_ax = fig.add_axes([0.18, 0.09, 0.52, 0.03])
    time_slider_ax = fig.add_axes([0.18, 0.04, 0.52, 0.03])
    play_ax = fig.add_axes([0.72, 0.06, 0.12, 0.05])

    traj_slider = Slider(traj_slider_ax, "Trajectory", valmin=0, valmax=num_trajectories - 1, valinit=0, valstep=1)
    time_slider = Slider(time_slider_ax, "Time", valmin=0, valmax=num_times - 1, valinit=0, valstep=1)
    play_button = Button(play_ax, "Play")

    traj_slider.on_changed(on_traj_change)
    time_slider.on_changed(on_time_change)
    play_button.on_clicked(toggle_play)

    draw_state()
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        animate(None)
        plt.pause(0.05)


if __name__ == "__main__":
    main()
