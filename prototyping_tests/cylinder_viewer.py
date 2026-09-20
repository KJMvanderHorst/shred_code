from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.io import loadmat


# ============================================================
# Load data
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
mat_path = PROJECT_ROOT / "data" / "CYLINDER_ALL.mat"

print(f"Loading {mat_path}...")

data = loadmat(mat_path)

print("Finished loading data.")

vortall = np.asarray(data["VORTALL"], dtype=np.float32)

nx = int(data.get("nx", [[0]])[0][0])
ny = int(data.get("ny", [[0]])[0][0])

# VORTALL is assumed to have shape:
#   (number_of_spatial_points, number_of_time_frames)
#
# We transpose it so that:
#   snapshots[frame] = one complete spatial snapshot
snapshots = vortall.T

num_frames, num_state_values = snapshots.shape

print(f"nx = {nx}")
print(f"ny = {ny}")
print(f"Number of frames = {num_frames}")
print(f"Spatial values per frame = {num_state_values}")

if nx <= 0 or ny <= 0:
    raise ValueError(f"Invalid grid dimensions: nx={nx}, ny={ny}")

if nx * ny != num_state_values:
    raise ValueError(
        f"Grid dimensions do not match data: "
        f"nx * ny = {nx * ny}, but each frame has {num_state_values} values."
    )


# ============================================================
# Color scale
# ============================================================

print("Calculating color scale...")

vmax = float(np.percentile(np.abs(snapshots), 99.5))
vmax = max(vmax, 1e-6)

print(f"Color scale: +/- {vmax:.4g}")


# ============================================================
# Viewer state
# ============================================================

current_frame = 0
start_frame = None
end_frame = None
playing = False

# Time between frames during playback, in milliseconds
interval = 100


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Leave space at the bottom for controls
plt.subplots_adjust(bottom=0.28)


# ============================================================
# Initial frame
# ============================================================

field = snapshots[0].reshape((nx, ny), order="F")

# IMPORTANT:
# No .T here.
#
# This is the orientation where the x-direction is displayed
# horizontally and the y-direction vertically.
image = ax.imshow(
    field,
    cmap="seismic",
    vmin=-vmax,
    vmax=vmax,
    aspect="auto",
    interpolation="nearest",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Cylinder wake — frame 0")

fig.colorbar(
    image,
    ax=ax,
    label="Vorticity",
)


# ============================================================
# Frame slider
# ============================================================

slider_ax = fig.add_axes(
    [0.15, 0.13, 0.70, 0.035]
)

frame_slider = Slider(
    ax=slider_ax,
    label="Frame",
    valmin=0,
    valmax=num_frames - 1,
    valinit=0,
    valstep=1,
)


# ============================================================
# Buttons
# ============================================================

play_ax = fig.add_axes(
    [0.15, 0.045, 0.12, 0.05]
)

start_ax = fig.add_axes(
    [0.30, 0.045, 0.14, 0.05]
)

end_ax = fig.add_axes(
    [0.47, 0.045, 0.14, 0.05]
)

play_button = Button(
    play_ax,
    "Play",
)

start_button = Button(
    start_ax,
    "Mark start",
)

end_button = Button(
    end_ax,
    "Mark end",
)


# ============================================================
# Period display
# ============================================================

period_text = fig.text(
    0.70,
    0.075,
    "Select a start and end frame.",
    ha="center",
    va="center",
)


# ============================================================
# Update displayed frame
# ============================================================

def update_frame(frame):
    global current_frame

    frame = int(frame)
    current_frame = frame

    field = snapshots[frame].reshape(
        (nx, ny),
        order="F",
    )

    image.set_data(field)

    ax.set_title(
        f"Cylinder wake — frame {frame} / {num_frames - 1}"
    )

    fig.canvas.draw_idle()


# ============================================================
# Slider callback
# ============================================================

def slider_changed(value):
    update_frame(int(value))


frame_slider.on_changed(slider_changed)


# ============================================================
# Period calculation
# ============================================================

def update_period_text():
    if start_frame is None or end_frame is None:

        period_text.set_text(
            "Select a start and end frame."
        )

    else:

        period = abs(end_frame - start_frame)

        period_text.set_text(
            f"Start: {start_frame}   |   "
            f"End: {end_frame}   |   "
            f"Candidate period: {period} frames"
        )

    fig.canvas.draw_idle()


# ============================================================
# Mark start
# ============================================================

def mark_start(event):
    global start_frame

    start_frame = current_frame

    update_period_text()


start_button.on_clicked(mark_start)


# ============================================================
# Mark end
# ============================================================

def mark_end(event):
    global end_frame

    end_frame = current_frame

    update_period_text()


end_button.on_clicked(mark_end)


# ============================================================
# Animation
# ============================================================

def advance_frame():
    global current_frame

    if not playing:
        return

    next_frame = current_frame + 1

    if next_frame >= num_frames:
        next_frame = 0

    frame_slider.set_val(next_frame)


timer = fig.canvas.new_timer(
    interval=interval
)

timer.add_callback(advance_frame)


# ============================================================
# Play / pause
# ============================================================

def toggle_play(event):
    global playing

    playing = not playing

    if playing:

        play_button.label.set_text("Pause")
        timer.start()

    else:

        play_button.label.set_text("Play")
        timer.stop()

    fig.canvas.draw_idle()


play_button.on_clicked(toggle_play)


# ============================================================
# Start viewer
# ============================================================

update_frame(0)

print("Opening viewer...")

plt.show()