import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def analyze_tracking(filepath, spatial_bin_cm=2.0, speed_smooth_ms=400.0):
    """
    Analyze animal tracking data and produce 3 plots:
      1) Raw trajectory trace
      2) Occupancy map (time spent per spatial bin)
      3) Speed map (mean speed per spatial bin)

    Parameters
    ----------
    filepath : str
        Path to an Excel file with 3 columns: time (s), X (cm), Y (cm).
        First data row may contain unit labels (skipped automatically).
    spatial_bin_cm : float
        Spatial bin size in cm for occupancy and speed maps. Default: 2 cm.
    speed_smooth_ms : float
        Gaussian smoothing window (FWHM) for instantaneous speed in ms.
        Default: 400 ms.

    Returns
    -------
    fig : matplotlib.figure.Figure
    results : dict
        Contains 'time', 'x', 'y', 'speed', 'occupancy_map', 'speed_map',
        'x_edges', 'y_edges'.
    """

    # ── Load and clean data ──────────────────────────────────────────────
    df = pd.read_excel(filepath)
    # Drop unit-header row if present (non-numeric first entry in col 0)
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = ["record", "time", "x", "y"]
    df = df.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")

    # Interpolate short gaps, then drop any remaining NaNs at edges
    df = df.interpolate(method="linear", limit=5)
    df = df.dropna().reset_index(drop=True)

    t = df["time"].values
    x = df["x"].values
    y = df["y"].values

    # ── Compute instantaneous speed ──────────────────────────────────────
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    inst_speed = np.sqrt(dx**2 + dy**2) / dt          # cm/s
    # Pad to same length as position (assign speed to leading sample)
    inst_speed = np.append(inst_speed, inst_speed[-1])

    # Gaussian smoothing: convert FWHM in ms → sigma in samples
    median_dt = np.median(dt)                          # s per sample
    fwhm_samples = (speed_smooth_ms / 1000.0) / median_dt
    sigma_samples = fwhm_samples / 2.355               # FWHM = 2.355 * sigma
    speed_smooth = gaussian_filter(inst_speed, sigma=sigma_samples)

    # ── Spatial bins ─────────────────────────────────────────────────────
    x_min, x_max = np.floor(x.min()), np.ceil(x.max())
    y_min, y_max = np.floor(y.min()), np.ceil(y.max())
    x_edges = np.arange(x_min, x_max + spatial_bin_cm, spatial_bin_cm)
    y_edges = np.arange(y_min, y_max + spatial_bin_cm, spatial_bin_cm)

    # Bin indices for each sample
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1
    x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
    y_idx = np.clip(y_idx, 0, len(y_edges) - 2)

    n_xbins = len(x_edges) - 1
    n_ybins = len(y_edges) - 1

    # ── Occupancy map (seconds per bin) ──────────────────────────────────
    occupancy = np.zeros((n_ybins, n_xbins))
    for i in range(len(t)):
        dt_i = median_dt if i == 0 else t[i] - t[i - 1]
        occupancy[y_idx[i], x_idx[i]] += dt_i

    # ── Speed map (mean smoothed speed per bin) ──────────────────────────
    speed_sum = np.zeros((n_ybins, n_xbins))
    speed_cnt = np.zeros((n_ybins, n_xbins))
    for i in range(len(t)):
        speed_sum[y_idx[i], x_idx[i]] += speed_smooth[i]
        speed_cnt[y_idx[i], x_idx[i]] += 1

    with np.errstate(invalid="ignore"):
        speed_map = speed_sum / speed_cnt
    speed_map[speed_cnt == 0] = np.nan

    # Mask unvisited bins in occupancy for display
    occupancy_display = occupancy.copy()
    occupancy_display[occupancy_display == 0] = np.nan

    # ── Plotting ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1) Raw trace
    ax = axes[0]
    ax.plot(x, y, color="0.3", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Raw Trajectory")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # 2) Occupancy map
    ax = axes[1]
    im1 = ax.imshow(
        occupancy_display,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
        aspect="equal",
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(f"Occupancy Map ({spatial_bin_cm} cm bins)")
    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Time (s)")

    # 3) Speed map
    ax = axes[2]
    im2 = ax.imshow(
        speed_map,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
        aspect="equal",
        cmap="jet",
        interpolation="nearest",
    )
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(f"Speed Map ({int(speed_smooth_ms)} ms smooth)")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Speed (cm/s)")

    fig.tight_layout()

    results = {
        "time": t,
        "x": x,
        "y": y,
        "speed": speed_smooth,
        "occupancy_map": occupancy,
        "speed_map": speed_map,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }
    return fig, results


# ── Run on the provided file ────────────────────────────────────────────
if __name__ == "__main__":
    fig, res = analyze_tracking(
        "/Users/amir/Desktop/genzel_lab/water_maze_path_dir/paths extracted/1pt3 SW.xlsx",
        spatial_bin_cm=2.0,
        speed_smooth_ms=400.0,
    )
    fig.savefig("/Users/amir/Desktop/genzel_lab/water_maze_path_dir/results/tracking_analysis.png", dpi=150, bbox_inches="tight")
    print("Done. Speed range: {:.1f} – {:.1f} cm/s".format(
        np.nanmin(res["speed"]), np.nanmax(res["speed"])
    ))
    print(f"Grid size: {res['occupancy_map'].shape}")
    plt.close(fig)