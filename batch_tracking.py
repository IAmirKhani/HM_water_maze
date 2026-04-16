import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
import re

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/amir/Desktop/genzel_lab/water_maze_path_dir/paths extracted")
OUTPUT_DIR = Path("/Users/amir/Desktop/genzel_lab/water_maze_path_dir/results")
OUTPUT_DIR.mkdir(exist_ok=True)

GROUP1_RATS = {1, 2, 7, 8, 11, 13, 14}
GROUP2_RATS = {3, 4, 5, 6, 9, 10, 15}

N_INTERP = 500  # number of steps to normalize each path to for averaging


def load_path(filepath):
    """Load an Excel tracking file and return x, y arrays (cleaned)."""
    df = pd.read_excel(filepath)
    # Drop unit-header row
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = ["trial_time", "rec_time", "x", "y"]
    df = df.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(method="linear", limit=5).dropna().reset_index(drop=True)
    return df["x"].values, df["y"].values


def rotate_180(x, y):
    """Rotate path 180° around the arena center (0, 0)."""
    return -x, -y


def normalize_path(x, y, n_steps=N_INTERP):
    """Resample a path to a fixed number of steps via linear interpolation."""
    t_orig = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_steps)
    x_new = np.interp(t_new, t_orig, x)
    y_new = np.interp(t_new, t_orig, y)
    return x_new, y_new


def parse_filename(filename):
    """
    Parse a tracking filename to extract rat number, probe type, and direction.

    Returns (rat_number, probe, direction) or None if not a valid data file.
    probe is 'PT3' or 'PT7', direction is 'SW' or 'NE'.
    """
    name = filename.stem  # e.g. "PT7-1pt3 NE" or "1pt3 SW"

    # Skip Excel temp files
    if name.startswith("~$"):
        return None

    is_pt7 = name.upper().startswith("PT7-")

    # Extract rat number: digits before "pt3"
    match = re.search(r"(\d+)pt3", name, re.IGNORECASE)
    if not match:
        return None
    rat_num = int(match.group(1))

    # Extract direction
    if "SW" in name.upper():
        direction = "SW"
    elif "NE" in name.upper():
        direction = "NE"
    else:
        return None

    probe = "PT7" if is_pt7 else "PT3"
    return rat_num, probe, direction


def load_all_paths(data_dir):
    """
    Load all tracking files. Returns a dict keyed by (rat, probe) with values
    being (x, y) arrays, already rotated if SW.
    """
    paths = {}
    for f in sorted(data_dir.glob("*.xlsx")):
        parsed = parse_filename(f)
        if parsed is None:
            continue
        rat_num, probe, direction = parsed

        x, y = load_path(f)
        if direction == "SW":
            x, y = rotate_180(x, y)

        paths[(rat_num, probe)] = (x, y)
        print(f"  Loaded rat {rat_num:>2} {probe} ({direction}{'→rot' if direction == 'SW' else ''}): {len(x)} samples")

    return paths


def compute_group_average(paths, rat_set, probe):
    """Average normalized paths for a set of rats at a given probe."""
    all_x, all_y = [], []
    for rat in sorted(rat_set):
        key = (rat, probe)
        if key not in paths:
            print(f"  WARNING: missing data for rat {rat} {probe}")
            continue
        x, y = normalize_path(*paths[key])
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return None, None
    return np.mean(all_x, axis=0), np.mean(all_y, axis=0)


def plot_individual_paths(paths, probe, output_dir):
    """Plot all individual paths for a given probe in a grid."""
    all_rats = sorted(set(r for r, p in paths if p == probe))
    n = len(all_rats)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for idx, rat in enumerate(all_rats):
        ax = axes[idx // cols, idx % cols]
        key = (rat, probe)
        if key in paths:
            x, y = paths[key]
            group = "G1" if rat in GROUP1_RATS else "G2"
            color = "tab:blue" if rat in GROUP1_RATS else "tab:orange"
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.8)
            ax.set_title(f"Rat {rat} ({group})")
        ax.set_aspect("equal")
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle(f"Individual Paths – {probe} (after SW rotation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"individual_paths_{probe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved individual_paths_{probe}.png")


def compute_group_heatmap(paths, rat_set, probe, bin_size=1.0, arena_radius=60,
                          sigma=4.0):
    """
    Build an occupancy heatmap by pooling all rat paths in a group.

    Each rat's path is binned into a 2D histogram, then all rats' histograms
    are summed and smoothed with a Gaussian kernel.
    """
    edges = np.arange(-arena_radius, arena_radius + bin_size, bin_size)
    total = np.zeros((len(edges) - 1, len(edges) - 1))

    for rat in sorted(rat_set):
        key = (rat, probe)
        if key not in paths:
            continue
        x, y = paths[key]
        h, _, _ = np.histogram2d(x, y, bins=[edges, edges])
        total += h.T  # transpose so row=y, col=x

    # Smooth
    total = gaussian_filter(total, sigma=sigma)

    # Mask outside the arena circle
    cx = (edges[:-1] + edges[1:]) / 2
    cy = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(cx, cy)
    outside = np.sqrt(xx**2 + yy**2) > arena_radius
    total[outside] = np.nan

    return total, edges


def plot_group_averages(paths, output_dir):
    """Plot occupancy heatmaps for both groups × both probes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for col, probe in enumerate(["PT3", "PT7"]):
        for row, (group_name, rat_set) in enumerate(
            [("Group 1 (1,2,7,8,11,13,14)", GROUP1_RATS),
             ("Group 2 (3,4,5,6,9,10,15)", GROUP2_RATS)]
        ):
            ax = axes[row, col]
            heatmap, edges = compute_group_heatmap(paths, rat_set, probe)

            im = ax.imshow(
                heatmap,
                extent=[edges[0], edges[-1], edges[-1], edges[0]],
                aspect="equal",
                cmap="jet",
                interpolation="bilinear",
                vmin=0, vmax=0.5,
            )
            # Arena circle
            circle = plt.Circle((0, 0), 60, fill=False, color="white", ls="-", lw=1.5)
            ax.add_patch(circle)

            ax.set_xlim(-65, 65)
            ax.set_ylim(-65, 65)
            ax.set_title(f"{group_name} – {probe}", fontsize=11)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Occupancy (pooled samples)")

    fig.suptitle("Group Occupancy", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "group_average_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved group_average_heatmaps.png")


def plot_group_averages_combined(paths, output_dir):
    """Plot occupancy heatmaps averaged across PT3 and PT7 for each group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for idx, (group_name, rat_set) in enumerate(
        [("Group 1 (1,2,7,8,11,13,14)", GROUP1_RATS),
         ("Group 2 (3,4,5,6,9,10,15)", GROUP2_RATS)]
    ):
        ax = axes[idx]
        heatmap_pt3, edges = compute_group_heatmap(paths, rat_set, "PT3")
        heatmap_pt7, _     = compute_group_heatmap(paths, rat_set, "PT7")

        # Average the two probe heatmaps (nanmean to handle masked edges)
        combined = np.nanmean([heatmap_pt3, heatmap_pt7], axis=0)

        im = ax.imshow(
            combined,
            extent=[edges[0], edges[-1], edges[-1], edges[0]],
            aspect="equal",
            cmap="jet",
            interpolation="bilinear",
            vmin=0, vmax=0.5,
        )
        circle = plt.Circle((0, 0), 60, fill=False, color="white", ls="-", lw=1.5)
        ax.add_patch(circle)
        ax.set_xlim(-65, 65)
        ax.set_ylim(-65, 65)
        ax.set_title(f"{group_name} – PT3+PT7 avg", fontsize=11)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Occupancy (pooled samples)")

    fig.suptitle("Group Occupancy – Averaged across PT3 & PT7", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "group_average_heatmaps_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved group_average_heatmaps_combined.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading paths...")
    paths = load_all_paths(DATA_DIR)
    print(f"\nLoaded {len(paths)} paths total.\n")

    print("Plotting individual paths...")
    plot_individual_paths(paths, "PT3", OUTPUT_DIR)
    plot_individual_paths(paths, "PT7", OUTPUT_DIR)

    print("\nComputing and plotting group heatmaps...")
    plot_group_averages(paths, OUTPUT_DIR)

    print("\nComputing and plotting combined PT3+PT7 heatmaps...")
    plot_group_averages_combined(paths, OUTPUT_DIR)

    print("\nDone!")
