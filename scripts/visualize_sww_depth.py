#!/usr/bin/env python
"""
Visualize water surface elevation (DEM + depth) from an ANUGA .sww file
at a single time step.

Example:
    python -m scripts.visualize_sww_depth \
        --sww sims_austin/sim_0000.sww \
        --nx 256 --ny 256 \
        --time-sec 3600 \
        --out depth_images_sww/sim_0000_t3600.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def _read_var_raw(ds, name: str) -> np.ndarray:
    v = ds.variables[name]
    arr = np.array(v[:].data, dtype=np.float32)

    fill_vals = []
    fv = getattr(v, "_FillValue", None)
    if fv is not None:
        fill_vals.append(fv)
    mv = getattr(v, "missing_value", None)
    if mv is not None:
        fill_vals.append(mv)

    for f in fill_vals:
        arr[np.isclose(arr, f)] = np.nan

    return arr


def _ensure_TN(arr: np.ndarray, T: int, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == T:
        return arr
    if arr.shape[1] == T:
        return arr.T
    raise ValueError(
        f"Cannot align array {name} of shape {arr.shape} with time length {T}"
    )


def _interp_knn_cpu(
    points: np.ndarray,
    values: np.ndarray,
    XX: np.ndarray,
    YY: np.ndarray,
    k: int = 4,
    max_points: int = 500_000,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    KNN-based inverse-distance weighting interpolation on CPU.

    points: (N, 2)
    values: (T, N)
    XX, YY: (ny, nx)
    returns: (T, ny, nx)
    """
    N = points.shape[0]
    T, N_val = values.shape
    assert N == N_val, f"points N={N}, values N={N_val}"

    # Optional downsampling for huge meshes
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=np.int64)
        points_sub = points[idx]
        values_sub = values[:, idx]
        print(
            f"[visualize_sww_depth] Downsampling points "
            f"{N} -> {max_points} for interpolation"
        )
    else:
        points_sub = points
        values_sub = values

    tree = cKDTree(points_sub)

    ny, nx = XX.shape
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (M, 2)
    M = grid_pts.shape[0]

    dists, inds = tree.query(grid_pts, k=k)
    dists = np.maximum(dists, eps)

    w = 1.0 / dists
    w /= w.sum(axis=1, keepdims=True)  # (M, k)

    out = np.empty((T, M), dtype=np.float32)
    for ti in range(T):
        vals_knn = values_sub[ti, inds]      # (M, k)
        out[ti] = (vals_knn * w).sum(axis=1) # (M,)

    return out.reshape(T, ny, nx)


def load_sww_single_time_to_grid(
    sww_path: Path,
    nx: int,
    ny: int,
    time_index: int | None = None,
    time_sec: float | None = None,
):
    """
    Load a single time slice from .sww and interpolate:
    - depth  = max(stage - elevation, 0)
    - elevation
    to a regular grid.

    Returns (t_value, depth_grid[ny, nx], elev_grid[ny, nx])
    """
    sww_path = Path(sww_path)
    with Dataset(sww_path) as ds:
        var_names = set(ds.variables.keys())

        # --- time vector ---
        t = _read_var_raw(ds, "time")
        T = t.shape[0]

        if time_index is not None and time_sec is not None:
            raise ValueError("Specify only one of --time-idx or --time-sec")

        if time_index is not None:
            idx = int(np.clip(time_index, 0, T - 1))
        elif time_sec is not None:
            idx = int(np.argmin(np.abs(t - time_sec)))
        else:
            idx = 0  # default: first frame

        t_val = float(t[idx])
        print(f"[visualize_sww_depth] Selected time index {idx} (t={t_val:.2f}s)")

        # --- geometry ---
        x_vert = _read_var_raw(ds, "x")
        y_vert = _read_var_raw(ds, "y")
        elev_v = _read_var_raw(ds, "elevation")

        # --- stage ---
        stage_v = _read_var_raw(ds, "stage")
        stage_v = _ensure_TN(stage_v, T, "stage")

        # --- centroids vs vertices ---
        use_centroids = (
            "stage_c" in var_names
            and "elevation_c" in var_names
            and "volumes" in var_names
        )

        if use_centroids:
            stage_c = _read_var_raw(ds, "stage_c")
            elev_c  = _read_var_raw(ds, "elevation_c")
            stage_c = _ensure_TN(stage_c, T, "stage_c")

            vols = ds.variables["volumes"][:].data.astype(np.int64)  # (Nc, 3)
            x_c = x_vert[vols].mean(axis=1)
            y_c = y_vert[vols].mean(axis=1)

            x_used = x_c
            y_used = y_c

            depth_vals = np.maximum(stage_c[idx] - elev_c, 0.0)
            elev_vals  = elev_c.astype(np.float32)

            print(f"[visualize_sww_depth] Using centroid vars: Nc={x_c.shape[0]}")
        else:
            x_used = x_vert
            y_used = y_vert

            depth_vals = np.maximum(stage_v[idx] - elev_v, 0.0)
            elev_vals  = elev_v.astype(np.float32)

            print(f"[visualize_sww_depth] Using vertex vars: Nv={x_vert.shape[0]}")

        depth_vals = np.nan_to_num(depth_vals, nan=0.0, posinf=0.0, neginf=0.0)
        elev_vals  = np.nan_to_num(elev_vals,  nan=0.0, posinf=0.0, neginf=0.0)

        # --- regular grid in x,y ---
        xmin, xmax = np.nanmin(x_used), np.nanmax(x_used)
        ymin, ymax = np.nanmin(y_used), np.nanmax(y_used)

        xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
        ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")

        # --- interpolate to regular grid ---
        points = np.stack([x_used, y_used], axis=1).astype(np.float32)  # (N, 2)

        depth_grid = _interp_knn_cpu(
            points,
            depth_vals[None, :],  # (1, N)
            XX,
            YY,
            k=4,
            max_points=500_000,
        )[0]  # (ny, nx)

        elev_grid = _interp_knn_cpu(
            points,
            elev_vals[None, :],   # (1, N)
            XX,
            YY,
            k=4,
            max_points=500_000,
        )[0]  # (ny, nx)

        depth_grid = np.nan_to_num(depth_grid, nan=0.0, posinf=0.0, neginf=0.0)
        elev_grid  = np.nan_to_num(elev_grid,  nan=0.0, posinf=0.0, neginf=0.0)

        return t_val, depth_grid, elev_grid


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize water surface elevation (DEM + depth) from .sww"
    )
    p.add_argument(
        "--sww",
        type=str,
        default="sims_austin/sim_0000.sww",
        help="Path to .sww file (relative to project root)",
    )
    p.add_argument(
        "--nx",
        type=int,
        default=256,
        help="Grid size in x (columns) for interpolation",
    )
    p.add_argument(
        "--ny",
        type=int,
        default=256,
        help="Grid size in y (rows) for interpolation",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output PNG path (relative to project root). "
            "If omitted, saves beside the .sww as "
            "sim_XXXX_surface_tXXXXXX.png."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved PNG",
    )
    p.add_argument(
        "--time-idx",
        type=int,
        default=None,
        help="Time index to visualize (0-based).",
    )
    p.add_argument(
        "--time-sec",
        type=float,
        default=None,
        help="Target time in seconds (nearest index will be used).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    project_root = ROOT
    sww_path = (project_root / args.sww).resolve()

    if not sww_path.is_file():
        raise FileNotFoundError(f"SWW file not found: {sww_path}")

    t_val, depth_grid, elev_grid = load_sww_single_time_to_grid(
        sww_path,
        nx=args.nx,
        ny=args.ny,
        time_index=args.time_idx,
        time_sec=args.time_sec,
    )

    # Water surface elevation = DEM + depth
    surface_grid = elev_grid + depth_grid

    if args.out is not None:
        out_path = (project_root / args.out).resolve()
    else:
        out_path = (
            sww_path.with_suffix("").parent
            / f"{sww_path.stem}_surface_t{int(t_val):06d}.png"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[visualize_sww_depth] Saving water surface image to {out_path}")
    cmap = "terrain"

    # Option 2: same as your DEM hillshade (grayscale)
    # cmap = "gray"

    plt.figure(figsize=(5, 4))
    im = plt.imshow(depth_grid*1000, origin="lower", cmap=cmap)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Depth (mm)")
    plt.title(f"Depth t={t_val:.1f}s")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # plt.figure(figsize=(5, 4))
    # im = plt.imshow(surface_grid, origin="lower")
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # plt.title(f"Water surface elevation (t={t_val:.1f}s)")
    # plt.tight_layout()
    # plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    # plt.close()

    print("[visualize_sww_depth] Done.")


if __name__ == "__main__":
    main()
