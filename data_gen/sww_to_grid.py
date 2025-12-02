from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.spatial import cKDTree  # NEW

# Optional CUDA path via PyTorch
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _read_var_raw(ds, name: str) -> np.ndarray:
    """
    Read a NetCDF variable and return a clean float32 array.

    - Uses .data to ignore the MaskedArray mask.
    - Replaces _FillValue / missing_value with NaN.
    """
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
    """
    Make time dimension first if needed. Expects either (T, N) or (N, T).
    """
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == T:
        return arr
    if arr.shape[1] == T:
        return arr.T
    raise ValueError(
        f"Cannot align array {name} of shape {arr.shape} with time length {T}"
    )


def _interp_idw_cuda(
    points: np.ndarray,     # (N, 2) float32
    values: np.ndarray,     # (T, N) float32
    XX: np.ndarray,         # (ny, nx)
    YY: np.ndarray,         # (ny, nx)
    k: int = 8,
    chunk_size: int = 4096,
    eps: float = 1e-6,
    device: str = "cuda",
) -> np.ndarray:
    """
    KNN-based inverse-distance weighting interpolation on CUDA.

    Parameters
    ----------
    points : (N, 2)
        Scattered sample locations.
    values : (T, N)
        Values at those locations for each time step.
    XX, YY : (ny, nx)
        Regular grid coordinates.

    Returns
    -------
    out : (T, ny, nx)
        Interpolated values on the grid.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available for CUDA interpolation.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for interpolation.")

    device = torch.device(device)

    # Convert to torch
    pts = torch.from_numpy(points).to(device=device, dtype=torch.float32)   # [N, 2]
    vals = torch.from_numpy(values).to(device=device, dtype=torch.float32)  # [T, N]

    ny, nx = XX.shape
    M = ny * nx

    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)  # [M, 2]
    grid_pts = torch.from_numpy(grid_pts).to(device=device, dtype=torch.float32)

    T, N = vals.shape
    out = torch.empty((T, M), device=device, dtype=torch.float32)

    # Process grid points in chunks to control memory
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        G = grid_pts[start:end]                # [B, 2]
        # Distances: [B, N]
        dists = torch.cdist(G.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)

        # Prevent numerical issues
        dists = torch.clamp(dists, min=eps)

        # k-NN (small k for speed; k <= N)
        k_eff = min(k, N)
        knn_dists, knn_idx = torch.topk(dists, k_eff, largest=False, dim=1)  # [B, k]

        # Weights: inverse distance
        w = 1.0 / (knn_dists + eps)           # [B, k]
        w = w / w.sum(dim=1, keepdim=True)    # normalize

        # Gather values: vals[:, knn_idx] -> [T, B, k]
        vals_knn = vals[:, knn_idx]           # broadcasting gather

        # Weighted sum over k
        interp_chunk = (vals_knn * w.unsqueeze(0)).sum(dim=-1)  # [T, B]

        out[:, start:end] = interp_chunk

    out = out.view(T, ny, nx)
    return out.detach().cpu().numpy().astype(np.float32)


def _interp_knn_cpu(
    points: np.ndarray,     # (N, 2)
    values: np.ndarray,     # (T, N)
    XX: np.ndarray,         # (ny, nx)
    YY: np.ndarray,         # (ny, nx)
    k: int = 4,
    max_points: int = 500_000,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    KNN-based inverse-distance weighting interpolation on CPU.

    Parameters
    ----------
    points : (N, 2)
        Scattered sample locations.
    values : (T, N)
        Values at those locations for each time step.
    XX, YY : (ny, nx)
        Regular grid coordinates.
    k : int
        Number of nearest neighbors to use.
    max_points : int
        If N > max_points, downsample points to this many for interpolation.

    Returns
    -------
    out : (T, ny, nx)
        Interpolated values on the grid.
    """
    N = points.shape[0]
    T, N_val = values.shape
    assert N == N_val, f"points N={N}, values N={N_val}"

    # Optional downsampling for huge meshes
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=np.int64)
        points_sub = points[idx]
        values_sub = values[:, idx]
        print(f"[sww_to_grid] Downsampling points {N} → {max_points} for interpolation")
    else:
        points_sub = points
        values_sub = values

    # Build KD-tree on (possibly downsampled) points
    tree = cKDTree(points_sub)

    ny, nx = XX.shape
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (M, 2)
    M = grid_pts.shape[0]

    # KNN query for all grid points (C-optimized, multi-threaded)
    dists, inds = tree.query(grid_pts, k=k)  # dists, inds: (M, k)

    # Handle exact matches / zero distances
    dists = np.maximum(dists, eps)

    # Inverse-distance weights
    w = 1.0 / dists
    w /= w.sum(axis=1, keepdims=True)  # (M, k)

    # Interpolate for each time step
    out = np.empty((T, M), dtype=np.float32)
    for ti in range(T):
        vals_knn = values_sub[ti, inds]      # (M, k)
        out[ti] = (vals_knn * w).sum(axis=1) # (M,)

    return out.reshape(T, ny, nx)


def load_sww_to_grid(
    sww_path: str | Path,
    nx: int = 256,
    ny: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an ANUGA .sww file and interpolate depth + discharges to a regular grid.

    Returns
    -------
    t : (T,)
        Time vector in seconds.
    H_grid : (T, ny, nx)
        Water depth on regular grid.
    Qx_grid, Qy_grid : (T, ny, nx)
        Discharges on regular grid.
    """
    sww_path = Path(sww_path)

    with Dataset(sww_path) as ds:
        var_names = set(ds.variables.keys())

        # --- time ---
        t = _read_var_raw(ds, "time")  # (T,)

        # --- base vertex arrays ---
        x_vert = _read_var_raw(ds, "x")          # (Nv,)
        y_vert = _read_var_raw(ds, "y")          # (Nv,)
        elev_v = _read_var_raw(ds, "elevation")  # (Nv,)

        stage_v = _read_var_raw(ds, "stage")         # (T, Nv) or (Nv, T)
        xmom_v  = _read_var_raw(ds, "xmomentum")     # (T, Nv) or (Nv, T)
        ymom_v  = _read_var_raw(ds, "ymomentum")     # (T, Nv) or (Nv, T)

        T = t.shape[0]
        stage_v = _ensure_TN(stage_v, T, "stage")
        xmom_v  = _ensure_TN(xmom_v,  T, "xmomentum")
        ymom_v  = _ensure_TN(ymom_v,  T, "ymomentum")

        _, Nv = stage_v.shape

        # --- check for centroid variables ---
        use_centroids = (
            "stage_c" in var_names
            and "elevation_c" in var_names
            and "xmomentum_c" in var_names
            and "ymomentum_c" in var_names
            and "volumes" in var_names
        )

        if use_centroids:
            stage_c = _read_var_raw(ds, "stage_c")        # (T, Nc) or (Nc, T)
            elev_c  = _read_var_raw(ds, "elevation_c")    # (Nc,)
            xmom_c  = _read_var_raw(ds, "xmomentum_c")    # (T, Nc) or (Nc, T)
            ymom_c  = _read_var_raw(ds, "ymomentum_c")    # (T, Nc) or (Nc, T)

            stage_c = _ensure_TN(stage_c, T, "stage_c")
            xmom_c  = _ensure_TN(xmom_c,  T, "xmomentum_c")
            ymom_c  = _ensure_TN(ymom_c,  T, "ymomentum_c")

            _, Nc = stage_c.shape

            vols_var = ds.variables["volumes"]
            vols = vols_var[:].data.astype(np.int64)      # (Nc, 3)

            x_c = x_vert[vols].mean(axis=1)               # (Nc,)
            y_c = y_vert[vols].mean(axis=1)               # (Nc,)

            x_used = x_c
            y_used = y_c
            depth  = np.maximum(stage_c - elev_c[None, :], 0.0)
            xmom   = xmom_c
            ymom   = ymom_c

            print(f"[sww_to_grid] Using centroid variables: Nc={Nc}")
        else:
            x_used = x_vert
            y_used = y_vert
            depth  = np.maximum(stage_v - elev_v[None, :], 0.0)
            xmom   = xmom_v
            ymom   = ymom_v

            print(f"[sww_to_grid] Using vertex variables: Nv={Nv}")

        # clean up NaNs/infs
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        xmom  = np.nan_to_num(xmom,  nan=0.0, posinf=0.0, neginf=0.0)
        ymom  = np.nan_to_num(ymom,  nan=0.0, posinf=0.0, neginf=0.0)

        # --- build regular grid ---
        xmin, xmax = np.nanmin(x_used), np.nanmax(x_used)
        ymin, ymax = np.nanmin(y_used), np.nanmax(y_used)

        xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
        ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")

        points = np.stack([x_used, y_used], axis=1).astype(np.float32)  # (Npts, 2)

        # ------------------------------------------------------------------
        # Interpolation: KNN IDW on CPU (fast, memory-safe for Nc ~ 4M)
        # ------------------------------------------------------------------
        print("[sww_to_grid] Using CPU KNN IDW interpolation (cKDTree)")
        H_grid  = _interp_knn_cpu(points, depth, XX, YY, k=4, max_points=500_000)
        Qx_grid = _interp_knn_cpu(points, xmom,  XX, YY, k=4, max_points=500_000)
        Qy_grid = _interp_knn_cpu(points, ymom,  XX, YY, k=4, max_points=500_000)

        H_grid  = np.nan_to_num(H_grid,  nan=0.0, posinf=0.0, neginf=0.0)
        Qx_grid = np.nan_to_num(Qx_grid, nan=0.0, posinf=0.0, neginf=0.0)
        Qy_grid = np.nan_to_num(Qy_grid, nan=0.0, posinf=0.0, neginf=0.0)

        return t, H_grid, Qx_grid, Qy_grid