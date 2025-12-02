# data_gen/build_pairs.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np

from data_gen.sww_to_grid import load_sww_to_grid


def build_pairs_for_run(
    sww_path: Path,
    nx: int = 100,
    ny: int = 100,
    horizon_minutes: float = 30.0,
    max_pairs: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build DeepFlood-style (X,Y) pairs from a single .sww run.

    X: [N, 5, ny, nx]
       0: depth(t)
       1: qx(t)
       2: qy(t)
       3: inflow_avg (here: 0)
       4: rain_avg   (here: 0)

    Y: [N, 3, ny, nx]
       0: depth(t+Δ)
       1: qx(t+Δ)
       2: qy(t+Δ)
    """
    sww_path = Path(sww_path)
    t, H, Qx, Qy = load_sww_to_grid(sww_path, nx=nx, ny=ny)
    T = len(t)

    # Ensure clean inputs
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    Qx = np.nan_to_num(Qx, nan=0.0, posinf=0.0, neginf=0.0)
    Qy = np.nan_to_num(Qy, nan=0.0, posinf=0.0, neginf=0.0)

    horizon_sec = horizon_minutes * 60.0

    X_list, Y_list = [], []

    # for each t_k find index j where t_j ≈ t_k + horizon
    for k in range(T):
        t_target = t[k] + horizon_sec
        j_candidates = np.where(np.abs(t - t_target) < 1e-6)[0]
        if len(j_candidates) == 0:
            # allow small tolerance (e.g. 0.5*output_dt)
            j_candidates = np.where(np.abs(t - t_target) < 600.0)[0]
        if len(j_candidates) == 0:
            continue
        j = int(j_candidates[0])
        if j <= k:
            continue

        depth_t = H[k]
        qx_t = Qx[k]
        qy_t = Qy[k]
        depth_tp = H[j]
        qx_tp = Qx[j]
        qy_tp = Qy[j]

        # sanity: clip negatives (shouldn't exist, but just in case)
        depth_t = np.clip(depth_t, 0.0, None)
        depth_tp = np.clip(depth_tp, 0.0, None)

        # inflow/rain avg placeholders (zeros)
        inflow_avg = np.zeros_like(depth_t, dtype=np.float32)
        rain_avg = np.zeros_like(depth_t, dtype=np.float32)

        X_k = np.stack(
            [depth_t, qx_t, qy_t, inflow_avg, rain_avg],
            axis=0,
        ).astype(np.float32)

        Y_k = np.stack(
            [depth_tp, qx_tp, qy_tp],
            axis=0,
        ).astype(np.float32)

        X_list.append(X_k)
        Y_list.append(Y_k)

        if max_pairs is not None and len(X_list) >= max_pairs:
            break

    if not X_list:
        raise RuntimeError(f"No valid time pairs found in {sww_path}")

    X = np.stack(X_list, axis=0)  # [N,5,ny,nx]
    Y = np.stack(Y_list, axis=0)  # [N,3,ny,nx]

    # final nan cleanup (paranoia)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, Y
