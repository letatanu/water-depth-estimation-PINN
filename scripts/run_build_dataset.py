from __future__ import annotations
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data_gen.build_pairs import build_pairs_for_run


def _process_one_sww(
    sww_path: Path,
    nx: int,
    ny: int,
    horizon_minutes: float,
    max_pairs: int | None,
):
    """
    Worker function run in a separate process.

    Returns:
        (sww_path, X, Y, err_msg)
        - If success: X, Y are np.ndarrays, err_msg is None
        - If failure: X, Y are None, err_msg is a string
    """
    try:
        X, Y = build_pairs_for_run(
            sww_path,
            nx=nx,
            ny=ny,
            horizon_minutes=horizon_minutes,
            max_pairs=max_pairs,
        )
        return (sww_path, X, Y, None)
    except Exception as e:
        return (sww_path, None, None, f"{type(e).__name__}: {e}")


def main():
    # -------- CONFIG --------
    sims_dir = Path("sims_austin")          # directory with *.sww
    out_npz = Path("dataset/deepflood_anuga_dataset.npz")
    nx, ny = 100, 100
    horizon_minutes = 40.0
    max_pairs_per_run = None    # or e.g. 200

    # how many processes to use (you can tune this)
    max_workers = os.cpu_count() or 4
    # ------------------------

    X_all, Y_all = [], []

    sww_files = sorted(sims_dir.glob("*.sww"))
    if not sww_files:
        raise FileNotFoundError(f"No .sww files found in {sims_dir}")

    print(f"Found {len(sww_files)} sww files")
    print(f"Using up to {max_workers} worker processes")

    # Submit all jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sww = {
            executor.submit(
                _process_one_sww,
                sww,
                nx,
                ny,
                horizon_minutes,
                max_pairs_per_run,
            ): sww
            for sww in sww_files
        }

        for future in as_completed(future_to_sww):
            sww = future_to_sww[future]
            try:
                sww_path, X, Y, err = future.result()
            except Exception as e:
                # This catches errors in the worker setup itself
                print(f"[FATAL worker error] {sww}: {e}")
                continue

            if err is not None:
                print(f"  Skipping {sww_path} due to error: {err}")
                continue

            print(f"Processed {sww_path}, got {X.shape[0]} pairs")
            X_all.append(X)
            Y_all.append(Y)

    if not X_all:
        raise RuntimeError("No data collected from any sww file")

    X_cat = np.concatenate(X_all, axis=0).astype(np.float32)
    Y_cat = np.concatenate(Y_all, axis=0).astype(np.float32)

    # final sanity
    print("Final X shape:", X_cat.shape)
    print("Final Y shape:", Y_cat.shape)
    print("X nan count:", np.isnan(X_cat).sum())
    print("Y nan count:", np.isnan(Y_cat).sum())

    X_cat = np.nan_to_num(X_cat, nan=0.0, posinf=0.0, neginf=0.0)
    Y_cat = np.nan_to_num(Y_cat, nan=0.0, posinf=0.0, neginf=0.0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X_cat, Y=Y_cat)
    print("Saved dataset to:", out_npz)


if __name__ == "__main__":
    main()
