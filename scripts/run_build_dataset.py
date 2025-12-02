# scripts/run_build_dataset.py
from __future__ import annotations
from pathlib import Path

import numpy as np

from data_gen.build_pairs import build_pairs_for_run


def main():
    # -------- CONFIG --------
    sims_dir = Path("sims_austin")          # directory with *.sww
    out_npz = Path("dataset/deepflood_anuga_dataset.npz")
    nx, ny = 100, 100
    horizon_minutes = 30.0
    max_pairs_per_run = None    # or e.g. 200
    # ------------------------

    X_all, Y_all = [], []

    sww_files = sorted(sims_dir.glob("*.sww"))
    if not sww_files:
        raise FileNotFoundError(f"No .sww files found in {sims_dir}")

    print(f"Found {len(sww_files)} sww files")

    for sww in sww_files:
        print(f"Processing {sww} ...")
        try:
            X, Y = build_pairs_for_run(
                sww,
                nx=nx,
                ny=ny,
                horizon_minutes=horizon_minutes,
                max_pairs=max_pairs_per_run,
            )
        except Exception as e:
            print(f"  Skipping {sww} due to error: {e}")
            continue

        X_all.append(X)
        Y_all.append(Y)
        print(f"  added {X.shape[0]} pairs")

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
