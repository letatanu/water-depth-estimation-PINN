# scripts/debug_depth_values.py
from pathlib import Path
import numpy as np
from data_gen.sww_to_grid import load_sww_to_grid

def main():
    sww_path = "sims_austin/sim_0032.sww"  # adjust if needed
    t, H, Qx, Qy = load_sww_to_grid(sww_path, nx=128, ny=128)

    print(f"t shape: {t.shape}, H shape: {H.shape}")  # (T,), (T, ny, nx)

    print("GLOBAL depth stats over all times:")
    print("  min:", float(H.min()), "max:", float(H.max()))

    T = H.shape[0]
    for idx in range(0, T, max(1, T // 10)):
        h = H[idx]
        print(
            f"t={t[idx]:8.1f}s  "
            f"min={float(h.min()):.6f}, max={float(h.max()):.6f}, "
            f"mean={float(h.mean()):.6f}"
        )

if __name__ == "__main__":
    main()
