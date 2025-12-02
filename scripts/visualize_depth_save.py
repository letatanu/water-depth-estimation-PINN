import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Config --------
npz_path = "dataset/deepflood_anuga_dataset.npz"
out_dir = Path("depth_images_npz")
num_samples = 20          # how many samples to save
dpi = 150
# ------------------------

out_dir.mkdir(parents=True, exist_ok=True)

# Load data
data = np.load(npz_path)
X = data["X"]   # [N, 5, H, W]
Y = data["Y"]   # [N, 3, H, W]

for i in range(min(num_samples, X.shape[0])):
    depth_now = X[i, 0]      # depth at t
    depth_future = Y[i, 0]   # depth at t+30

    vmax = max(depth_now.max(), depth_future.max())

    # ---- Save t ----
    plt.figure(figsize=(4,4))
    plt.imshow(depth_now, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar()
    plt.title(f"Depth_t sample {i}")
    plt.savefig(out_dir / f"depth_t_{i:04d}.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # ---- Save t+30 ----
    plt.figure(figsize=(4,4))
    plt.imshow(depth_future, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar()
    plt.title(f"Depth_tplus30 sample {i}")
    plt.savefig(out_dir / f"depth_tplus30_{i:04d}.png", dpi=dpi, bbox_inches="tight")
    plt.close()

print("Saved depth images to:", out_dir)
