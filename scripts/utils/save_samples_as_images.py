import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = np.load("dataset/deepflood_anuga_dataset.npz")
X, Y = data["X"], data["Y"]

out_dir = Path("visualization/images")
out_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(100, len(X))):  # save first 100 samples
    depth_now = X[i, 0]
    depth_future = Y[i, 0]
    vmax = max(depth_now.max(), depth_future.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(depth_now, origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title("Depth(t)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(depth_future, origin="lower", vmin=0, vmax=vmax)
    axes[1].set_title("Depth(t+30 min)")
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    filepath = out_dir / f"sample_{i:04d}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

print("Saved images to:", out_dir)
