import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

data = np.load("dataset/deepflood_anuga_dataset.npz")
X, Y = data["X"], data["Y"]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
writer = FFMpegWriter(fps=2)   # 2 frames per second

with writer.saving(fig, "visualization/sample_sequence.mp4", dpi=150):

    for i in range(min(200, len(X))):  # first 200 frames
        depth_now = X[i, 0]
        depth_future = Y[i, 0]
        vmax = max(depth_now.max(), depth_future.max())

        axes[0].clear()
        axes[1].clear()

        im0 = axes[0].imshow(depth_now, origin="lower", vmin=0, vmax=vmax)
        axes[0].set_title("Depth(t)")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        im1 = axes[1].imshow(depth_future, origin="lower", vmin=0, vmax=vmax)
        axes[1].set_title("Depth(t + 30 min)")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        plt.tight_layout()
        writer.grab_frame()

print("Video saved to visualization/sample_sequence.mp4")
