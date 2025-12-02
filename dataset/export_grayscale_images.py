# export_grayscale_images.py
import numpy as np
from pathlib import Path
from PIL import Image

# ---------- config ----------
npz_path = "dataset/deepflood_anuga_dataset.npz"
out_root = Path("image_data")
max_samples = 500   # limit for testing; set None for all
# ----------------------------

out_X = out_root / "X"
out_Y = out_root / "Y"
out_X.mkdir(parents=True, exist_ok=True)
out_Y.mkdir(parents=True, exist_ok=True)

data = np.load(npz_path)
X = data["X"]  # [N, 5, H, W]
Y = data["Y"]  # [N, 3, H, W]

N, Cx, H, W = X.shape
Ny, Cy, _, _ = Y.shape
assert N == Ny

if max_samples is None or max_samples > N:
    max_samples = N

channel_names_X = ["depth_t", "qx_t", "qy_t", "inflow", "rain"]
channel_names_Y = ["depth_tplus", "qx_tplus", "qy_tplus"]

def normalize_to_uint8(arr):
    """Min-max normalize a 2D array to 0–255 uint8."""
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    arr_norm = (arr - vmin) / (vmax - vmin)
    arr_uint8 = (arr_norm * 255).clip(0, 255).astype(np.uint8)
    return arr_uint8

for i in range(max_samples):
    x_i = X[i]  # [5, H, W]
    y_i = Y[i]  # [3, H, W]

    # ----- inputs -----
    for c in range(Cx):
        img_arr = normalize_to_uint8(x_i[c])
        img = Image.fromarray(img_arr)
        fname = out_X / f"sample_{i:05d}_{channel_names_X[c]}.png"
        img.save(fname)

    # ----- targets -----
    for c in range(Cy):
        img_arr = normalize_to_uint8(y_i[c])
        img = Image.fromarray(img_arr)
        fname = out_Y / f"sample_{i:05d}_{channel_names_Y[c]}.png"
        img.save(fname)

print(f"Saved grayscale images for {max_samples} samples into {out_root}")
