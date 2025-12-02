import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pathlib import Path
import rasterio

# ---------- CONFIG ----------
dem_path = Path("DEM/austin.dem")   # adjust path if needed
out_dir = Path("dem_images_clean")
dpi = 150
# -----------------------------

out_dir.mkdir(parents=True, exist_ok=True)

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)   # first band
    meta = src.meta

print("Raw DEM shape:", dem.shape)
print("Raw min/max:", dem.min(), dem.max())

# 1) Mask nodata values (here: -9999 and rasterio's nodata if present)
nodata_vals = [-9999.0]
with rasterio.open(dem_path) as src:
    if src.nodata is not None:
        nodata_vals.append(src.nodata)

mask = np.zeros_like(dem, dtype=bool)
for nd in nodata_vals:
    mask |= np.isclose(dem, nd)

dem_masked = np.where(mask, np.nan, dem)

print("After masking:")
print("  min:", np.nanmin(dem_masked), "max:", np.nanmax(dem_masked))
print("  NaN count:", np.isnan(dem_masked).sum())

# 2) Plain elevation image (terrain colormap)
plt.figure(figsize=(6, 5))
plt.imshow(dem_masked, origin="lower", cmap="terrain")
plt.axis("off")
plt.colorbar(label="Elevation (m)")
plt.title("DEM Elevation")
plt.savefig(out_dir / "dem_elevation_clean.png", dpi=dpi, bbox_inches="tight")
plt.close()

# 3) Hillshade-style grayscale (closer to your screenshot)
ls = LightSource(azdeg=315, altdeg=45)
# Replace NaNs with mean elevation for hillshade computation
dem_for_hs = np.where(np.isnan(dem_masked), np.nanmean(dem_masked), dem_masked)
hs = ls.hillshade(dem_for_hs, vert_exag=1.0, dx=1.0, dy=1.0)

plt.figure(figsize=(6, 5))
plt.imshow(hs, origin="lower", cmap="gray")
plt.title("DEM Hillshade")
plt.axis("off")
plt.savefig(out_dir / "dem_hillshade.png", dpi=dpi, bbox_inches="tight")
plt.close()

print("Saved DEM images to:", out_dir)
