import rasterio
import numpy as np
from pathlib import Path

in_path = Path("DEM/austin.asc")       # <-- your original DEM
out_path = Path("DEM/austin_filled.asc")

with rasterio.open(in_path) as src:
    profile = src.profile.copy()
    nodata = src.nodata

    # Read first band as masked array (nodata -> mask)
    data = src.read(1, masked=True)

    if nodata is None:
        print("Warning: DEM has no nodata value set; assuming values == 0 are valid.")
    else:
        print(f"Input nodata value: {nodata}")

    # Replace masked (nodata) cells with 0
    filled = data.filled(0.0).astype(profile["dtype"])

    # Update metadata: declare nodata = 0
    profile.update(nodata=0)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filled, 1)

print(f"Saved filled DEM to: {out_path}")
