# data_gen/anuga_simulator.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anuga
import rasterio
from osgeo import gdal


@dataclass
class SimConfig:
    """
    Configuration for a single ANUGA SWE simulation.

    dem_path:      path to the DEM raster used for elevation
    output_dir:    directory where .sww will be written
    run_name:      base name of the .sww file (e.g., sim_0000)

    max_triangle_area:   mesh resolution control (smaller → finer mesh)
    yieldstep:           seconds between outputs
    finaltime:           final simulation time (seconds)
    friction:            Manning coefficient
    """
    dem_path: str | Path
    output_dir: str | Path
    run_name: str

    max_triangle_area: float = 200.0
    yieldstep: float = 300.0
    finaltime: float = 12 * 3600.0
    friction: float = 0.03


def _get_dem_bounds(dem_path: Path):
    """
    Read DEM bounds and shrink them slightly to avoid edge interpolation issues.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        b = src.bounds   # left, bottom, right, top
    
    # --- ADD THIS BUFFER ---
    # Shrink the domain by 1.0 meter to ensure all vertices are inside valid data
    buffer = 0
    left = b.left + buffer
    bottom = b.bottom + buffer
    right = b.right - buffer
    top = b.top - buffer
    
    return left, bottom, right, top

import os
import time
import uuid

def build_domain_from_dem(cfg: SimConfig, mesh_prefix=None) -> anuga.Domain:
    if mesh_prefix is None:
        mesh_prefix = f"mesh_{os.getpid()}_{int(time.time()*1000)}_{uuid.uuid4().hex}"
    mesh_filename = mesh_prefix + ".msh"

    dem_path = Path(cfg.dem_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left, bottom, right, top = _get_dem_bounds(dem_path)

    boundary_polygon = [
        (left,  bottom),
        (right, bottom),
        (right, top),
        (left,  top),
    ]

    boundary_tags = {
        "left":   [0],
        "top":    [1],
        "right":  [2],
        "bottom": [3],
    }

    domain = anuga.create_domain_from_regions(
        bounding_polygon=boundary_polygon,
        boundary_tags=boundary_tags,
        maximum_triangle_area=cfg.max_triangle_area,
        mesh_filename=mesh_filename,
        use_cache=False,
    )

    # --- Elevation from DEM ---
    domain.set_quantity(
        "elevation",
        filename=str(dem_path),
        use_cache=False,
        location="centroids",
        verbose=True,
    )

    # Clean NaNs at centroids, then push back through ANUGA's machinery
    elev = domain.quantities["elevation"].centroid_values.copy()
    nan_mask = np.isnan(elev)
    print("NaN in elevation before fix:", nan_mask.sum())

    if nan_mask.any():
        mean_elev = np.nanmean(elev)
        elev[nan_mask] = mean_elev
        # IMPORTANT: use set_quantity again with the numeric array
        # so ANUGA updates vertices/edges as well.
        domain.set_quantity("elevation", elev, location="centroids")
        elev = domain.quantities["elevation"].centroid_values
        print("NaN in elevation after fix:", np.isnan(elev).sum())

    # Friction
    domain.set_quantity("friction", cfg.friction)

    # Initial stage = elevation (dry bed, zero depth)
    domain.set_quantity("elevation", elev, location="centroids")
    domain.set_quantity("stage", expression="elevation", location="centroids")

    # Name + IO + numerics
    domain.set_name(cfg.run_name)
    domain.set_datadir(str(out_dir.resolve()))
    domain.set_minimum_storable_height(0.01)

    try:
        if os.path.exists(mesh_filename):
            os.remove(mesh_filename)
    except OSError:
        pass

    return domain

def set_simple_boundaries(domain: anuga.Domain):
    """
    Attach simple reflective or Dirichlet boundaries on all 4 sides.
    This matches the 'left', 'right', 'top', 'bottom' tags used in
    build_domain_from_dem().
    """
    # Option A: reflective (no inflow/outflow across boundary)
    Br = anuga.Reflective_boundary(domain)

    # If you want still-water Dirichlet instead, uncomment this and replace:
    # Bd = anuga.Dirichlet_boundary([0.0, 0.0, 0.0])  # [stage, xmom, ymom]

    domain.set_boundary({
        "left":   Br,
        "right":  Br,
        "top":    Br,
        "bottom": Br,
    })
    
def apply_rainfall(domain: anuga.Domain,
                   rain_rate_func: Callable[[float], float],
                   factor: float = 1.0):
    """
    Attach spatially uniform rainfall via ANUGA Rate_operator.

    rain_rate_func(t) returns intensity in m/s (see UniformHyetograph).
    """

    # ANUGA may call rate(t) or rate(t, x, y), so accept both
    def rate_fn(t, *args):
        # t can be scalar or array-like; just use the first value
        try:
            t_scalar = float(t[0])  # if array-like
        except (TypeError, IndexError):
            t_scalar = float(t)
        return float(rain_rate_func(t_scalar) * factor)

    op = anuga.Rate_operator(domain, rate=rate_fn, factor=1.0)
    # It will register itself in domain.forcing_terms
    return op



import numpy as np  # make sure this is imported at top of file
def run_simulation(cfg: SimConfig, rain_rate_func):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    domain = build_domain_from_dem(cfg)
    set_simple_boundaries(domain)
    apply_rainfall(domain, rain_rate_func)  
    print(
        f"[{cfg.run_name}] Starting evolve (manual rainfall): "
        f"yieldstep={cfg.yieldstep}, finaltime={cfg.finaltime}"
    )

    for t in domain.evolve(yieldstep=cfg.yieldstep, finaltime=cfg.finaltime):
        rain = float(rain_rate_func(float(t)))  # m/s (for logging only)

        # Use ANUGA's height quantity instead of manual stage - elev
        depth_c = domain.quantities["height"].centroid_values
        elev_c  = domain.quantities["elevation"].centroid_values

        dmin = float(depth_c.min())
        dmax = float(depth_c.max())
        emin = float(elev_c.min())
        emax = float(elev_c.max())

        print(
            f"[{cfg.run_name}] t={t:7.1f}s  "
            f"rain={rain:.3e} m/s  "
            f"depth[min,max]=({dmin:.3e}, {dmax:.3e}) "
            f"elev[min,max]=({emin:.3e}, {emax:.3e})"
        )

    sww_path = cfg.output_dir / f"{cfg.run_name}.sww"
    return sww_path