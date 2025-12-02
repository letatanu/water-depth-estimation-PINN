# scripts/debug_sww_depth_raw.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

sww_path = Path("sims_austin/sim_0000.sww")  # change if needed
out_dir = Path("debug_sww_raw")
out_dir.mkdir(parents=True, exist_ok=True)

with Dataset(sww_path) as ds:
    print("Variables and shapes:")
    for name, var in ds.variables.items():
        print(f"  {name}: {var.shape}")

    # ---- 1. pick stage-like var (2D: [T, N]) ----
    stage_candidates = [name for name, v in ds.variables.items() if ds.variables[name].ndim == 2]
    if not stage_candidates:
        raise RuntimeError("No 2D stage-like variable found")

    # Prefer names containing 'stage'
    stage_name = None
    for cand in stage_candidates:
        if "stage" in cand:
            stage_name = cand
            break
    if stage_name is None:
        stage_name = stage_candidates[0]

    v_stage = ds.variables[stage_name]
    stage_np = np.array(v_stage[:].data, dtype=np.float32)  # [T, Nc]
    T, Nc = stage_np.shape
    print(f"Selected stage var: {stage_name}, shape={stage_np.shape}")

    # ---- 2. pick elevation-like var with matching N ----
    elev_candidates = []
    for name, v in ds.variables.items():
        if "elevation" in name and v.ndim in (1, 2):
            elev_candidates.append(name)
    if not elev_candidates:
        raise RuntimeError("No elevation-like variable found")

    elev_name = None
    for cand in elev_candidates:
        v = ds.variables[cand]
        if v.ndim == 1 and v.shape[0] == Nc:
            elev_name = cand
            break
        if v.ndim == 2 and v.shape[1] == Nc:
            elev_name = cand
            break
    if elev_name is None:
        # fallback: first candidate
        elev_name = elev_candidates[0]

    v_elev = ds.variables[elev_name]
    elev_raw = v_elev[:]

    if v_elev.ndim == 1:
        elev_np = np.array(elev_raw.data, dtype=np.float32)  # [Nc]
    else:
        elev_np = np.array(elev_raw.data[0], dtype=np.float32)  # [Nc] from first time slice

    print(f"Selected elevation var: {elev_name}, shape={v_elev.shape}")

    # ---- 3. pick x,y vars with length N = Nc ----
    def pick_coord(name_hint):
        candidates = []
        for name, v in ds.variables.items():
            if name_hint in name and v.ndim == 1 and v.shape[0] == Nc:
                candidates.append(name)
        if candidates:
            return candidates[0]
        # fallback: any 1D variable with length Nc
        for name, v in ds.variables.items():
            if v.ndim == 1 and v.shape[0] == Nc:
                return name
        raise RuntimeError(f"No {name_hint}-like coordinate with length {Nc} found")

    x_name = pick_coord("x")
    y_name = pick_coord("y")
    v_x = ds.variables[x_name]
    v_y = ds.variables[y_name]
    x = np.array(v_x[:].data, dtype=np.float32)
    y = np.array(v_y[:].data, dtype=np.float32)
    print(f"Selected x var: {x_name}, shape={x.shape}")
    print(f"Selected y var: {y_name}, shape={y.shape}")

    # ---- 4. handle fill/missing values ----
    def apply_fill(arr, var):
        arr = np.array(arr, dtype=np.float32)
        fv = getattr(var, "_FillValue", None)
        mv = getattr(var, "missing_value", None)
        if fv is not None:
            arr[np.isclose(arr, fv)] = np.nan
        if mv is not None:
            arr[np.isclose(arr, mv)] = np.nan
        return arr

    stage_np = apply_fill(stage_np, v_stage)
    elev_np = apply_fill(elev_np, v_elev)

    print("stage_np NaN count:", np.isnan(stage_np).sum(), "/", stage_np.size)
    print("elev_np  NaN count:", np.isnan(elev_np).sum(), "/", elev_np.size)

    depth = stage_np - elev_np[None, :]  # [T, Nc]
    depth = np.maximum(depth, 0.0)

    print("depth NaN count:", np.isnan(depth).sum(), "/", depth.size)
    print("depth min/max (ignoring NaN):",
          np.nanmin(depth), np.nanmax(depth))

    t = ds.variables["time"][:]
    print("T, N:", depth.shape)
    print("time range:", float(t.min()), "->", float(t.max()))

# ---- 5. scatter plot at ~900s ----
t_target = 900.0
idx = int(np.argmin(np.abs(t - t_target)))
d = depth[idx]

print(f"t[{idx}] = {float(t[idx])} s, depth min/max:",
      float(np.nanmin(d)), float(np.nanmax(d)))

plt.figure(figsize=(6, 5))
sc = plt.scatter(x, y, c=d, s=4, cmap="Blues")
plt.colorbar(sc, label="Depth (m)")
plt.title(f"Raw depth t={float(t[idx]):.0f}s (N={Nc})")
plt.axis("equal")
plt.savefig(out_dir / f"raw_depth_t{int(t[idx]):04d}.png",
            dpi=150, bbox_inches="tight")
plt.close()
