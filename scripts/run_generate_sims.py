# scripts/run_generate_sims.py
from pathlib import Path
import yaml
import numpy as np

from data_gen.anuga_simulator import SimConfig, run_simulation
from data_gen.rainfall_scenarios import make_uniform_rain

from multiprocessing import Pool, cpu_count


def _run_single_sim(args):
    """Worker for a single simulation (runs in its own process)."""
    sim_cfg, r_peak = args

    # local import is fine; avoids any weirdness if you move things later
    from data_gen.rainfall_scenarios import make_uniform_rain
    from data_gen.anuga_simulator import run_simulation

    run_name = sim_cfg.run_name
    rain_func = make_uniform_rain(r_peak_mm_hr=r_peak)  # t -> rain_rate

    print(f"[{run_name}] Running with r_peak={r_peak:.1f} mm/hr")
    sww_path = run_simulation(sim_cfg, rain_func)
    print(f"    → Finished. Output: {sww_path}\n")
    return str(sww_path)



def main(cfg_path: str):
    # -----------------------------
    # Load YAML config
    # -----------------------------
    cfg_path = Path(cfg_path)
    with cfg_path.open("r") as f:
        cfg_dict = yaml.safe_load(f)

    dem_path = Path(cfg_dict["dem_path"])
    output_dir = Path(cfg_dict["output_dir"])
    n_sims = int(cfg_dict.get("n_sims", 10))

    # Make sure base output dir exists (single folder for all sims)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Solver / mesh parameters
    max_triangle_area = float(cfg_dict.get("max_triangle_area", 200.0))
    yieldstep = float(cfg_dict.get("yieldstep", 300.0))
    finaltime = float(cfg_dict.get("finaltime", 12 * 3600.0))
    friction = float(cfg_dict.get("friction", 0.03))

    # Rainfall settings from config
    rain_min = float(cfg_dict.get("rainfall_peak_min", 20.0))
    rain_max = float(cfg_dict.get("rainfall_peak_max", 80.0))

    # Optional reproducibility
    seed = cfg_dict.get("random_seed", None)
    if seed is not None:
        np.random.seed(int(seed))

    # Optional: number of worker processes
    # Add this to YAML if you like: n_workers: 4
    n_workers = int(cfg_dict.get("n_workers", 1))
    if n_workers <= 0:
        n_workers = 1

    print("========== Flood simulation batch ==========")
    print(f"Config file:      {cfg_path}")
    print(f"DEM path:         {dem_path}")
    print(f"Output directory: {output_dir}")
    print(f"n_sims:           {n_sims}")
    print(f"max_triangle_area:{max_triangle_area}")
    print(f"yieldstep (s):    {yieldstep}")
    print(f"finaltime (s):    {finaltime}")
    print(f"friction:         {friction}")
    print(f"rainfall range:   [{rain_min}, {rain_max}] mm/hr")
    print(f"n_workers:        {n_workers}")
    if seed is not None:
        print(f"random_seed:      {seed}")
    print("============================================\n")

    # -----------------------------
    # Build job list
    # -----------------------------
    jobs = []
    for i in range(n_sims):
        run_idx = i
        run_name = f"sim_{run_idx:04d}"   # sim_0000, sim_0001, ...

        sim_cfg = SimConfig(
            dem_path=dem_path,
            output_dir=output_dir,
            run_name=run_name,
            max_triangle_area=max_triangle_area,
            yieldstep=yieldstep,
            finaltime=finaltime,
            friction=friction,
        )

        # Draw rainfall peak here (in parent) so it's reproducible
        r_peak = np.random.uniform(rain_min, rain_max)  # mm/hr

        jobs.append((sim_cfg, r_peak))

    # -----------------------------
    # Run sequential vs parallel
    # -----------------------------
    if n_workers == 1:
        # original behaviour
        for args in jobs:
            _run_single_sim(args)
    else:
        # cap workers at CPU count
        max_procs = cpu_count()
        n_workers = min(n_workers, max_procs)
        print(f"Using {n_workers} worker processes (cpu_count={max_procs})")

        with Pool(processes=n_workers) as pool:
            pool.map(_run_single_sim, jobs)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print("Usage: python -m scripts.run_generate_sims configs/demo_austin.yaml")
        raise SystemExit(1)
    cfg = sys.argv[1] if len(sys.argv) == 2 else "configs/demo_austin.yaml"
    main(cfg)
