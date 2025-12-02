# data_gen/rainfall_scenarios.py
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple


class UniformHyetograph:
    """
    A simple temporal rainfall pattern (hyetograph) in m/s.

    Default shape (can be tweaked via __init__):
      - 0–1 h: no rain
      - 1–6 h: constant heavy rain (r_peak_mm_hr)
      - 6–12 h: exponential decay to ~0
      - >12 h: no rain
    """

    def __init__(
        self,
        r_peak_mm_hr: float = 40.0,
        t_on: float = 1.0 * 3600.0,
        t_peak_end: float = 6.0 * 3600.0,
        t_decay_end: float = 12.0 * 3600.0,
    ):
        # convert mm/hr -> m/s
        self.r_peak = r_peak_mm_hr / 1000.0 / 3600.0
        self.t_on = t_on
        self.t_peak_end = t_peak_end
        self.t_decay_end = t_decay_end

    def __call__(self, t: float) -> float:
        """Return rainfall intensity at time t in m/s."""
        if t < self.t_on or t > self.t_decay_end:
            return 0.0

        if t <= self.t_peak_end:
            # constant peak rain
            return self.r_peak

        # exponential decay between t_peak_end and t_decay_end
        tau = self.t_decay_end - self.t_peak_end
        if tau <= 0:
            return 0.0

        x = (t - self.t_peak_end) / tau  # 0 -> 1
        # e^-3 ~ 0.05 at x = 1
        return self.r_peak * math.exp(-3.0 * x)


def make_uniform_rain(r_peak_mm_hr: float = 40.0) -> Callable[[float], float]:
    # start raining immediately
    h = UniformHyetograph(
        r_peak_mm_hr=r_peak_mm_hr,
        t_on=0.0,
        t_peak_end=4.0 * 3600.0,
        t_decay_end=10.0 * 3600.0,
    )

    def f(t: float) -> float:
        return h(t)

    return f


# -------------------------------------------------------------------------
# Optional: polygon helper if you later want spatially-limited rainfall
# -------------------------------------------------------------------------


def point_in_poly(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    """
    Ray-casting point-in-polygon test.

    Parameters
    ----------
    x, y : float
        Point coordinates.
    poly : sequence[(x, y)]
        Polygon vertices in order.

    Returns
    -------
    bool
        True if point is inside polygon.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if cond:
            inside = not inside
    return inside
