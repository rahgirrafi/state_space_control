"""Analysis helpers for plants and closed loops (scipy-only)."""

from typing import Optional, Tuple

import numpy as np

from .base import Plant


def damping_report(sys: Plant) -> str:
    """Per-pole natural frequency and damping ratio table."""
    lines = ['pole                     wn [rad/s]   zeta']
    for p in np.sort_complex(sys.poles()):
        wn = abs(p)
        zeta = -p.real / wn if wn > 0 else float('nan')
        lines.append(f'{p:24.4g} {wn:11.4g} {zeta:6.3f}')
    return '\n'.join(lines)


def step_response(
    sys: Plant,
    input_index: int = 0,
    t_final: Optional[float] = None,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Step response of one input channel; returns (t, y[t, outputs])."""
    from scipy.signal import StateSpace, step
    lti = StateSpace(sys.A, sys.B[:, [input_index]], sys.C,
                     sys.D[:, [input_index]])
    if t_final is None:
        # ~8 time constants of the slowest stable pole, fallback 10 s.
        stable = [p for p in sys.poles() if p.real < -1e-9]
        t_final = 8.0 / min(-p.real for p in stable) if stable else 10.0
    t = np.linspace(0.0, t_final, n_points)
    t, y = step(lti, T=t)
    return t, np.atleast_2d(y.T).T


def settling_time(
    t: np.ndarray, y: np.ndarray, band: float = 0.02
) -> float:
    """Time after which y stays within +/-band of its final value."""
    yf = y[-1]
    scale = max(abs(yf), 1e-12)
    outside = np.abs(y - yf) > band * scale
    if not outside.any():
        return 0.0
    return float(t[np.max(np.nonzero(outside)) + 1]) \
        if np.max(np.nonzero(outside)) + 1 < len(t) else float('inf')
