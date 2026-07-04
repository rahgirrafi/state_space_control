"""Closed-loop response simulation → RobotTrajectory production.

``simulate_response`` works on any ControllerResult through
``closed_loop()`` — static gain or dynamic controller, present or future
plugin — in deviation coordinates. ``to_robot_trajectory`` is the single
place where deviation states become absolute joint positions
(q = q_eq + δq) and where reproducibility metadata and event annotations
are attached; everything downstream consumes only the canonical
RobotTrajectory.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis import settling_time
from .base import ControllerResult
from .excitations import Excitation, Impulse
from .trajectory import SCHEMA, RobotTrajectory, TrajectoryEvent

# Cap an unstable run once the state has grown ~1000x: the divergence is
# visible long before the numbers overflow the renderer.
_GROWTH_CAP = np.log(1000.0)
_UNSTABLE_T_MAX = 10.0


def control_output_matrix(result: ControllerResult) -> np.ndarray:
    """Cu such that u(t) = Cu @ x_cl(t) for the closed loop of ``result``.

    Mirrors the state ordering of ControllerResult.closed_loop():
    [x_plant] for static K, [x_plant; x_controller] for dynamic.
    """
    plant = result.plant
    if result.K is not None:
        return -np.asarray(result.K)
    k = result.controller
    # u = k.D y + k.C xk,  y = C x  (plant strictly proper).
    return np.hstack([np.asarray(k.D) @ plant.C, np.asarray(k.C)])


def default_t_final(poles: np.ndarray) -> float:
    """~8 time constants of the slowest stable pole, fallback 10 s."""
    stable = [p for p in poles if p.real < -1e-9]
    return 8.0 / min(-p.real for p in stable) if stable else 10.0


@dataclass
class SimOutput:
    """Deviation-space result of one closed-loop simulation."""

    t: np.ndarray
    x_plant: np.ndarray            # (N, n_plant) deviation states
    x_ctrl: np.ndarray             # (N, n_ctrl), empty for static K
    y: np.ndarray                  # (N, n_outputs)
    u: np.ndarray                  # (N, m) controller output (deviation)
    d: np.ndarray                  # (N,) exogenous input on the channel
    channel: int
    x0: np.ndarray
    stable: bool
    capped: bool                   # True if t_final was shortened
    excitation_meta: Dict = field(default_factory=dict)


def _excitation_params(exc: Excitation) -> Dict:
    params = {}
    for key, val in vars(exc).items():
        if key.startswith('_'):
            continue
        params[key] = val.tolist() if isinstance(val, np.ndarray) else val
    return params


def simulate_response(
    result: ControllerResult,
    excitation: Excitation,
    *,
    x0: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    t_final: Optional[float] = None,
    n_points: int = 1200,
    channel: int = 0,
) -> SimOutput:
    """Simulate the closed loop under an excitation at one plant input.

    The excitation enters as an input disturbance, u = u_ctrl + d(t);
    ``x0`` is the initial deviation state of the *plant* (controller states
    start at zero). Unstable loops are simulated but time-capped so the
    divergence stays renderable.
    """
    from scipy.signal import lsim

    cl = result.closed_loop()
    Cu = control_output_matrix(result)
    n_plant = result.plant.n_states
    n_cl = cl.n_states
    m = result.plant.n_inputs
    if not 0 <= channel < m:
        raise ValueError(f'channel must be in [0, {m}), got {channel}')

    poles = cl.poles()
    stable = bool(np.all(poles.real < 0))
    capped = False
    if t is None:
        if t_final is None:
            t_final = default_t_final(poles)
        if not stable:
            sigma = float(np.max(poles.real))
            t_growth = _GROWTH_CAP / sigma if sigma > 1e-9 else _UNSTABLE_T_MAX
            cap = min(t_growth, _UNSTABLE_T_MAX)
            if t_final > cap:
                t_final, capped = cap, True
        t = np.linspace(0.0, float(t_final), int(n_points))
    else:
        t = np.asarray(t, dtype=float)

    x0_plant = np.zeros(n_plant) if x0 is None \
        else np.asarray(x0, dtype=float).reshape(n_plant)
    x0_cl = np.zeros(n_cl)
    x0_cl[:n_plant] = x0_plant

    d = np.asarray(excitation.sample(t), dtype=float).reshape(len(t))
    if isinstance(excitation, Impulse):
        # Exact LTI impulse: equivalent initial-state jump (grid-independent).
        x0_cl += cl.B[:, channel] * excitation.area

    U = np.zeros((len(t), cl.B.shape[1]))
    U[:, channel] = d
    _, _, x_cl = lsim((cl.A, cl.B, cl.C, cl.D), U=U, T=t, X0=x0_cl)
    x_cl = np.atleast_2d(x_cl)
    if x_cl.shape != (len(t), n_cl):
        x_cl = x_cl.reshape(len(t), n_cl)

    return SimOutput(
        t=t,
        x_plant=x_cl[:, :n_plant],
        x_ctrl=x_cl[:, n_plant:],
        y=x_cl @ cl.C.T,
        u=x_cl @ Cu.T,
        d=d,
        channel=channel,
        x0=x0_plant,
        stable=stable,
        capped=capped,
        excitation_meta={
            'name': excitation.describe(),
            'params': _excitation_params(excitation),
            'channel': channel,
            'injection': excitation.injection,
        },
    )


def _versions() -> Dict[str, str]:
    out = {}
    for pkg in ('state_space_control', 'urdf_state_space', 'numpy', 'scipy'):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = 'unknown'
    return out


def _violation_spans(t: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous True runs in ``mask`` as (t_start, t_end) spans."""
    spans = []
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return spans
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    for s, e in zip(starts, ends):
        spans.append((float(t[s]), float(t[e])))
    return spans


def annotate_events(
    traj: RobotTrajectory,
    *,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    validity_threshold: float = 0.2,
) -> List[TrajectoryEvent]:
    """Honesty annotations, computable for a trajectory from *any* source.

    - ``limit_violation``: q leaves the URDF limits (linear sims ignore
      them; flagged, never clamped).
    - ``linear_validity``: ``|q − q_eq|`` exceeds the threshold — beyond it the
      linearized response is fiction (needs meta['operating_point']).
    - ``instability`` / ``settling_time`` / ``overshoot_peak``: read off the
      actuated joints when the producer recorded stability info.
    """
    events: List[TrajectoryEvent] = []
    t = np.asarray(traj.t, dtype=float)
    q_eq = (traj.meta.get('operating_point') or {}).get('q_eq') or {}
    stable = (traj.meta.get('sim') or {}).get('stable')

    for j, name in enumerate(traj.joint_names):
        qj = traj.q[:, j]
        if limits and name in limits:
            lo, hi = limits[name]
            for t0, t1 in _violation_spans(t, (qj < lo) | (qj > hi)):
                events.append(TrajectoryEvent(
                    t=t0, type='limit_violation', subject=name,
                    message=f'{name} leaves [{lo:.4g}, {hi:.4g}]',
                    data={'t_start': t0, 't_end': t1}))
        if name in q_eq:
            dq = np.abs(qj - float(q_eq[name]))
            if np.any(dq > validity_threshold):
                t0 = float(t[int(np.argmax(dq > validity_threshold))])
                events.append(TrajectoryEvent(
                    t=t0, type='linear_validity', subject=name,
                    message=(f'|{name} − q_eq| exceeds '
                             f'{validity_threshold:.3g} rad — linear-model '
                             'validity is doubtful beyond this point'),
                    data={'threshold': validity_threshold,
                          'max_deviation': float(dq.max())}))

    if stable is False:
        events.append(TrajectoryEvent(
            t=float(t[0]), type='instability',
            message='closed loop is unstable — trajectory shows divergence'))
    elif stable and q_eq:
        for name in traj.actuated_joint_names:
            if name not in traj.joint_names or name not in q_eq:
                continue
            j = traj.joint_names.index(name)
            dq = traj.q[:, j] - float(q_eq[name])
            if np.max(np.abs(dq)) < 1e-12:
                continue
            ts = settling_time(t, dq)
            if np.isfinite(ts) and ts > 0.0:
                events.append(TrajectoryEvent(
                    t=float(ts), type='settling_time', subject=name,
                    message=f'{name} settled (2% band)'))
            k = int(np.argmax(np.abs(dq)))
            if 0 < k < len(t) - 1:
                events.append(TrajectoryEvent(
                    t=float(t[k]), type='overshoot_peak', subject=name,
                    message=f'{name} peak deviation {dq[k]:+.4g} rad',
                    data={'peak': float(dq[k])}))

    events.sort(key=lambda ev: ev.t)
    return events


def to_robot_trajectory(
    model,
    result: ControllerResult,
    sim: SimOutput,
    *,
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
    validity_threshold: float = 0.2,
    extra_meta: Optional[Dict] = None,
) -> RobotTrajectory:
    """Deviation states → canonical trajectory. THE q = q_eq + δq step.

    ``model`` is duck-typed: anything with q_eq, u_eq, joint_names and
    actuated_joint_names (e.g. a urdf_state_space.StateSpaceModel).
    """
    joint_names = list(model.joint_names)
    nj = len(joint_names)
    if sim.x_plant.shape[1] != 2 * nj:
        raise ValueError(
            f'plant state dimension {sim.x_plant.shape[1]} does not match '
            f'2 x {nj} joints — is this trajectory from this model?')
    q_eq = np.asarray(model.q_eq, dtype=float).reshape(nj)
    u_eq = np.asarray(model.u_eq, dtype=float).ravel()

    meta = {
        'schema': SCHEMA,
        'source': 'linear-state-space',
        'created': datetime.datetime.now().astimezone().isoformat(),
        'operating_point': {
            'q_eq': {j: float(q_eq[k]) for k, j in enumerate(joint_names)},
            'u_eq': u_eq.tolist(),
        },
        'controller': {'type': result.name},
        'excitation': dict(sim.excitation_meta),
        'x0': sim.x0.tolist(),
        'sim': {
            'solver': 'scipy.signal.lsim',
            't_final': float(sim.t[-1]),
            'n_points': int(len(sim.t)),
            'dt': float(sim.t[1] - sim.t[0]) if len(sim.t) > 1 else 0.0,
            'stable': sim.stable,
            'capped': sim.capped,
        },
        'versions': _versions(),
    }
    for key, val in (extra_meta or {}).items():
        if isinstance(val, dict) and isinstance(meta.get(key), dict):
            meta[key].update(val)
        else:
            meta[key] = val

    traj = RobotTrajectory(
        t=sim.t,
        q=q_eq[None, :] + sim.x_plant[:, :nj],
        qd=sim.x_plant[:, nj:],
        joint_names=joint_names,
        actuated_joint_names=list(model.actuated_joint_names),
        u=sim.u,
        meta=meta,
    )
    traj.events = annotate_events(traj, limits=limits,
                                  validity_threshold=validity_threshold)
    return traj.validate()
