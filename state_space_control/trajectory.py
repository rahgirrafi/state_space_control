"""RobotTrajectory: the framework's canonical motion-interchange format.

Every module that touches motion data speaks this format — producers
(the linear state-space simulator today; nonlinear simulators, MuJoCo,
rosbag importers, real-robot logs later) write it, consumers (web viewer,
RViz playback, benchmark comparison, report/video export) read it, and
neither side knows about the other. A consumer requires only a *valid
trajectory*; nothing here imports controllers, excitations, or ROS.

This module must stay importable with numpy alone.

npz schema ``robot_trajectory/1``
---------------------------------
Arrays: ``t (N,)``, ``q (N, nj)``, ``qd (N, nj)``, optional ``u (N, m)``,
optional ``base_pose (N, 7)`` as [x y z, qx qy qz qw] in the world frame,
optional ``base_twist (N, 6)``; ``joint_names``, ``actuated_joint_names``
as string arrays; ``schema`` and a JSON-encoded ``header`` string holding
``meta`` and ``events``. Loads with ``allow_pickle=False``.

``base_pose is None`` *means* fixed base: the URDF root stays wherever the
consumer's world anchor puts it — do not invent an identity transform.

Reserved event types (consumers ignore unknown types — additive evolution):
``limit_violation`` (subject=joint, data={'t_start','t_end'}),
``linear_validity`` (subject=joint), ``instability``, ``settling_time``,
``overshoot_peak``, ``saturation``, ``user``.

Control inputs ``u`` are stored in deviation form (u − u_eq) with ``u_eq``
recorded in ``meta['operating_point']`` — one convention, so producers
never disagree silently.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 'robot_trajectory/1'

EVENT_TYPES = ('limit_violation', 'linear_validity', 'instability',
               'settling_time', 'overshoot_peak', 'saturation', 'user')


@dataclass
class TrajectoryEvent:
    """A time-stamped annotation on a trajectory."""

    t: float
    type: str
    subject: str = ''
    message: str = ''
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {'t': float(self.t), 'type': self.type, 'subject': self.subject,
                'message': self.message, 'data': self.data}

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrajectoryEvent':
        return cls(t=float(d['t']), type=str(d['type']),
                   subject=str(d.get('subject', '')),
                   message=str(d.get('message', '')),
                   data=dict(d.get('data') or {}))


@dataclass
class RobotFrame:
    """The robot's configuration at one instant — what renderers consume."""

    t: float
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    base_pose: Optional[np.ndarray] = None      # [x y z, qx qy qz qw]
    base_twist: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None


@dataclass
class RobotTrajectory:
    """Time-stamped motion of one URDF-based robot, any base type.

    Joint positions are *absolute* (operating-point offsets already applied
    by the producer); consumers never see deviation coordinates.
    """

    t: np.ndarray                                # (N,)
    q: np.ndarray                                # (N, nj) absolute positions
    qd: np.ndarray                               # (N, nj)
    joint_names: List[str]
    actuated_joint_names: List[str] = field(default_factory=list)
    u: Optional[np.ndarray] = None               # (N, m), deviation form
    base_pose: Optional[np.ndarray] = None       # (N, 7); None = fixed base
    base_twist: Optional[np.ndarray] = None      # (N, 6)
    events: List[TrajectoryEvent] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def n_joints(self) -> int:
        return self.q.shape[1]

    def validate(self) -> 'RobotTrajectory':
        """Fail loudly at the boundary, not deep inside a renderer."""
        t = np.asarray(self.t, dtype=float)
        if t.ndim != 1 or len(t) < 2:
            raise ValueError('trajectory needs a 1-D time array with >= 2 samples')
        if np.any(np.diff(t) <= 0):
            raise ValueError('trajectory time must be strictly increasing')
        n = len(t)
        nj = len(self.joint_names)
        for name, arr, cols in (('q', self.q, nj), ('qd', self.qd, nj)):
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n, cols):
                raise ValueError(
                    f'{name} must have shape ({n}, {cols}), got {arr.shape}')
        if not np.all(np.isfinite(self.q)):
            raise ValueError('q contains non-finite samples')
        if self.u is not None and np.asarray(self.u).shape[0] != n:
            raise ValueError('u must have one row per time sample')
        if self.base_pose is not None:
            bp = np.asarray(self.base_pose, dtype=float)
            if bp.shape != (n, 7):
                raise ValueError(
                    f'base_pose must have shape ({n}, 7), got {bp.shape}')
        if self.base_twist is not None:
            bt = np.asarray(self.base_twist, dtype=float)
            if bt.shape != (n, 6):
                raise ValueError(
                    f'base_twist must have shape ({n}, 6), got {bt.shape}')
        unknown = [j for j in self.actuated_joint_names
                   if j not in self.joint_names]
        if unknown:
            raise ValueError(f'actuated joints not in joint_names: {unknown}')
        for ev in self.events:
            if not (t[0] - 1e-9 <= ev.t <= t[-1] + 1e-9):
                raise ValueError(
                    f'event {ev.type!r} at t={ev.t} outside [{t[0]}, {t[-1]}]')
        return self

    def save_npz(self, path: str) -> None:
        self.validate()
        header = json.dumps({
            'meta': self.meta,
            'events': [ev.to_dict() for ev in self.events],
        })
        data = {
            'schema': np.array(SCHEMA),
            't': np.asarray(self.t, dtype=float),
            'q': np.asarray(self.q, dtype=float),
            'qd': np.asarray(self.qd, dtype=float),
            'joint_names': np.array(self.joint_names),
            'actuated_joint_names': np.array(self.actuated_joint_names),
            'header': np.array(header),
        }
        if self.u is not None:
            data['u'] = np.asarray(self.u, dtype=float)
        if self.base_pose is not None:
            data['base_pose'] = np.asarray(self.base_pose, dtype=float)
        if self.base_twist is not None:
            data['base_twist'] = np.asarray(self.base_twist, dtype=float)
        np.savez(path, **data)

    @classmethod
    def from_npz(cls, path: str) -> 'RobotTrajectory':
        d = np.load(path, allow_pickle=False)
        schema = str(d['schema']) if 'schema' in d else '<missing>'
        if schema != SCHEMA:
            raise ValueError(
                f'{path}: unsupported trajectory schema {schema!r} '
                f'(this reader understands {SCHEMA!r})')
        header = json.loads(str(d['header'])) if 'header' in d else {}
        traj = cls(
            t=d['t'], q=d['q'], qd=d['qd'],
            joint_names=[str(s) for s in d['joint_names']],
            actuated_joint_names=[str(s) for s in d['actuated_joint_names']]
            if 'actuated_joint_names' in d else [],
            u=d['u'] if 'u' in d else None,
            base_pose=d['base_pose'] if 'base_pose' in d else None,
            base_twist=d['base_twist'] if 'base_twist' in d else None,
            events=[TrajectoryEvent.from_dict(e)
                    for e in header.get('events', [])],
            meta=header.get('meta', {}),
        )
        return traj.validate()


def _slerp(p: np.ndarray, q: np.ndarray, s: float) -> np.ndarray:
    """Spherical interpolation between two xyzw quaternions."""
    dot = float(np.dot(p, q))
    if dot < 0.0:
        q, dot = -q, -dot
    if dot > 0.9995:
        out = p + s * (q - p)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - s) * theta) * p + np.sin(s * theta) * q) / np.sin(theta)


class FrameSampler:
    """Answers "what does the robot look like at simulation time t?".

    Owns interpolation entirely; playback engines, plot cursors and offline
    exporters all call ``frame_at`` so a 30 fps video render and a jittery
    60 Hz timer follow the identical code path. ``t`` is clamped to the
    trajectory's time span.
    """

    METHODS = ('linear', 'hold')

    def __init__(self, traj: RobotTrajectory, method: str = 'linear'):
        if method not in self.METHODS:
            raise ValueError(f'unknown interpolation {method!r}; '
                             f'available: {list(self.METHODS)}')
        self.traj = traj.validate()
        self.method = method
        self._t = np.asarray(traj.t, dtype=float)

    def _index(self, t: float):
        """Return (i, s): segment index and normalized position in it."""
        tt = self._t
        t = float(np.clip(t, tt[0], tt[-1]))
        i = int(np.searchsorted(tt, t, side='right') - 1)
        i = min(max(i, 0), len(tt) - 2)
        span = tt[i + 1] - tt[i]
        s = (t - tt[i]) / span if span > 0 else 0.0
        if self.method == 'hold':
            s = 0.0
        return i, s, t

    def _lerp_rows(self, arr: Optional[np.ndarray], i: int, s: float):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        return arr[i] if s == 0.0 else (1 - s) * arr[i] + s * arr[i + 1]

    def frame_at(self, t: float) -> RobotFrame:
        i, s, t = self._index(t)
        q = self._lerp_rows(self.traj.q, i, s)
        qd = self._lerp_rows(self.traj.qd, i, s)
        base_pose = None
        if self.traj.base_pose is not None:
            bp = np.asarray(self.traj.base_pose, dtype=float)
            if s == 0.0:
                base_pose = bp[i].copy()
            else:
                base_pose = np.empty(7)
                base_pose[:3] = (1 - s) * bp[i, :3] + s * bp[i + 1, :3]
                base_pose[3:] = _slerp(bp[i, 3:], bp[i + 1, 3:], s)
        names = self.traj.joint_names
        return RobotFrame(
            t=t,
            joint_positions={n: float(q[k]) for k, n in enumerate(names)},
            joint_velocities={n: float(qd[k]) for k, n in enumerate(names)},
            base_pose=base_pose,
            base_twist=self._lerp_rows(self.traj.base_twist, i, s),
            u=self._lerp_rows(self.traj.u, i, s),
        )
