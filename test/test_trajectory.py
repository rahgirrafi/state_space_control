"""Canonical trajectory format: hand-built trajectories only, no simulation.

The whole point of RobotTrajectory is that consumers depend on the format,
not on any producer — so nothing in this file imports the simulator.
"""

import numpy as np
import pytest

from state_space_control.trajectory import (
    SCHEMA, FrameSampler, RobotTrajectory, TrajectoryEvent)


def make_traj(**kw):
    t = np.linspace(0.0, 2.0, 5)
    base = dict(
        t=t,
        q=np.column_stack([t, -t]),          # j1 ramps up, j2 ramps down
        qd=np.column_stack([np.ones(5), -np.ones(5)]),
        joint_names=['j1', 'j2'],
        actuated_joint_names=['j1'],
        u=np.linspace(0.0, 1.0, 5)[:, None],
    )
    base.update(kw)
    return RobotTrajectory(**base)


def test_numpy_only_import():
    """trajectory.py is the stable boundary: importing it must not pull in
    scipy, ROS, or the rest of the framework."""
    import importlib
    import sys
    for mod in [m for m in list(sys.modules)
                if m.startswith('state_space_control.trajectory')]:
        del sys.modules[mod]
    before = {m: m in sys.modules for m in ('scipy', 'rclpy')}
    importlib.import_module('state_space_control.trajectory')
    for mod, was_loaded in before.items():
        if not was_loaded:
            assert mod not in sys.modules, f'trajectory.py pulled in {mod}'


def test_validate_accepts_good_trajectory():
    traj = make_traj()
    assert traj.validate() is traj
    assert traj.duration == pytest.approx(2.0)
    assert traj.n_joints == 2


@pytest.mark.parametrize('bad', [
    dict(t=np.array([0.0, 1.0, 1.0, 1.5, 2.0])),         # non-monotonic
    dict(q=np.zeros((5, 3))),                            # wrong joint count
    dict(qd=np.zeros((4, 2))),                           # wrong sample count
    dict(actuated_joint_names=['nope']),                 # unknown actuated
    dict(base_pose=np.zeros((5, 6))),                    # pose needs 7 cols
    dict(base_twist=np.zeros((5, 5))),                   # twist needs 6 cols
    dict(u=np.zeros((3, 1))),                            # u row mismatch
    dict(events=[TrajectoryEvent(t=99.0, type='user')]),  # event out of span
])
def test_validate_rejects_malformed(bad):
    with pytest.raises(ValueError):
        make_traj(**bad).validate()


def test_npz_roundtrip_with_events_and_meta(tmp_path):
    events = [
        TrajectoryEvent(t=0.5, type='limit_violation', subject='j1',
                        message='j1 leaves [-1, 1]',
                        data={'t_start': 0.5, 't_end': 1.0}),
        TrajectoryEvent(t=1.5, type='user', message='note'),
    ]
    meta = {'schema': SCHEMA, 'source': 'test', 'controller': {'type': 'lqr'},
            'operating_point': {'q_eq': {'j1': 0.1}, 'u_eq': [0.0]}}
    traj = make_traj(events=events, meta=meta)
    path = str(tmp_path / 'traj.npz')
    traj.save_npz(path)

    loaded = RobotTrajectory.from_npz(path)
    np.testing.assert_allclose(loaded.t, traj.t)
    np.testing.assert_allclose(loaded.q, traj.q)
    np.testing.assert_allclose(loaded.u, traj.u)
    assert loaded.joint_names == ['j1', 'j2']
    assert loaded.actuated_joint_names == ['j1']
    assert loaded.base_pose is None
    assert loaded.meta['controller'] == {'type': 'lqr'}
    assert [e.to_dict() for e in loaded.events] == \
        [e.to_dict() for e in events]
    # allow_pickle=False must be enough to read the file back.
    np.load(path, allow_pickle=False)


def test_npz_roundtrip_with_base_pose(tmp_path):
    n = 5
    base_pose = np.zeros((n, 7))
    base_pose[:, 0] = np.linspace(0, 1, n)     # x drifts
    base_pose[:, 6] = 1.0                      # identity quaternion (w last)
    traj = make_traj(base_pose=base_pose, base_twist=np.zeros((n, 6)))
    path = str(tmp_path / 'floating.npz')
    traj.save_npz(path)
    loaded = RobotTrajectory.from_npz(path)
    np.testing.assert_allclose(loaded.base_pose, base_pose)
    assert loaded.base_twist.shape == (n, 6)


def test_from_npz_rejects_wrong_schema(tmp_path):
    path = str(tmp_path / 'bad.npz')
    np.savez(path, schema=np.array('robot_trajectory/999'),
             t=np.arange(3.0), q=np.zeros((3, 1)), qd=np.zeros((3, 1)),
             joint_names=np.array(['j1']), header=np.array('{}'))
    with pytest.raises(ValueError, match='schema'):
        RobotTrajectory.from_npz(path)


def test_from_npz_rejects_plain_npz(tmp_path):
    path = str(tmp_path / 'not_a_traj.npz')
    np.savez(path, A=np.eye(2))
    with pytest.raises(ValueError, match='schema'):
        RobotTrajectory.from_npz(path)


def test_sampler_linear_interpolation():
    sampler = FrameSampler(make_traj())
    frame = sampler.frame_at(0.25)             # halfway into [0, 0.5]
    assert frame.t == pytest.approx(0.25)
    assert frame.joint_positions['j1'] == pytest.approx(0.25)
    assert frame.joint_positions['j2'] == pytest.approx(-0.25)
    assert frame.joint_velocities['j1'] == pytest.approx(1.0)
    assert frame.u[0] == pytest.approx(0.125)
    assert frame.base_pose is None


def test_sampler_hold_interpolation():
    sampler = FrameSampler(make_traj(), method='hold')
    frame = sampler.frame_at(0.49)
    assert frame.joint_positions['j1'] == pytest.approx(0.0)   # previous knot


def test_sampler_clamps_out_of_range():
    sampler = FrameSampler(make_traj())
    lo, hi = sampler.frame_at(-10.0), sampler.frame_at(10.0)
    assert lo.t == 0.0 and lo.joint_positions['j1'] == pytest.approx(0.0)
    assert hi.t == 2.0 and hi.joint_positions['j1'] == pytest.approx(2.0)


def test_sampler_exact_knots():
    sampler = FrameSampler(make_traj())
    for tk, qk in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]:
        assert sampler.frame_at(tk).joint_positions['j1'] == pytest.approx(qk)


def test_sampler_base_pose_slerp():
    n = 5
    base_pose = np.zeros((n, 7))
    base_pose[:, 6] = 1.0
    # Last sample rotated 90 deg about z: q = [0,0,sin45,cos45].
    base_pose[-1, 3:] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    base_pose[-1, 6] = np.cos(np.pi / 4)
    sampler = FrameSampler(make_traj(base_pose=base_pose))
    frame = sampler.frame_at(1.75)             # halfway through last segment
    quat = frame.base_pose[3:]
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    # Halfway between identity and 90 deg = 45 deg about z.
    assert quat[2] == pytest.approx(np.sin(np.pi / 8), abs=1e-9)
    assert quat[3] == pytest.approx(np.cos(np.pi / 8), abs=1e-9)


def test_sampler_rejects_unknown_method():
    with pytest.raises(ValueError, match='interpolation'):
        FrameSampler(make_traj(), method='cubic')
