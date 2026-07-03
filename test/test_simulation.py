"""Closed-loop response simulation and RobotTrajectory production."""

import numpy as np
import pytest

from state_space_control.base import ControllerResult, Plant
from state_space_control.excitations import make_excitation
from state_space_control.simulation import (
    annotate_events, simulate_response, to_robot_trajectory)


@pytest.fixture
def plant():
    """Inverted pendulum linearization: poles at +/- 2 rad/s."""
    return Plant(
        A=np.array([[0.0, 1.0], [4.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        C=np.array([[1.0, 0.0]]),
        D=np.zeros((1, 1)),
    )


@pytest.fixture
def stabilized(plant):
    """Hand-picked K placing both closed-loop poles at -2."""
    return ControllerResult(name='test-k', plant=plant,
                            K=np.array([[8.0, 4.0]]))


class StubModel:
    """Duck-typed StateSpaceModel: one 'hinge' joint about q_eq=0.3."""

    joint_names = ['hinge']
    actuated_joint_names = ['hinge']
    q_eq = np.array([0.3])
    u_eq = np.array([1.2])


def test_step_reaches_disturbance_steady_state(stabilized):
    """Final state of a unit step disturbance is -Acl^-1 B (analytic)."""
    out = simulate_response(stabilized, make_excitation('step'),
                            t_final=8.0, n_points=1600)
    Acl = stabilized.closed_loop().A
    x_ss = -np.linalg.solve(Acl, stabilized.plant.B[:, 0])
    np.testing.assert_allclose(out.x_plant[-1], x_ss, atol=1e-4)
    # u = -K x at every sample, read off the same trajectory.
    np.testing.assert_allclose(
        out.u, -(out.x_plant @ np.asarray(stabilized.K).T), atol=1e-12)


def test_impulse_matches_scipy_impulse(stabilized):
    """x0-jump realization == the textbook impulse response."""
    from scipy.signal import impulse
    cl = stabilized.closed_loop()
    t = np.linspace(0.0, 5.0, 500)
    out = simulate_response(stabilized, make_excitation('impulse', area=1.0),
                            t=t)
    _, y_ref = impulse((cl.A, cl.B, cl.C, cl.D), T=t)
    np.testing.assert_allclose(out.y[:, 0], np.asarray(y_ref).ravel(),
                               atol=1e-6)


def test_zero_excitation_with_x0_decays(stabilized):
    out = simulate_response(stabilized, make_excitation('zero'),
                            x0=[0.5, 0.0], t_final=6.0)
    assert abs(out.x_plant[0, 0] - 0.5) < 1e-9
    assert np.max(np.abs(out.x_plant[-1])) < 1e-3
    assert out.stable and not out.capped


def test_unstable_loop_is_time_capped(plant):
    """K = 0 leaves the pole at +2: capped, finite, flagged."""
    result = ControllerResult(name='no-op', plant=plant,
                              K=np.zeros((1, 2)))
    out = simulate_response(result, make_excitation('zero'), x0=[0.01, 0.0],
                            t_final=60.0)
    assert not out.stable and out.capped
    assert out.t[-1] <= np.log(1000.0) / 2.0 + 1e-6
    assert np.all(np.isfinite(out.x_plant))


def test_static_and_dynamic_forms_give_identical_states(plant):
    """Same loop as static K and as a D-only dynamic controller: identical
    plant states, and the state split puts nothing in x_ctrl for static."""
    full = Plant(A=plant.A, B=plant.B, C=np.eye(2), D=np.zeros((2, 1)))
    K = np.array([[8.0, 4.0]])
    r_static = ControllerResult(name='k', plant=full, K=K)
    d_only = Plant(A=np.zeros((0, 0)), B=np.zeros((0, 2)),
                   C=np.zeros((1, 0)), D=-K)
    r_dyn = ControllerResult(name='k-dyn', plant=full, controller=d_only)

    t = np.linspace(0.0, 4.0, 400)
    exc = make_excitation('sine', amplitude=0.5, freq_hz=1.0)
    o_s = simulate_response(r_static, exc, t=t)
    o_d = simulate_response(r_dyn, exc, t=t)
    assert o_s.x_ctrl.shape == (400, 0) and o_d.x_ctrl.shape == (400, 0)
    np.testing.assert_allclose(o_s.x_plant, o_d.x_plant, atol=1e-9)
    np.testing.assert_allclose(o_s.u, o_d.u, atol=1e-9)


def test_bad_channel_rejected(stabilized):
    with pytest.raises(ValueError, match='channel'):
        simulate_response(stabilized, make_excitation('zero'), channel=1)


def test_to_robot_trajectory_zero_response_is_equilibrium_pose(stabilized):
    """THE deviation-coordinate test: no motion renders q_eq, not zero."""
    out = simulate_response(stabilized, make_excitation('zero'), t_final=2.0)
    traj = to_robot_trajectory(StubModel(), stabilized, out)
    np.testing.assert_allclose(traj.q, 0.3, atol=1e-12)
    np.testing.assert_allclose(traj.qd, 0.0, atol=1e-12)
    assert traj.joint_names == ['hinge']


def test_to_robot_trajectory_offsets_deviation_states(stabilized):
    out = simulate_response(stabilized, make_excitation('zero'),
                            x0=[0.1, 0.0], t_final=2.0)
    traj = to_robot_trajectory(StubModel(), stabilized, out)
    assert traj.q[0, 0] == pytest.approx(0.4)          # 0.3 + 0.1
    np.testing.assert_allclose(traj.q[:, 0],
                               0.3 + out.x_plant[:, 0], atol=1e-12)
    np.testing.assert_allclose(traj.qd[:, 0], out.x_plant[:, 1], atol=1e-12)


def test_to_robot_trajectory_meta_is_self_describing(stabilized, tmp_path):
    out = simulate_response(
        stabilized, make_excitation('step', amplitude=0.5), x0=[0.05, 0.0])
    traj = to_robot_trajectory(
        StubModel(), stabilized, out,
        extra_meta={'robot': {'name': 'pendulum'},
                    'controller': {'params': {'Q': 10}}})
    m = traj.meta
    assert m['schema'] == 'robot_trajectory/1'
    assert m['source'] == 'linear-state-space'
    assert m['operating_point']['q_eq'] == {'hinge': pytest.approx(0.3)}
    assert m['operating_point']['u_eq'] == [pytest.approx(1.2)]
    assert m['controller'] == {'type': 'test-k', 'params': {'Q': 10}}
    assert m['excitation']['name'] == 'step'
    assert m['excitation']['params']['amplitude'] == 0.5
    assert m['x0'] == [pytest.approx(0.05), 0.0]
    assert m['sim']['solver'] == 'scipy.signal.lsim'
    assert m['sim']['stable'] is True
    assert 'created' in m and 'numpy' in m['versions']
    # And the whole thing survives the npz boundary.
    path = str(tmp_path / 'traj.npz')
    traj.save_npz(path)
    from state_space_control.trajectory import RobotTrajectory
    assert RobotTrajectory.from_npz(path).meta == m


def test_events_fire_on_large_excursion(stabilized):
    """Big step -> linear_validity + limit_violation; small -> neither."""
    big = simulate_response(stabilized, make_excitation('step', amplitude=8.0),
                            t_final=6.0)
    traj = to_robot_trajectory(StubModel(), stabilized, big,
                               limits={'hinge': (-0.5, 0.5)},
                               validity_threshold=0.2)
    kinds = {e.type for e in traj.events}
    assert 'linear_validity' in kinds and 'limit_violation' in kinds
    lv = next(e for e in traj.events if e.type == 'limit_violation')
    assert lv.subject == 'hinge'
    assert lv.data['t_end'] >= lv.data['t_start']

    small = simulate_response(
        stabilized, make_excitation('step', amplitude=0.1), t_final=6.0)
    traj2 = to_robot_trajectory(StubModel(), stabilized, small,
                                limits={'hinge': (-0.5, 0.5)})
    kinds2 = {e.type for e in traj2.events}
    assert 'linear_validity' not in kinds2
    assert 'limit_violation' not in kinds2
    assert 'settling_time' in kinds2          # stable + moved -> settled


def test_instability_event(plant):
    result = ControllerResult(name='no-op', plant=plant, K=np.zeros((1, 2)))
    out = simulate_response(result, make_excitation('zero'), x0=[0.01, 0.0])
    traj = to_robot_trajectory(StubModel(), result, out)
    assert any(e.type == 'instability' for e in traj.events)


def test_annotate_events_works_on_foreign_trajectory():
    """Annotators are pure functions over the format — no simulator needed
    (a rosbag import gets the same honesty checks)."""
    from state_space_control.trajectory import RobotTrajectory
    t = np.linspace(0.0, 1.0, 50)
    traj = RobotTrajectory(
        t=t, q=(2.0 * t)[:, None], qd=np.full((50, 1), 2.0),
        joint_names=['j1'], meta={})
    events = annotate_events(traj, limits={'j1': (-1.0, 1.0)})
    assert any(e.type == 'limit_violation' for e in events)


def test_state_dimension_mismatch_rejected(stabilized):
    class TwoJointModel(StubModel):
        joint_names = ['a', 'b']
        q_eq = np.zeros(2)
    out = simulate_response(stabilized, make_excitation('zero'), t_final=1.0)
    with pytest.raises(ValueError, match='does not match'):
        to_robot_trajectory(TwoJointModel(), stabilized, out)
