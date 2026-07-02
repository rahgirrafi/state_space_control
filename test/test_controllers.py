"""Controller synthesis tests on small analytic plants."""

import numpy as np
import pytest

from state_space_control import Plant, available_controllers, make_controller


@pytest.fixture
def unstable_pendulum():
    """Inverted pendulum linearization: poles at +/- 2 rad/s."""
    return Plant(
        A=np.array([[0.0, 1.0], [4.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        C=np.array([[1.0, 0.0]]),
        D=np.zeros((1, 1)),
    )


def test_registry_lists_core_controllers():
    names = available_controllers()
    assert 'lqr' in names and 'lqg' in names


def test_unknown_controller_raises():
    with pytest.raises(ValueError, match='Unknown controller'):
        make_controller('does_not_exist')


def test_lqr_stabilizes_and_satisfies_riccati(unstable_pendulum):
    result = make_controller('lqr', Q=[10.0, 1.0], R=0.5).design(
        unstable_pendulum)
    assert result.is_stable()
    # Riccati residual: A'P + PA - PBR^-1B'P + Q = 0
    P = result.info['riccati_P']
    A, B = unstable_pendulum.A, unstable_pendulum.B
    Q, R = result.info['cost_Q'], result.info['cost_R']
    residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)
    # K = R^-1 B' P
    np.testing.assert_allclose(result.K, np.linalg.solve(R, B.T @ P))


def test_lqg_output_feedback_stabilizes(unstable_pendulum):
    result = make_controller(
        'lqg', Q=[10.0, 1.0], R=0.5, W=1.0, V=0.01).design(unstable_pendulum)
    assert result.K is None and result.controller is not None
    assert result.controller.n_states == 2
    assert result.is_stable()
    # Separation principle: closed-loop poles = LQR poles + observer poles.
    A, B, C = unstable_pendulum.A, unstable_pendulum.B, unstable_pendulum.C
    K, L = result.info['K'], result.info['L']
    expected = np.concatenate([
        np.linalg.eigvals(A - B @ K), np.linalg.eigvals(A - L @ C)])
    got = result.closed_loop_poles()
    np.testing.assert_allclose(
        np.sort_complex(got), np.sort_complex(expected), atol=1e-8)


def test_result_npz_roundtrip(tmp_path, unstable_pendulum):
    result = make_controller('lqr', Q=1.0, R=1.0).design(unstable_pendulum)
    path = str(tmp_path / 'ctrl.npz')
    result.save_npz(path)
    d = np.load(path)
    np.testing.assert_allclose(d['K'], result.K)
    np.testing.assert_allclose(d['plant_A'], unstable_pendulum.A)


def test_as_matrix_specs(unstable_pendulum):
    from state_space_control.base import as_matrix
    np.testing.assert_allclose(as_matrix(2.0, 2), 2.0 * np.eye(2))
    np.testing.assert_allclose(as_matrix([1.0, 3.0], 2), np.diag([1.0, 3.0]))
    full = [[1.0, 0.1], [0.1, 2.0]]
    np.testing.assert_allclose(as_matrix(full, 2), full)
    with pytest.raises(ValueError, match='diagonal'):
        as_matrix([1.0, 2.0, 3.0], 2)


def test_hinf_mixsyn_stabilizes(unstable_pendulum):
    pytest.importorskip('control')
    pytest.importorskip('slycot')
    design = make_controller(
        'hinf_mixsyn',
        W1={'num': [1.0, 10.0], 'den': [1.0, 0.001]},
        W2=0.1)
    result = design.design(unstable_pendulum)
    assert result.is_stable()
    assert result.info['gamma'] > 0
