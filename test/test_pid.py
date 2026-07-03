"""PID plugin: registry pick-up, composition, stabilization, name matching."""

import numpy as np
import pytest

from state_space_control.base import (
    Plant, available_controllers, make_controller)

# Pendulum about the hanging equilibrium (matches urdf_state_space tests).
M, L, G = 1.5, 0.8, 9.81
I = M * L * L
A = np.array([[0.0, 1.0], [-G / L, 0.0]])
B = np.array([[0.0], [1.0 / I]])


def plant(output_names=None, input_names=None, C=None):
    C = np.array([[1.0, 0.0]]) if C is None else C
    return Plant(A=A, B=B, C=C, D=np.zeros((C.shape[0], 1)),
                 input_names=input_names or [], output_names=output_names or [])


def test_pid_registered():
    assert 'pid' in available_controllers()


def test_pid_controller_dimensions():
    result = make_controller('pid', Kp=50, Ki=10, Kd=5).design(plant())
    assert result.K is None
    k = result.controller
    assert k.A.shape == (2, 2)        # integrator + derivative filter
    assert k.B.shape == (2, 1)
    assert k.C.shape == (1, 2)


def test_pid_stabilizes_pendulum():
    result = make_controller('pid', Kp=50, Ki=10, Kd=5).design(plant())
    assert result.is_stable()


def test_pid_pure_p_matches_static_gain():
    """P-only PID equals static output feedback: closed-loop A = A - B Kp C."""
    result = make_controller('pid', Kp=30.0, Ki=0.0, Kd=0.0).design(plant())
    cl = result.closed_loop()
    expected = A - B @ (30.0 * np.array([[1.0, 0.0]]))
    # The controller adds 2 (unreachable) states; compare the plant block.
    np.testing.assert_allclose(cl.A[:2, :2], expected)


def test_pid_matches_outputs_by_name():
    """Measurement columns follow output names, not positional order."""
    C = np.array([[0.0, 1.0], [1.0, 0.0]])   # row0 = velocity, row1 = position
    p = plant(C=C, output_names=['hinge.qd', 'hinge.q'],
              input_names=['hinge'])
    result = make_controller('pid', Kp=50, Ki=10, Kd=5).design(p)
    k = result.controller
    # All measurement pick-up must be on column 1 ('hinge.q').
    assert np.all(k.B[:, 0] == 0) and np.any(k.B[:, 1] != 0)
    assert np.all(k.D[:, 0] == 0) and np.any(k.D[:, 1] != 0)


def test_pid_missing_position_output_raises():
    p = plant(output_names=['hinge.qd'], input_names=['hinge'])
    with pytest.raises(ValueError, match='position'):
        make_controller('pid', Kp=1.0).design(p)


def test_pid_gain_length_mismatch_raises():
    with pytest.raises(ValueError, match='Kp'):
        make_controller('pid', Kp=[1.0, 2.0]).design(plant())


def test_pid_nonpositive_tau_raises():
    with pytest.raises(ValueError, match='tau'):
        make_controller('pid', Kp=1.0, tau=0.0)
