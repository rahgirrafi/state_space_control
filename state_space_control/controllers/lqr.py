"""Continuous-time LQR: u = u_eq - K x minimizing integral of x'Qx + u'Ru."""

from scipy.linalg import solve, solve_continuous_are

from ..base import ControllerDesign, ControllerResult, Plant, as_matrix, register


@register('lqr')
class LQR(ControllerDesign):
    """State-feedback LQR. Q and R may be scalars, diagonals, or matrices."""

    def __init__(self, Q=1.0, R=1.0):
        self.Q = Q
        self.R = R

    def design(self, plant: Plant) -> ControllerResult:
        Q = as_matrix(self.Q, plant.n_states, 'Q')
        R = as_matrix(self.R, plant.n_inputs, 'R')
        P = solve_continuous_are(plant.A, plant.B, Q, R)
        K = solve(R, plant.B.T @ P)
        return ControllerResult(
            name='lqr', plant=plant, K=K,
            info={'riccati_P': P, 'cost_Q': Q, 'cost_R': R})
