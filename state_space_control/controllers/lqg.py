"""LQG: LQR state feedback on a Kalman-filter state estimate.

Controller (from measurement y to control u):

    xhat_dot = (A - B K - L C) xhat + L y
    u        = -K xhat
"""

import numpy as np
from scipy.linalg import solve, solve_continuous_are

from ..base import ControllerDesign, ControllerResult, Plant, as_matrix, register


@register('lqg')
class LQG(ControllerDesign):
    """LQR cost (Q, R) plus Kalman filter with process noise covariance W
    (entering through the input matrix B) and measurement noise covariance V.
    """

    def __init__(self, Q=1.0, R=1.0, W=1.0, V=1.0):
        self.Q = Q
        self.R = R
        self.W = W
        self.V = V

    def design(self, plant: Plant) -> ControllerResult:
        A, B, C = plant.A, plant.B, plant.C
        n, m, p = plant.n_states, plant.n_inputs, plant.n_outputs

        Q = as_matrix(self.Q, n, 'Q')
        R = as_matrix(self.R, m, 'R')
        W = as_matrix(self.W, m, 'W')     # noise on the actuators
        V = as_matrix(self.V, p, 'V')

        P = solve_continuous_are(A, B, Q, R)
        K = solve(R, B.T @ P)

        # Kalman gain via the dual Riccati equation.
        Pe = solve_continuous_are(A.T, C.T, B @ W @ B.T, V)
        L = solve(V.T, (Pe @ C.T).T).T

        controller = Plant(
            A=A - B @ K - L @ C,
            B=L,
            C=-K,
            D=np.zeros((m, p)),
        )
        return ControllerResult(
            name='lqg', plant=plant, controller=controller,
            info={'K': K, 'L': L, 'riccati_P': P, 'riccati_Pe': Pe})
