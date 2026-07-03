"""Per-joint PID as a registered LTI controller design.

Like every design in this toolbox, the PID is a *regulator* about the
operating point: it drives the deviation outputs to zero,

    u = u_eq + PID(-delta_y),

not a setpoint-tracking loop. The returned ``ControllerResult.controller``
is the LTI system from plant measurement y to control u (sign included), so
``closed_loop()`` composes it exactly like the LQG/H-infinity cases.

Each actuated input channel gets an independent parallel PID with a filtered
derivative, acting on the position measurement of its own joint:

    u_i = -( Kp_i y_i + Ki_i \\int y_i dt + Kd_i d/dt[y_i]_filtered )

with derivative filter Kd s / (tau s + 1) -- two states per channel
(integrator + filter). Channel i's measurement is the plant output named
'<actuated_joint_i>.q' when names are available; otherwise output i is used
and the plant must have at least as many outputs as inputs.
"""

import numpy as np

from ..base import ControllerDesign, ControllerResult, Plant, register


def _per_channel(value, m: int, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.shape == (1,):
        return np.full(m, arr[0])
    if arr.shape != (m,):
        raise ValueError(f'{name}: need a scalar or {m} values, '
                         f'got shape {arr.shape}')
    return arr


def _measurement_indices(plant: Plant) -> list:
    """Output index measured by each input channel."""
    m = plant.n_inputs
    if plant.input_names and plant.output_names:
        idx = []
        for joint in plant.input_names:
            want = f'{joint}.q'
            if want not in plant.output_names:
                raise ValueError(
                    f'pid: plant outputs {plant.output_names} do not include '
                    f'{want!r}; the PID needs the position of every actuated '
                    'joint among the outputs')
            idx.append(plant.output_names.index(want))
        return idx
    if plant.n_outputs < m:
        raise ValueError(
            f'pid: plant has {plant.n_outputs} outputs but {m} inputs and no '
            'signal names to match them by; need one position measurement '
            'per actuated joint')
    return list(range(m))


@register('pid')
class PID(ControllerDesign):
    """Parallel PID with filtered derivative; Kp/Ki/Kd scalar or per-joint."""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, tau=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError('pid: tau must be positive')

    def design(self, plant: Plant) -> ControllerResult:
        m, p = plant.n_inputs, plant.n_outputs
        Kp = _per_channel(self.Kp, m, 'Kp')
        Ki = _per_channel(self.Ki, m, 'Ki')
        Kd = _per_channel(self.Kd, m, 'Kd')
        meas = _measurement_indices(plant)

        # Per channel i, states [z_i (integrator), w_i (derivative filter)]:
        #   z_dot = y,  w_dot = (y - w)/tau
        #   u = -(Kp + Kd/tau) y - Ki z + (Kd/tau) w
        nk = 2 * m
        A = np.zeros((nk, nk))
        B = np.zeros((nk, p))
        C = np.zeros((m, nk))
        D = np.zeros((m, p))
        for i in range(m):
            zi, wi = 2 * i, 2 * i + 1
            A[wi, wi] = -1.0 / self.tau
            B[zi, meas[i]] = 1.0
            B[wi, meas[i]] = 1.0 / self.tau
            C[i, zi] = -Ki[i]
            C[i, wi] = Kd[i] / self.tau
            D[i, meas[i]] = -(Kp[i] + Kd[i] / self.tau)

        controller = Plant(A=A, B=B, C=C, D=D)
        return ControllerResult(
            name='pid', plant=plant, controller=controller,
            info={'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'tau': self.tau})
