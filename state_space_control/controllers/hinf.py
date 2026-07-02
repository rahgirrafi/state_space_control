"""H-infinity synthesis (needs python-control + slycot).

``hinf_mixsyn`` solves the standard S/KS/T mixed-sensitivity problem

    min_K || [W1 S; W2 K S; W3 T] ||_inf

W1 shapes the sensitivity S (tracking / disturbance rejection), W2 the
control effort K S, W3 the complementary sensitivity T (robustness,
high-frequency rolloff). ``hinf`` runs plain hinfsyn on a user-supplied
generalized plant.
"""

import numpy as np

from ..base import ControllerDesign, ControllerResult, Plant, register


def _weight_to_lti(spec, label: str):
    """scalar -> static gain; {num: [...], den: [...]} -> transfer function."""
    import control
    if spec is None:
        return None
    if np.isscalar(spec):
        return control.tf([float(spec)], [1.0])
    if isinstance(spec, dict) and 'num' in spec and 'den' in spec:
        return control.tf(spec['num'], spec['den'])
    raise ValueError(
        f'{label}: expected a scalar or {{num: [...], den: [...]}}, '
        f'got {spec!r}')


def _flip_feedback_sign(K) -> Plant:
    """mixsyn/hinfsyn controllers expect u = K(-y) around negative feedback;
    store as a y -> u system so closed_loop() can wire it directly."""
    return Plant(A=np.asarray(K.A), B=-np.asarray(K.B),
                 C=np.asarray(K.C), D=-np.asarray(K.D))


@register('hinf_mixsyn')
class MixedSensitivityHInf(ControllerDesign):
    """Weights may be scalars or {num, den} transfer-function coefficients."""

    def __init__(self, W1=None, W2=None, W3=None):
        if W1 is None and W2 is None and W3 is None:
            raise ValueError('hinf_mixsyn needs at least one weight W1/W2/W3')
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3

    def design(self, plant: Plant) -> ControllerResult:
        import control
        P = control.ss(plant.A, plant.B, plant.C, plant.D)
        K, _, (gamma, rcond) = control.mixsyn(
            P,
            w1=_weight_to_lti(self.W1, 'W1'),
            w2=_weight_to_lti(self.W2, 'W2'),
            w3=_weight_to_lti(self.W3, 'W3'))
        return ControllerResult(
            name='hinf_mixsyn', plant=plant,
            controller=_flip_feedback_sign(K),
            info={'gamma': float(gamma), 'rcond': np.asarray(rcond)})


@register('hinf')
class HInf(ControllerDesign):
    """General H-infinity synthesis on an augmented plant.

    The augmented plant (inputs [w; u], outputs [z; y]) is supplied as
    matrices; the design is independent of how you built it.
    """

    def __init__(self, A, B, C, D, n_meas: int, n_ctrl: int):
        self.aug = Plant(A=np.asarray(A, float), B=np.asarray(B, float),
                         C=np.asarray(C, float), D=np.asarray(D, float))
        self.n_meas = int(n_meas)
        self.n_ctrl = int(n_ctrl)

    def design(self, plant: Plant) -> ControllerResult:
        import control
        P_aug = control.ss(self.aug.A, self.aug.B, self.aug.C, self.aug.D)
        K, _, gamma, rcond = control.hinfsyn(P_aug, self.n_meas, self.n_ctrl)
        return ControllerResult(
            name='hinf', plant=plant,
            controller=_flip_feedback_sign(K),
            info={'gamma': float(gamma), 'rcond': np.asarray(rcond)})
