"""Excitation plugin registry: the inputs that drive a response simulation.

An excitation is anything that produces the exogenous input d(t) for a
closed-loop experiment. In this regulator framework v1 excitations enter as
an *input disturbance at the plant input*, u = u_ctrl + d(t) — the same
convention the benchmark's step metrics use. Pure initial-condition
experiments use the ``zero`` excitation with an x0.

Registering a new excitation mirrors the controller registry exactly::

    from state_space_control.excitations import Excitation, register_excitation

    @register_excitation('chirp')
    class Chirp(Excitation):
        PARAMS = [{'name': 'f0', 'default': 0.1}, ...]
        def sample(self, t): ...

and it appears in the wizard/CLI with no other change. The ``injection``
class attribute is 'input' for everything in v1; 'reference' and 'output'
are reserved so reference tracking and measurement disturbances can be
added later without renaming anything.
"""

from typing import Dict, List, Type

import numpy as np


class Excitation:
    """Base class. Parameters go in __init__; ``sample`` maps a time grid
    to the input samples d(t) with shape (len(t),)."""

    injection = 'input'
    PARAMS: List[Dict] = []

    def sample(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return getattr(self, 'registry_name', type(self).__name__)


_REGISTRY: Dict[str, Type[Excitation]] = {}


def register_excitation(name: str):
    def deco(cls: Type[Excitation]):
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def make_excitation(name: str, **params) -> Excitation:
    if name not in _REGISTRY:
        raise ValueError(f'Unknown excitation {name!r}; '
                         f'available: {available_excitations()}')
    return _REGISTRY[name](**params)


def available_excitations() -> List[str]:
    return sorted(_REGISTRY)


def excitation_schemas() -> List[Dict]:
    """UI form metadata, same shape as the controller schemas."""
    out = []
    for name in available_excitations():
        cls = _REGISTRY[name]
        out.append({'name': name, 'doc': (cls.__doc__ or '').strip(),
                    'injection': cls.injection,
                    'params': list(cls.PARAMS)})
    return out


def _num(x, name):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number, got {x!r}')


@register_excitation('step')
class Step(Excitation):
    """Constant disturbance switched on at t_start."""

    PARAMS = [
        {'name': 'amplitude', 'default': 1.0, 'doc': 'step height'},
        {'name': 't_start', 'default': 0.0, 'doc': 'switch-on time [s]'},
    ]

    def __init__(self, amplitude=1.0, t_start=0.0):
        self.amplitude = _num(amplitude, 'amplitude')
        self.t_start = _num(t_start, 't_start')

    def sample(self, t):
        return np.where(np.asarray(t) >= self.t_start, self.amplitude, 0.0)


@register_excitation('impulse')
class Impulse(Excitation):
    """Ideal impulse of the given area, realized exactly as the equivalent
    initial-state jump x0 += B[:, channel] * area (a sampled 1-step pulse
    would depend on the grid spacing; the LTI equivalence does not).
    ``sample`` therefore returns zeros — the simulator applies the jump."""

    PARAMS = [{'name': 'area', 'default': 1.0,
               'doc': 'impulse area (N*m*s for torque inputs)'}]

    def __init__(self, area=1.0):
        self.area = _num(area, 'area')

    def sample(self, t):
        return np.zeros(len(np.asarray(t)))


@register_excitation('ramp')
class Ramp(Excitation):
    """Linearly growing disturbance, optionally saturating."""

    PARAMS = [
        {'name': 'slope', 'default': 1.0, 'doc': 'growth rate [1/s]'},
        {'name': 't_start', 'default': 0.0, 'doc': 'onset time [s]'},
        {'name': 'saturation', 'default': None,
         'doc': 'clip magnitude (empty = unbounded)'},
    ]

    def __init__(self, slope=1.0, t_start=0.0, saturation=None):
        self.slope = _num(slope, 'slope')
        self.t_start = _num(t_start, 't_start')
        self.saturation = None if saturation in (None, '') \
            else abs(_num(saturation, 'saturation'))

    def sample(self, t):
        d = self.slope * np.maximum(np.asarray(t, dtype=float) - self.t_start,
                                    0.0)
        if self.saturation is not None:
            d = np.clip(d, -self.saturation, self.saturation)
        return d


@register_excitation('sine')
class Sine(Excitation):
    """Sinusoidal disturbance amplitude*sin(2*pi*freq_hz*t + phase)."""

    PARAMS = [
        {'name': 'amplitude', 'default': 1.0},
        {'name': 'freq_hz', 'default': 0.5, 'doc': 'frequency [Hz]'},
        {'name': 'phase', 'default': 0.0, 'doc': 'phase [rad]'},
    ]

    def __init__(self, amplitude=1.0, freq_hz=0.5, phase=0.0):
        self.amplitude = _num(amplitude, 'amplitude')
        self.freq_hz = _num(freq_hz, 'freq_hz')
        self.phase = _num(phase, 'phase')

    def sample(self, t):
        return self.amplitude * np.sin(
            2.0 * np.pi * self.freq_hz * np.asarray(t, dtype=float)
            + self.phase)


@register_excitation('custom')
class Custom(Excitation):
    """User-supplied samples, linearly interpolated (zero outside the
    given span)."""

    PARAMS = [
        {'name': 't_samples', 'default': [], 'doc': 'time points [s]'},
        {'name': 'u_samples', 'default': [], 'doc': 'values at those times'},
    ]

    def __init__(self, t_samples=(), u_samples=()):
        ts = np.asarray(t_samples, dtype=float)
        us = np.asarray(u_samples, dtype=float)
        if ts.ndim != 1 or ts.shape != us.shape or len(ts) < 2:
            raise ValueError('custom excitation needs matching 1-D t_samples/'
                             'u_samples with at least 2 points')
        if np.any(np.diff(ts) <= 0):
            raise ValueError('custom t_samples must be strictly increasing')
        if not np.all(np.isfinite(us)):
            raise ValueError('custom u_samples must be finite')
        self.ts, self.us = ts, us

    def sample(self, t):
        return np.interp(np.asarray(t, dtype=float), self.ts, self.us,
                         left=0.0, right=0.0)


@register_excitation('zero')
class Zero(Excitation):
    """No input — for pure initial-condition experiments (set x0)."""

    PARAMS = []

    def sample(self, t):
        return np.zeros(len(np.asarray(t)))
