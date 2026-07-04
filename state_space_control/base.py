"""Core types: Plant, ControllerResult, and the controller registry.

A controller design is a class decorated with ``@register('name')`` that
implements ``design(plant) -> ControllerResult``. Adding a new controller
type to the toolbox means adding one module under ``controllers/`` — nothing
else has to change::

    from state_space_control.base import ControllerDesign, register

    @register('my_controller')
    class MyController(ControllerDesign):
        def __init__(self, gain=1.0):
            self.gain = gain

        def design(self, plant):
            ...
            return ControllerResult(name='my_controller', plant=plant, K=K)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np


@dataclass
class Plant:
    """A linear plant x_dot = A x + B u, y = C x + D u."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    u_eq: Optional[np.ndarray] = None   # feedforward at the operating point

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    @classmethod
    def from_model(cls, model) -> 'Plant':
        """Adapt anything with A/B/C/D attributes (e.g. a
        urdf_state_space.StateSpaceModel or a python-control StateSpace)."""
        return cls(
            A=np.asarray(model.A, dtype=float),
            B=np.asarray(model.B, dtype=float),
            C=np.asarray(model.C, dtype=float),
            D=np.asarray(model.D, dtype=float),
            input_names=list(getattr(model, 'actuated_joint_names', [])
                             or getattr(model, 'input_names', [])),
            output_names=list(getattr(model, 'output_names', [])),
            u_eq=getattr(model, 'u_eq', None),
        )

    @classmethod
    def from_npz(cls, path: str) -> 'Plant':
        """Load a plant saved by urdf_state_space (StateSpaceModel.save_npz)."""
        d = np.load(path, allow_pickle=False)
        return cls(
            A=d['A'], B=d['B'], C=d['C'], D=d['D'],
            input_names=[str(s) for s in d['actuated_joint_names']]
            if 'actuated_joint_names' in d else [],
            output_names=[str(s) for s in d['output_names']]
            if 'output_names' in d else [],
            u_eq=d['u_eq'] if 'u_eq' in d else None,
        )

    def poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)


@dataclass
class ControllerResult:
    """Outcome of a controller synthesis.

    Exactly one of the two is set by a design:

    - ``K``: static state-feedback gain, control law u = u_eq - K x
      (needs full state measurement/estimation).
    - ``controller``: dynamic output-feedback controller as an LTI system
      from the plant measurement y to the control u, sign included — the
      closed loop is formed by literally connecting u = controller(y).
    """

    name: str
    plant: Plant
    K: Optional[np.ndarray] = None
    controller: Optional[Plant] = None
    info: Dict = field(default_factory=dict)

    def closed_loop(self) -> Plant:
        """Assemble the closed-loop system (outputs = plant outputs)."""
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        if np.any(self.plant.D):
            raise NotImplementedError(
                'closed_loop currently assumes a strictly proper plant (D=0)')
        if self.K is not None:
            Acl = A - B @ self.K
            return Plant(A=Acl, B=B, C=C, D=self.plant.D,
                         output_names=self.plant.output_names)
        if self.controller is not None:
            k = self.controller
            nk = k.n_states
            Acl = np.block([
                [A + B @ k.D @ C, B @ k.C],
                [k.B @ C, k.A],
            ])
            Bcl = np.vstack([B, np.zeros((nk, B.shape[1]))])
            Ccl = np.hstack([C, np.zeros((C.shape[0], nk))])
            return Plant(A=Acl, B=Bcl, C=Ccl,
                         D=np.zeros((C.shape[0], B.shape[1])),
                         output_names=self.plant.output_names)
        raise ValueError('result has neither a static gain nor a controller')

    def closed_loop_poles(self) -> np.ndarray:
        return self.closed_loop().poles()

    def is_stable(self, tol: float = 0.0) -> bool:
        return bool(np.all(self.closed_loop_poles().real < -tol))

    def save_npz(self, path: str) -> None:
        data = {'name': self.name,
                'plant_A': self.plant.A, 'plant_B': self.plant.B,
                'plant_C': self.plant.C, 'plant_D': self.plant.D}
        if self.plant.u_eq is not None:
            data['u_eq'] = self.plant.u_eq
        if self.K is not None:
            data['K'] = self.K
        if self.controller is not None:
            data.update(ctrl_A=self.controller.A, ctrl_B=self.controller.B,
                        ctrl_C=self.controller.C, ctrl_D=self.controller.D)
        for key, val in self.info.items():
            arr = np.asarray(val)
            if arr.dtype.kind in 'ifc':
                data[f'info_{key}'] = arr
        np.savez(path, **data)

    def summary(self) -> str:
        lines = [f'controller: {self.name}']
        if self.K is not None:
            lines.append(f'static state-feedback gain K '
                         f'{self.K.shape}:\n{np.array_str(self.K, precision=4)}')
        if self.controller is not None:
            lines.append(f'dynamic controller: {self.controller.n_states} '
                         f'states, y({self.controller.n_inputs}) -> '
                         f'u({self.controller.n_outputs})')
        for key, val in self.info.items():
            if np.isscalar(val):
                lines.append(f'{key}: {val:.6g}' if isinstance(val, float)
                             else f'{key}: {val}')
        poles = np.sort_complex(self.closed_loop_poles())
        lines.append(f'closed-loop poles: {np.array_str(poles, precision=4)}')
        lines.append(f'closed-loop stable: {self.is_stable()}')
        return '\n'.join(lines)


class ControllerDesign:
    """Base class for controller designs. Parameters go in __init__;
    ``design`` maps a Plant to a ControllerResult."""

    def design(self, plant: Plant) -> ControllerResult:
        raise NotImplementedError


_REGISTRY: Dict[str, Type[ControllerDesign]] = {}


def register(name: str):
    """Class decorator adding a ControllerDesign to the registry."""
    def deco(cls: Type[ControllerDesign]):
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def make_controller(name: str, **params) -> ControllerDesign:
    """Instantiate a registered design by name (see available_controllers)."""
    from . import controllers  # noqa: F401  (triggers registration)
    if name not in _REGISTRY:
        raise ValueError(f'Unknown controller {name!r}; '
                         f'available: {available_controllers()}')
    return _REGISTRY[name](**params)


def available_controllers() -> List[str]:
    from . import controllers  # noqa: F401
    return sorted(_REGISTRY)


def as_matrix(spec, n: int, name: str = 'matrix') -> np.ndarray:
    """Turn a YAML-friendly spec into an (n, n) matrix.

    scalar -> scalar * I,  flat list -> diag(list),  nested list -> full.
    """
    if np.isscalar(spec):
        return float(spec) * np.eye(n)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (n,):
            raise ValueError(f'{name}: need {n} diagonal values, got {arr.shape[0]}')
        return np.diag(arr)
    if arr.shape != (n, n):
        raise ValueError(f'{name}: need a {n}x{n} matrix, got {arr.shape}')
    return arr
