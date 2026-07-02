# state_space_control

Modular **controller-synthesis toolbox** for linear state-space plants —
the control-design companion to
[`urdf_state_space`](https://github.com/rahgirrafi/urdf_state_space.git), which produces the
plants. Ships with LQR, LQG, and H∞ (mixed-sensitivity and general);
new controllers plug in through a registry without touching existing code.

## Dependencies

```bash
# LQR / LQG need only numpy + scipy.
pip install control slycot   # required for the H-infinity controllers
```

## Quick start

```python
from state_space_control import Plant, make_controller

plant = Plant.from_npz('model.npz')        # exported by urdf2ss
# or: Plant.from_model(build_state_space('robot.urdf'))

result = make_controller('lqr', Q=[100, 100, 1, 1], R=0.1).design(plant)
print(result.summary())                    # gains, closed-loop poles, stability
result.save_npz('controller.npz')
```

H∞ mixed sensitivity (weights as scalars or `{num, den}` coefficients):

```python
design = make_controller(
    'hinf_mixsyn',
    W1={'num': [0.5, 10.0], 'den': [1.0, 1e-4]},   # sensitivity shaping
    W2={'num': [1.0, 1.0], 'den': [0.01, 100.0]})  # control effort / rolloff
result = design.design(plant)
print(result.info['gamma'], result.is_stable())
```

## CLI

Design specs live in YAML (see [examples/](examples/)):

```yaml
controller: lqr
params:
  Q: [100, 100, 1, 1]   # scalar => q*I, list => diagonal, nested list => full
  R: 0.1
```

```bash
ros2 run state_space_control ss_design plant.npz lqr_design.yaml -o controller.npz
```

## Controller semantics

A `ControllerResult` holds exactly one of:

- `K` — static state-feedback gain, control law `u = u_eq − K x`
  (LQR; needs full state, e.g. from joint encoders + derivatives).
- `controller` — dynamic output-feedback LTI system from the measured
  outputs `y` to `u`, sign convention already absorbed: the closed loop is
  literally `u = controller(y)` (LQG, H∞).

`result.closed_loop()`, `result.closed_loop_poles()` and
`result.is_stable()` work for both. `u_eq` from the plant is carried along —
apply it as feedforward on the real robot.

`analysis.py` adds `damping_report`, `step_response`, and `settling_time`.

## Adding your own controller

Create `state_space_control/controllers/my_ctrl.py`:

```python
from ..base import ControllerDesign, ControllerResult, register

@register('my_ctrl')
class MyController(ControllerDesign):
    def __init__(self, some_param=1.0):
        self.some_param = some_param

    def design(self, plant):
        K = ...                      # your synthesis
        return ControllerResult(name='my_ctrl', plant=plant, K=K)
```

then import it in `controllers/__init__.py`. It is immediately usable from
`make_controller('my_ctrl', ...)`, the YAML specs, and the CLI.
