"""Excitation registry and built-in signal shapes."""

import numpy as np
import pytest

from state_space_control.excitations import (
    available_excitations, excitation_schemas, make_excitation)

T = np.linspace(0.0, 2.0, 201)


def test_registry_lists_builtins():
    names = available_excitations()
    assert {'step', 'impulse', 'ramp', 'sine', 'custom', 'zero'} <= set(names)


def test_unknown_excitation_raises():
    with pytest.raises(ValueError, match='Unknown excitation'):
        make_excitation('does_not_exist')


def test_schemas_have_ui_form_shape():
    schemas = {s['name']: s for s in excitation_schemas()}
    assert schemas['step']['injection'] == 'input'
    assert any(p['name'] == 'amplitude' for p in schemas['step']['params'])
    assert schemas['zero']['params'] == []


def test_step_switches_at_t_start():
    d = make_excitation('step', amplitude=2.0, t_start=1.0).sample(T)
    assert d[T < 1.0].max() == 0.0
    assert np.all(d[T >= 1.0] == 2.0)


def test_ramp_saturates():
    d = make_excitation('ramp', slope=2.0, t_start=0.5,
                        saturation=1.0).sample(T)
    assert d[0] == 0.0
    assert d.max() == pytest.approx(1.0)
    assert d[np.searchsorted(T, 0.75)] == pytest.approx(0.5, abs=0.02)


def test_sine_amplitude_and_frequency():
    d = make_excitation('sine', amplitude=3.0, freq_hz=0.5).sample(T)
    assert d.max() == pytest.approx(3.0, abs=1e-3)
    assert d[np.searchsorted(T, 1.0)] == pytest.approx(0.0, abs=1e-6)


def test_custom_interpolates_and_zero_pads():
    exc = make_excitation('custom', t_samples=[0.5, 1.0, 1.5],
                          u_samples=[0.0, 2.0, 0.0])
    d = exc.sample(T)
    assert d[0] == 0.0 and d[-1] == 0.0
    assert d[np.searchsorted(T, 0.75)] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize('kw', [
    dict(t_samples=[0.0], u_samples=[1.0]),              # too short
    dict(t_samples=[0.0, 1.0], u_samples=[1.0]),         # length mismatch
    dict(t_samples=[1.0, 0.5], u_samples=[0.0, 1.0]),    # non-increasing
    dict(t_samples=[0.0, 1.0], u_samples=[0.0, np.inf]),  # non-finite
])
def test_custom_rejects_bad_samples(kw):
    with pytest.raises(ValueError):
        make_excitation('custom', **kw)


def test_impulse_samples_are_zero():
    """The impulse is realized as a state jump, not a fat pulse."""
    assert np.all(make_excitation('impulse', area=5.0).sample(T) == 0.0)


def test_params_reject_non_numbers():
    with pytest.raises(ValueError, match='amplitude'):
        make_excitation('step', amplitude='big')
