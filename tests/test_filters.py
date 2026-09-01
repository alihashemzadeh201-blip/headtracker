"""Smoothing and glitch rejection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from headtracker.filters import GlitchGate, LowPassFilter, OneEuroFilter


def test_one_euro_passes_the_first_sample_unchanged():
    filt = OneEuroFilter()
    assert filt.filter(42.0, 0.0) == 42.0


def test_one_euro_converges_on_a_step():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    value = 0.0
    for index in range(400):
        value = filt.filter(100.0, index / 60.0)
    assert value == pytest.approx(100.0, abs=0.5)


def test_one_euro_reduces_jitter_on_a_still_signal():
    rng = np.random.default_rng(3)
    noisy = 50.0 + rng.normal(0.0, 3.0, 600)

    filt = OneEuroFilter(min_cutoff=0.8, beta=0.0)
    smoothed = np.array([filt.filter(v, i / 60.0) for i, v in enumerate(noisy)])[50:]

    assert float(smoothed.std()) < float(noisy[50:].std()) / 2.0
    assert float(smoothed.mean()) == pytest.approx(50.0, abs=1.5)


def test_higher_beta_reaches_a_step_faster():
    """The whole point of One Euro: open up when the signal actually moves."""
    signal = [0.0] * 60 + [40.0] * 60

    def settle_time(filt):
        out = [filt.filter(v, i / 60.0) for i, v in enumerate(signal)]
        return next(i for i in range(60, 120) if out[i] >= 36.0) - 60

    slow = settle_time(OneEuroFilter(min_cutoff=0.5, beta=0.0))
    fast = settle_time(OneEuroFilter(min_cutoff=0.5, beta=0.5))
    assert fast < slow


def test_one_euro_reset_forgets_history():
    filt = OneEuroFilter()
    for index in range(50):
        filt.filter(10.0, index / 60.0)
    filt.reset()
    assert filt.filter(-90.0, 1.0) == -90.0


def test_one_euro_handles_a_repeated_timestamp():
    """A zero interval must not divide by zero or explode."""
    filt = OneEuroFilter()
    filt.filter(1.0, 5.0)
    value = filt.filter(2.0, 5.0)
    assert math.isfinite(value)


def test_low_pass_smoother_tracks_a_constant():
    filt = LowPassFilter(cutoff_hz=5.0)
    value = 0.0
    for index in range(300):
        value = filt.filter(7.0, index / 60.0)
    assert value == pytest.approx(7.0, abs=0.01)
    assert filt.value == pytest.approx(7.0, abs=0.01)
    filt.reset()
    assert filt.value is None


def test_glitch_gate_accepts_the_first_sample():
    gate = GlitchGate(max_speed=1000.0)
    assert gate.check(10.0, 10.0, 0.0) == (10.0, 10.0)


def test_glitch_gate_drops_a_teleport():
    gate = GlitchGate(max_speed=2000.0, recovery_frames=2)
    gate.check(0.0, 0.0, 0.0)
    # 1500 px in 16 ms is ~94000 px/s: a tracking failure, not a real glance.
    assert gate.check(1500.0, 0.0, 1 / 60.0) is None


def test_glitch_gate_accepts_a_fast_but_plausible_glance():
    gate = GlitchGate(max_speed=9000.0)
    gate.check(0.0, 0.0, 0.0)
    assert gate.check(120.0, 0.0, 1 / 60.0) == (120.0, 0.0)


def test_glitch_gate_recovers_after_a_sustained_change():
    """Several 'glitches' in a row mean the user really did move."""
    gate = GlitchGate(max_speed=2000.0, recovery_frames=1)
    gate.check(0.0, 0.0, 0.0)
    assert gate.check(1500.0, 0.0, 1 / 60.0) is None
    assert gate.check(1500.0, 0.0, 2 / 60.0) == (1500.0, 0.0)


def test_glitch_gate_reset_forgets_the_reference():
    gate = GlitchGate(max_speed=100.0)
    gate.check(0.0, 0.0, 0.0)
    gate.reset()
    assert gate.check(9000.0, 9000.0, 1 / 60.0) == (9000.0, 9000.0)
