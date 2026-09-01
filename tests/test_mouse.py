"""Absolute cursor positioning logic."""

from __future__ import annotations

import pytest

from headtracker.mouse import AbsoluteMouse, NullBackend, create_backend


@pytest.fixture(name="backend")
def fixture_backend() -> NullBackend:
    return NullBackend((1920, 1080))


@pytest.fixture(name="mouse")
def fixture_mouse(backend: NullBackend) -> AbsoluteMouse:
    return AbsoluteMouse(backend)


def test_screen_size_comes_from_the_backend(mouse):
    assert mouse.screen == (1920, 1080)


def test_move_to_warps_the_backend(mouse, backend):
    assert mouse.move_to(100.0, 200.0) is True
    assert backend.position() == (100, 200)


def test_float_target_is_kept_but_rounded_for_the_os(mouse, backend):
    """The OS cursor is integral, but the requested position stays fractional.

    Keeping the float matters because the target is recomputed from the gaze
    every frame; rounding only at the last step avoids a half-pixel staircase
    during slow movement.
    """
    mouse.move_to(100.4, 200.6)
    assert backend.position() == (100, 201)
    assert mouse.target() == (100.4, 200.6)


def test_identical_rounded_targets_skip_the_syscall(mouse, backend):
    """Consecutive frames usually land on the same pixel; do not re-warp."""
    assert mouse.move_to(100.2, 200.1) is True
    assert mouse.move_to(100.4, 200.3) is False
    assert len(backend.moves) == 1


def test_a_crossed_pixel_boundary_does_warp(mouse, backend):
    mouse.move_to(100.4, 200.0)
    assert mouse.move_to(100.6, 200.0) is True
    assert backend.position() == (101, 200)


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((-500.0, 300.0), (0, 300)),
        ((5000.0, 300.0), (1919, 300)),
        ((100.0, -1.0), (100, 0)),
        ((100.0, 9999.0), (100, 1079)),
    ],
)
def test_targets_are_clamped_to_the_screen(mouse, backend, requested, expected):
    mouse.move_to(*requested)
    assert backend.position() == expected


def test_click_is_forwarded_and_validated(mouse, backend):
    mouse.click("left")
    mouse.click("right")
    assert backend.clicks == ["left", "right"]
    with pytest.raises(ValueError, match="unknown button"):
        mouse.click("pinky")


def test_set_screen_updates_clamping(mouse, backend):
    mouse.set_screen((800, 600))
    mouse.move_to(5000.0, 5000.0)
    assert backend.position() == (799, 599)


def test_create_backend_can_produce_a_null_backend():
    backend = create_backend("null")
    assert backend.name == "null"


def test_abstract_backend_is_not_usable():
    from headtracker.mouse import MouseBackend  # pylint: disable=import-outside-toplevel

    with pytest.raises(NotImplementedError):
        MouseBackend().warp(0, 0)
