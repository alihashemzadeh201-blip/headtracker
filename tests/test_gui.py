"""The calibration overlay's keyboard handling, without a display server.

``customtkinter`` needs Tk and a display, neither of which exists in CI.  The
stub below replaces the module with permissive widget classes that *record*
bindings, so the tests drive the real ``_on_key`` handler through the real
binding table instead of calling it by hand -- a test that called ``_on_key``
directly would still pass if ``bind("<Key>", ...)`` were deleted.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any, Optional

import pytest

from headtracker.calibration import MIN_SAMPLES_PER_POINT, CalibrationSession, grid_points

SCREEN = (1920, 1080)


class _FakeWidget:
    """Minimal stand-in for a Tk widget: records bindings, ignores the rest."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.bindings: dict[str, Any] = {}
        self.focus_taken = False
        self.attrs: list[Any] = []

    # -- the parts the overlay actually calls ---------------------------
    def bind(self, sequence: str, handler: Any, *_args: Any, **_kwargs: Any) -> None:
        self.bindings[sequence] = handler

    def attributes(self, *args: Any) -> None:
        self.attrs.append(args)

    def configure(self, **_kwargs: Any) -> None:
        return None

    def protocol(self, *_args: Any) -> None:
        return None

    def focus_set(self) -> None:
        self.focus_taken = True

    def pack(self, **_kwargs: Any) -> None:
        return None

    def winfo_width(self) -> int:
        return SCREEN[0]

    def winfo_height(self) -> int:
        return SCREEN[1]

    # -- test driver -----------------------------------------------------
    def press(self, keysym: str) -> None:
        """Deliver a key event through the binding the overlay registered."""
        assert "<Key>" in self.bindings, "the overlay never bound <Key>"
        self.bindings["<Key>"](types.SimpleNamespace(keysym=keysym))

    def press_escape(self) -> None:
        self.bindings["<Escape>"](types.SimpleNamespace(keysym="Escape"))


class _FakeCanvas(_FakeWidget):
    """A canvas that keeps every draw call, so text can be asserted on."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.drawn: list[tuple] = []

    def delete(self, *_args: Any) -> None:
        self.drawn.clear()

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> None:
            self.__dict__.setdefault("drawn", []).append((name, args, kwargs))

        return _record


class _CustomTkinterStub(  # pylint: disable=too-few-public-methods
    types.ModuleType
):
    """A module whose every attribute is a permissive widget class.

    Classes rather than callables, because ``gui`` subclasses several of them
    (``ctk.CTkToplevel``, ``ctk.CTk``) and a lambda cannot be a base class.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []  # marks it as a package
        # Names mirror the real customtkinter API on purpose.
        self.CTkToplevel = _FakeWidget  # pylint: disable=invalid-name
        self.CTkCanvas = _FakeCanvas  # pylint: disable=invalid-name
        self._cache: dict[str, Any] = {}

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        cache = self.__dict__.setdefault("_cache", {})
        if item not in cache:
            cache[item] = type(item, (_FakeWidget,), {})
        return cache[item]


@pytest.fixture(name="gui_module")
def _gui_module(monkeypatch):
    """Import ``headtracker.gui`` against the customtkinter stub."""
    monkeypatch.setitem(sys.modules, "customtkinter", _CustomTkinterStub("customtkinter"))
    sys.modules.pop("headtracker.gui", None)
    try:
        from headtracker import gui as module  # pylint: disable=import-outside-toplevel

        yield module
    finally:
        sys.modules.pop("headtracker.gui", None)


def make_session(countdown_s: float = 0.0, dwell_s: float = 5.0) -> CalibrationSession:
    """A 3x3 session already past its countdown."""
    session = CalibrationSession(
        grid_points(3, 3), screen=SCREEN, dwell_s=dwell_s, countdown_s=countdown_s
    )
    session.start(0.0)
    session.update(0.0)
    return session


def fill_point(session: CalibrationSession) -> None:
    for _ in range(MIN_SAMPLES_PER_POINT):
        session.add_sample(1.0, 1.0, distance=4000.0)


# --------------------------------------------------------------------------
# The overlay itself
# --------------------------------------------------------------------------
def test_a_keypress_advances_the_calibration_point(gui_module):
    """The whole point: the user is not held by the timer thirty times over."""
    calls: list[int] = []
    overlay = gui_module.CalibrationOverlay(
        None, make_session(), on_done=lambda: None, on_advance=lambda: calls.append(1)
    )

    overlay.press("space")
    overlay.press("Return")
    overlay.press("a")

    assert calls == [1, 1, 1], "every key should advance, not just Return"


def test_escape_still_cancels_instead_of_advancing(gui_module):
    """Escape is bound twice -- to <Escape> and to <Key>.  Cancel must win."""
    done: list[int] = []
    advanced: list[int] = []
    overlay = gui_module.CalibrationOverlay(
        None,
        make_session(),
        on_done=lambda: done.append(1),
        on_advance=lambda: advanced.append(1),
    )

    overlay.press("Escape")
    assert not advanced, "Escape must not be read as 'next point'"

    overlay.press_escape()
    assert done == [1]


def test_the_overlay_survives_a_keypress_without_an_advance_callback(gui_module):
    """``on_advance`` is optional; a caller that omits it must not crash."""
    overlay = gui_module.CalibrationOverlay(None, make_session(), on_done=lambda: None)
    overlay.press("space")  # would raise TypeError on a None callback


def test_the_overlay_takes_keyboard_focus(gui_module):
    """A Toplevel does not receive keys until it has focus."""
    overlay = gui_module.CalibrationOverlay(
        None, make_session(), on_done=lambda: None, on_advance=lambda: None
    )
    assert overlay.focus_taken is True


def test_the_on_screen_hint_mentions_the_key(gui_module):
    session = make_session()
    fill_point(session)
    overlay = gui_module.CalibrationOverlay(
        None, session, on_done=lambda: None, on_advance=lambda: None
    )
    overlay.render()

    labels = [kw.get("text", "") for _name, _args, kw in overlay.canvas.drawn]
    assert any("any key" in text for text in labels), labels


# --------------------------------------------------------------------------
# The application side of the wiring
# --------------------------------------------------------------------------
class _FakeOverlay:  # pylint: disable=too-few-public-methods
    """Counts redraws."""

    def __init__(self) -> None:
        self.rendered = 0

    def render(self, *_args: Any, **_kwargs: Any) -> None:
        self.rendered += 1


class _RecordingApp:  # pylint: disable=too-few-public-methods
    """Just enough of ``HeadTrackerApp`` to run its real advance method.

    The method is bound from the shipped class rather than reimplemented, so
    this exercises the code the user actually runs.
    """

    def __init__(self, session: CalibrationSession, advance_calibration: Any) -> None:
        self.session: Optional[CalibrationSession] = session
        self.overlay: Optional[_FakeOverlay] = _FakeOverlay()
        self.finished = 0
        self.advance_calibration = advance_calibration.__get__(self, _RecordingApp)

    def _finish_calibration(self) -> None:
        self.finished += 1


def test_advance_calibration_commits_the_point_and_redraws(gui_module):
    session = make_session()
    fill_point(session)
    app = _RecordingApp(session, gui_module.HeadTrackerApp.advance_calibration)

    app.advance_calibration()

    assert session.index == 1, "the point should have been committed"
    assert session.features[0] == (1.0, 1.0)
    assert app.finished == 0
    assert app.overlay.rendered == 1


def test_advance_calibration_finishes_on_the_last_point(gui_module):
    """The last point must tear the overlay down, not redraw an empty one.

    Each ``advance`` re-arms the countdown for the point it lands on, so this
    walks the session with one ``update`` (to leave the countdown) and one
    keypress per point.  That is deliberate: it means holding a key down cannot
    machine-gun through the grid without ever looking at a dot.
    """
    session = CalibrationSession(
        [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8)], screen=SCREEN, dwell_s=5.0, countdown_s=0.0
    )
    session.start(0.0)
    app = _RecordingApp(session, gui_module.HeadTrackerApp.advance_calibration)

    for _ in range(3):
        session.update(time.monotonic())  # countdown_s is 0, so this returns at once
        fill_point(session)
        app.advance_calibration()

    assert session.finished
    assert len(session.features) == 3
    assert app.finished == 1
    assert app.overlay.rendered == 2, "only the two intermediate points are redrawn"


def test_advance_calibration_does_nothing_without_a_session(gui_module):
    app = _RecordingApp(make_session(), gui_module.HeadTrackerApp.advance_calibration)
    app.session = None

    app.advance_calibration()  # must not raise

    assert app.finished == 0
    assert app.overlay.rendered == 0
