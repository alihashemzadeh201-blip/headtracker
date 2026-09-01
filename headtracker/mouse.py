"""Cross-platform absolute cursor positioning.

The original controller used ``pyautogui.moveRel``.  Relative moves integrate
every per-frame error, so the cursor slowly drifts away from where the user is
looking and there is no way to recover short of re-centring.  Absolute moves
recompute the target from scratch every frame: a bad frame can be off by a
pixel, but it can never accumulate.

The backend talks to the window system directly where possible.  ``pyautogui``
is kept as a fallback, but each of its calls goes through a slow generic path
that becomes the bottleneck once tracking runs at 60 Hz.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

Point = Tuple[float, float]
IntPoint = Tuple[int, int]

BUTTONS = ("left", "right", "middle")


class MouseBackend:
    """Interface every backend implements."""

    name = "abstract"

    def screen_size(self) -> IntPoint:
        raise NotImplementedError

    def warp(self, x: int, y: int) -> None:
        """Move the cursor to integer pixel ``(x, y)``."""
        raise NotImplementedError

    def position(self) -> IntPoint:
        raise NotImplementedError

    def click(self, button: str = "left") -> None:
        raise NotImplementedError


class NullBackend(MouseBackend):
    """Records moves instead of performing them.  Used by the tests."""

    name = "null"

    def __init__(self, screen: IntPoint = (1920, 1080)) -> None:
        self._screen = screen
        self._position: IntPoint = (0, 0)
        self.moves: list = []
        self.clicks: list = []

    def screen_size(self) -> IntPoint:
        return self._screen

    def warp(self, x: int, y: int) -> None:
        self._position = (int(x), int(y))
        self.moves.append(self._position)

    def position(self) -> IntPoint:
        return self._position

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)


class WindowsBackend(MouseBackend):
    """``SendInput`` with absolute virtual-desktop coordinates."""

    name = "windows"

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_VIRTUALDESK = 0x4000
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
    INPUT_MOUSE = 0

    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

    def __init__(self) -> None:
        import ctypes  # pylint: disable=import-outside-toplevel
        from ctypes import wintypes  # pylint: disable=import-outside-toplevel

        self._ctypes = ctypes
        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]

        pointer = ctypes.POINTER(ctypes.c_ulong)

        class MouseInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
            """``MOUSEINPUT`` -- a single pointer event."""

            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", pointer),
            ]

        class KeyboardInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
            """``KEYBDINPUT`` -- unused, but the union needs it."""

            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", pointer),
            ]

        class HardwareInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
            """``HARDWAREINPUT`` -- unused, but the union needs it."""

            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class InputUnion(ctypes.Union):  # pylint: disable=too-few-public-methods
            """The ``INPUT`` payload, one of the three event kinds."""

            _fields_ = [
                ("mi", MouseInput),
                ("ki", KeyboardInput),
                ("hi", HardwareInput),
            ]

        class Input(ctypes.Structure):  # pylint: disable=too-few-public-methods
            """``INPUT`` -- what ``SendInput`` consumes."""

            _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]

        self._Input = Input  # pylint: disable=invalid-name
        self._MouseInput = MouseInput  # pylint: disable=invalid-name
        self._origin = (
            self._user32.GetSystemMetrics(self.SM_XVIRTUALSCREEN),
            self._user32.GetSystemMetrics(self.SM_YVIRTUALSCREEN),
        )
        self._span = (
            max(self._user32.GetSystemMetrics(self.SM_CXVIRTUALSCREEN), 1),
            max(self._user32.GetSystemMetrics(self.SM_CYVIRTUALSCREEN), 1),
        )

    def screen_size(self) -> IntPoint:
        return (self._span[0], self._span[1])

    def _send(self, flags: int, x: int = 0, y: int = 0) -> None:
        event = self._Input()
        # `type` is the name of the Win32 struct field, not a Python builtin use.
        event.type = self.INPUT_MOUSE  # pylint: disable=attribute-defined-outside-init
        event.union.mi.dx = x
        event.union.mi.dy = y
        event.union.mi.mouseData = 0
        event.union.mi.dwFlags = flags
        event.union.mi.time = 0
        event.union.mi.dwExtraInfo = None
        self._user32.SendInput(1, self._ctypes.byref(event), self._ctypes.sizeof(event))

    def warp(self, x: int, y: int) -> None:
        # Absolute mode maps the whole virtual desktop onto 0..65535.
        normalised_x = int((x - self._origin[0]) * 65535 / max(self._span[0] - 1, 1))
        normalised_y = int((y - self._origin[1]) * 65535 / max(self._span[1] - 1, 1))
        flags = (
            self.MOUSEEVENTF_MOVE
            | self.MOUSEEVENTF_ABSOLUTE
            | self.MOUSEEVENTF_VIRTUALDESK
        )
        self._send(flags, normalised_x, normalised_y)

    def position(self) -> IntPoint:
        class _POINT(self._ctypes.Structure):  # pylint: disable=too-few-public-methods
            _fields_ = [("x", self._ctypes.c_long), ("y", self._ctypes.c_long)]

        point = _POINT()
        self._user32.GetCursorPos(self._ctypes.byref(point))
        return (int(point.x), int(point.y))

    def click(self, button: str = "left") -> None:
        pairs = {
            "left": (self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP),
            "right": (self.MOUSEEVENTF_RIGHTDOWN, self.MOUSEEVENTF_RIGHTUP),
            "middle": (self.MOUSEEVENTF_MIDDLEDOWN, self.MOUSEEVENTF_MIDDLEUP),
        }
        down, up = pairs.get(button, pairs["left"])
        self._send(down)
        self._send(up)


class X11Backend(MouseBackend):
    """X11 pointer warping through python-xlib."""

    name = "x11"

    def __init__(self) -> None:
        from Xlib import X, display  # pylint: disable=import-outside-toplevel
        from Xlib.ext import xtest  # pylint: disable=import-outside-toplevel

        self._button = X  # Xlib's constants module, not a class
        self._xtest = xtest
        self._display = display.Display()
        self._root = self._display.screen().root

    def screen_size(self) -> IntPoint:
        geometry = self._root.get_geometry()
        return (int(geometry.width), int(geometry.height))

    def warp(self, x: int, y: int) -> None:
        self._root.warp_pointer(int(x), int(y))
        self._display.sync()

    def position(self) -> IntPoint:
        data = self._root.query_pointer()
        return (int(data.root_x), int(data.root_y))

    def click(self, button: str = "left") -> None:
        number = {"left": 1, "middle": 2, "right": 3}.get(button, 1)
        self._xtest.fake_input(self._display, self._button.ButtonPress, number)
        self._xtest.fake_input(self._display, self._button.ButtonRelease, number)
        self._display.sync()


class QuartzBackend(MouseBackend):
    """macOS cursor warping through Quartz."""

    name = "quartz"

    def __init__(self) -> None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from Quartz.CoreGraphics import (
            CGEventCreateMouseEvent,
            CGEventPost,
            CGMainDisplayID,
            CGDisplayPixelsWide,
            CGDisplayPixelsHigh,
            CGPointMake,
            kCGEventLeftMouseDown,
            kCGEventLeftMouseUp,
            kCGEventRightMouseDown,
            kCGEventRightMouseUp,
            kCGEventOtherMouseDown,
            kCGEventOtherMouseUp,
            kCGHIDEventTap,
        )

        self._q = {
            "post": CGEventPost,
            "create": CGEventCreateMouseEvent,
            "point": CGPointMake,
            "tap": kCGHIDEventTap,
            "down": {
                "left": kCGEventLeftMouseDown,
                "right": kCGEventRightMouseDown,
                "middle": kCGEventOtherMouseDown,
            },
            "up": {
                "left": kCGEventLeftMouseUp,
                "right": kCGEventRightMouseUp,
                "middle": kCGEventOtherMouseUp,
            },
        }
        display_id = CGMainDisplayID()
        self._screen = (int(CGDisplayPixelsWide(display_id)), int(CGDisplayPixelsHigh(display_id)))

    def screen_size(self) -> IntPoint:
        return self._screen

    def warp(self, x: int, y: int) -> None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from Quartz.CoreGraphics import CGWarpMouseCursorPosition

        CGWarpMouseCursorPosition(self._q["point"](float(x), float(y)))

    def position(self) -> IntPoint:
        # pylint: disable-next=import-outside-toplevel,import-error
        from Quartz.CoreGraphics import CGEventGetLocation, CGEventCreate

        location = CGEventGetLocation(CGEventCreate(None))
        return (int(location.x), int(location.y))

    def click(self, button: str = "left") -> None:
        point = self._q["point"](0.0, 0.0)
        self._q["post"](self._q["tap"], self._q["create"](None, self._q["down"][button], point, 0))
        self._q["post"](self._q["tap"], self._q["create"](None, self._q["up"][button], point, 0))


class PyAutoGuiBackend(MouseBackend):
    """Slowest but always available fallback."""

    name = "pyautogui"

    def __init__(self) -> None:
        import pyautogui  # pylint: disable=import-outside-toplevel

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self._pyautogui = pyautogui

    def screen_size(self) -> IntPoint:
        return tuple(self._pyautogui.size())  # type: ignore[return-value]

    def warp(self, x: int, y: int) -> None:
        self._pyautogui.moveTo(int(x), int(y), _pause=False)

    def position(self) -> IntPoint:
        return tuple(self._pyautogui.position())  # type: ignore[return-value]

    def click(self, button: str = "left") -> None:
        self._pyautogui.click(button=button, _pause=False)


def create_backend(preferred: Optional[str] = None) -> MouseBackend:
    """Pick the fastest backend the current platform supports."""
    if preferred == "null":
        return NullBackend()

    candidates = []
    if sys.platform.startswith("win"):
        candidates = [WindowsBackend, PyAutoGuiBackend]
    elif sys.platform == "darwin":
        candidates = [QuartzBackend, PyAutoGuiBackend]
    else:
        candidates = [X11Backend, PyAutoGuiBackend]

    errors = []
    for candidate in candidates:
        try:
            return candidate()
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{candidate.__name__}: {exc}")
    raise RuntimeError("no usable mouse backend: " + "; ".join(errors))


class AbsoluteMouse:
    """Drives the cursor to absolute, sub-pixel targets.

    Targets are kept as floats and only rounded when they reach the window
    system.  Because every frame recomputes the target from the current gaze,
    the rounding never accumulates the way it does with relative moves -- but
    keeping the float still avoids a visible half-pixel staircase when the gaze
    moves slowly.
    """

    def __init__(self, backend: Optional[MouseBackend] = None) -> None:
        self.backend = backend if backend is not None else create_backend()
        self._screen: IntPoint = self.backend.screen_size()
        self._target: Point = (self._screen[0] / 2.0, self._screen[1] / 2.0)
        self._last_sent: Optional[IntPoint] = None

    @property
    def screen(self) -> IntPoint:
        return self._screen

    def set_screen(self, size: IntPoint) -> None:
        self._screen = (int(size[0]), int(size[1]))

    def target(self) -> Point:
        """The last requested position, as a float."""
        return self._target

    def position(self) -> IntPoint:
        return self.backend.position()

    def move_to(self, x: float, y: float) -> bool:
        """Clamp to the screen and warp.  Returns ``True`` if the OS was called.

        Consecutive frames often round to the same pixel; skipping those warps
        keeps the window system out of the hot loop.
        """
        width, height = self._screen
        clamped = (
            min(max(float(x), 0.0), max(width - 1.0, 0.0)),
            min(max(float(y), 0.0), max(height - 1.0, 0.0)),
        )
        self._target = clamped
        rounded = (int(round(clamped[0])), int(round(clamped[1])))
        if rounded == self._last_sent:
            return False
        self.backend.warp(*rounded)
        self._last_sent = rounded
        return True

    def click(self, button: str = "left") -> None:
        if button not in BUTTONS:
            raise ValueError(f"unknown button {button!r}")
        self.backend.click(button)
