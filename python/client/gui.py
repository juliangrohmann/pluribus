import sys, ctypes, ctypes.wintypes as wt, time
from typing import List, Optional, Tuple

try:
  from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
  from PyQt6.QtGui import QPainter, QColor, QFont, QPen
  from PyQt6.QtWidgets import QApplication, QWidget
  QT_LIB = "PyQt6"
except ImportError:
  from PySide6.QtCore import Qt, QTimer, QRect, QPoint
  from PySide6.QtGui import QPainter, QColor, QFont, QPen
  from PySide6.QtWidgets import QApplication, QWidget
  QT_LIB = "PySide6"

user32 = ctypes.windll.user32
gdi32  = ctypes.windll.gdi32

FindWindowW        = user32.FindWindowW
FindWindowExW      = user32.FindWindowExW
GetWindowRect      = user32.GetWindowRect
GetWindowTextW     = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
IsWindowVisible    = user32.IsWindowVisible
EnumWindows        = user32.EnumWindows
SetWindowPos       = user32.SetWindowPos
GetWindowLongPtrW  = user32.GetWindowLongPtrW
SetWindowLongPtrW  = user32.SetWindowLongPtrW

GWL_EXSTYLE        = -20
WS_EX_LAYERED      = 0x00080000
WS_EX_TRANSPARENT  = 0x00000020
WS_EX_TOOLWINDOW   = 0x00000080
WS_EX_NOACTIVATE   = 0x08000000

SWP_NOSIZE         = 0x0001
SWP_NOMOVE         = 0x0002
SWP_NOACTIVATE     = 0x0010
SWP_SHOWWINDOW     = 0x0040
HWND_TOPMOST       = -1

RECT = wt.RECT
HWND = wt.HWND

def get_window_rect(hwnd: int) -> Optional[Tuple[int,int,int,int]]: return r.left, r.top, r.right, r.bottom if GetWindowRect(HWND(hwnd), ctypes.byref(r := RECT())) else None
def raise_topmost(hwnd: int) -> None: SetWindowPos(HWND(hwnd), HWND(HWND_TOPMOST), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
def make_click_through(hwnd: int) -> None:
  ex = GetWindowLongPtrW(HWND(hwnd), GWL_EXSTYLE)
  ex |= (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
  SetWindowLongPtrW(HWND(hwnd), GWL_EXSTYLE, ex)

class Overlay(QWidget):
  def __init__(self, hwnd_target: int, poll_ms: int = 50, parent=None):
    super().__init__(parent)
    self.hwnd_target = hwnd_target
    self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
    self._sync_to_target(force=True)
    self.show()
    make_click_through(int(self.winId()))
    raise_topmost(int(self.winId()))
    self.timer = QTimer(self)
    self.timer.setInterval(poll_ms)
    self.timer.timeout.connect(self._sync_to_target)
    self.timer.start()

    # example dynamic values
    self.example_pot      = "$141.03"
    self.example_stack    = "$2,450"
    self.example_equity   = "36%"
    self.example_timer_ms = 0

    self.anim = QTimer(self)
    self.anim.setInterval(200)
    self.anim.timeout.connect(self._tick)
    self.anim.start()

  def _tick(self):
    self.example_timer_ms += 200
    self.update()  # trigger repaint

  def _sync_to_target(self, force: bool = False):
    rect = get_window_rect(self.hwnd_target)
    if not rect:
      return
    x0, y0, x1, y1 = rect
    w, h = max(0, x1 - x0), max(0, y1 - y0)
    # Move/resize overlay to exactly cover the target window
    if force or (self.x() != x0 or self.y() != y0 or self.width() != w or self.height() != h):
      self.setGeometry(QRect(x0, y0, w, h))
      # keep topmost & click-through in case styles get reset by DWM changes
      make_click_through(int(self.winId()))
      raise_topmost(int(self.winId()))

  # ---- example HUD drawing ----
  def paintEvent(self, ev):
    p = QPainter(self)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    self._draw_badge(p, QPoint(12, 10), text=f"Pot {self.example_pot}")
    self._draw_badge(p, QPoint(self.width() - 12, 10), text=f"Stack {self.example_stack}", anchor="tr")
    self._draw_pill(p, QPoint(self.width() // 2, self.height() - 28), f"Equity {self.example_equity}")

    # Example small timer dot (animating alpha)
    alpha = 160 + int(95 * abs((self.example_timer_ms // 200) % 10 - 5) / 5)
    self._draw_dot(p, QPoint(self.width() // 2, 36), QColor(0, 180, 255, alpha))

    p.end()

  def _draw_badge(self, p: QPainter, pos: QPoint, text: str, anchor: str = "tl"):
    # rounded rectangle with subtle shadow
    pad_h, pad_v, r = 10, 6, 8
    font = QFont("Segoe UI", 11)
    p.setFont(font)
    metrics = p.fontMetrics()
    tw, th = metrics.horizontalAdvance(text), metrics.height()
    w = tw + pad_h * 2
    h = th + pad_v * 2

    if anchor == "tl":
      x, y = pos.x(), pos.y()
    elif anchor == "tr":
      x, y = pos.x() - w, pos.y()
    elif anchor == "bl":
      x, y = pos.x(), pos.y() - h
    else:  # center
      x, y = pos.x() - w // 2, pos.y() - h // 2

    # shadow
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 120))
    p.drawRoundedRect(x+2, y+2, w, h, r, r)
    # body
    p.setBrush(QColor(25, 25, 25, 180))
    p.drawRoundedRect(x, y, w, h, r, r)
    # text
    p.setPen(QColor(255, 255, 255, 230))
    p.drawText(QRect(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

  def _draw_pill(self, p: QPainter, center: QPoint, text: str):
    pad_h, pad_v, r = 12, 6, 12
    font = QFont("Segoe UI Semibold", 11)
    p.setFont(font)
    metrics = p.fontMetrics()
    tw, th = metrics.horizontalAdvance(text), metrics.height()
    w = tw + pad_h * 2
    h = th + pad_v * 2
    x = center.x() - w // 2
    y = center.y() - h // 2

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 105, 60, 175))
    p.drawRoundedRect(x, y, w, h, r, r)
    p.setPen(QColor(255, 255, 255, 235))
    p.drawText(QRect(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

  def _draw_dot(self, p: QPainter, center: QPoint, color: QColor, radius: int = 6):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawEllipse(center, radius, radius)
