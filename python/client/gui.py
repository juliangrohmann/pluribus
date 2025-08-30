import sys, ctypes, ctypes.wintypes as wt, time
from typing import List, Optional, Tuple
from PyQt6.QtCore import Qt, QObject, QTimer, QRect, QPoint, pyqtSlot, pyqtSignal as Signal
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import QApplication, QWidget
import win32gui
from config import GuiLayout

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

def raise_topmost(hwnd: int) -> None: SetWindowPos(HWND(hwnd), HWND(HWND_TOPMOST), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
def make_click_through(hwnd: int) -> None:
  ex = GetWindowLongPtrW(HWND(hwnd), GWL_EXSTYLE)
  ex |= (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
  SetWindowLongPtrW(HWND(hwnd), GWL_EXSTYLE, ex)

class Bus(QObject):
  set_actions = Signal(list)
  set_freq = Signal(list)

class Overlay(QWidget):
  def __init__(self, hwnd_target:int, layout:GuiLayout):
    super().__init__(None)
    self.hwnd_target = hwnd_target
    self.layout = layout
    self.bus = Bus()
    self.bus.set_actions.connect(self.set_actions)
    self.bus.set_freq.connect(self.set_freq)
    self.last_rect = None
    self.actions: List[str] = ["Fold", "Check/Call", "Bet 33%", "Bet 50%", "Bet 75%", "Bet 100%", "Bet 125%", "Bet 150%", "Bet 200%", "All-in"]
    self.freq: List[float] = [0.8232, 0.1213, 0.0500, 0.1200, 0.4232, 0.4800, 0.0320, 0.0120, 0.9832, 0.3213]
    self.rng: int = 3241
    self.chosen_idx: int = 2
    self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
    self._sync_to_target(force=True)
    self.show()
    make_click_through(int(self.winId()))
    raise_topmost(int(self.winId()))
    self.anim = QTimer(self)
    self.anim.setInterval(200)
    self.anim.timeout.connect(self._tick)
    self.anim.start()

  def _tick(self):
    self._sync_to_target(False)
    self.update()

  def set_actions(self, actions:List[str]):
    self.actions = actions
    self.update()

  def set_freq(self, freq:List[float]):
    self.freq = freq
    self.update()

  def _sync_to_target(self, force:bool=False):
    self.last_rect = win32gui.GetWindowRect(self.hwnd_target)
    if not self.last_rect: return
    x0, y0, x1, y1 = self.last_rect

    w, h = max(0, x1 - x0), max(0, y1 - y0)
    # Move/resize overlay to exactly cover the target window
    if force or (self.x() != x0 or self.y() != y0 or self.width() != w or self.height() != h):
      print("Update")
      self.setGeometry(QRect(x0, y0, w, h))
      # keep topmost & click-through in case styles get reset by DWM changes
      make_click_through(int(self.winId()))
      raise_topmost(int(self.winId()))

  def paintEvent(self, ev):
    p = QPainter(self)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    w, h = self.last_rect[2] - self.last_rect[0], self.last_rect[3] - self.last_rect[1]
    x_min, y_min, x_max, y_max = [int(b * r) for b,r in zip(self.layout.action_bounds, (w,h,w,h))]
    x, y = x_min, y_min
    freq_h = 20
    for i,(action,frequency) in enumerate(zip(self.actions,self.freq)):
      x_end, y_end = self._draw_label(p, QPoint(x, y), action, i == self.chosen_idx)
      self._draw_label(p, QPoint(x, y_end), f"{frequency:.2f}", False, w=x_end-x, h=freq_h, font_size=10)
      x = x_end
      if x > x_max: x, y = x_min, y_end + 30
    # self._draw_label(p, QPoint(12, 10), text=self.actions[0])
    # self._draw_label(p, QPoint(self.width() - 12, 10), text=self.actions[1], anchor="tr")
    p.end()

  def _draw_label(self, p:QPainter, pos:QPoint, text:str, highlight:bool, w:int=None, h:int=None, font_size:int=11, anchor:str="tl"):
    p.setPen(QColor(255, 255, 255, 230))
    # rounded rectangle with subtle shadow
    pad_h, pad_v, r = 10, 6, 8
    font = QFont("Segoe UI", font_size)
    p.setFont(font)
    metrics = p.fontMetrics()
    tw, th = metrics.horizontalAdvance(text), metrics.height()
    w = tw + pad_h * 2 if w is None else w
    h = th + pad_v * 2 if h is None else h

    if anchor == "tl": x, y = pos.x(), pos.y()
    elif anchor == "tr": x, y = pos.x() - w, pos.y()
    elif anchor == "bl": x, y = pos.x(), pos.y() - h
    else: x, y = pos.x() - w // 2, pos.y() - h // 2 # center

    # body
    p.setBrush(QColor(0, 130, 0, 180) if highlight else QColor(25, 25, 25, 180))
    p.drawRect(x, y, w, h)
    # text
    p.drawText(QRect(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)
    return x + w, y + h

  def _draw_dot(self, p: QPainter, center: QPoint, color: QColor, radius: int = 6):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawEllipse(center, radius, radius)

class OverlayManager(QObject):
  create_overlay = Signal(int, GuiLayout)
  destroy_overlay = Signal(int)
  set_actions = Signal(int, list)
  set_freq = Signal(int, list)

  def __init__(self):
    super().__init__()
    self._overlays: Dict[int, Overlay] = {}
    self.create_overlay.connect(self._on_create)
    self.destroy_overlay.connect(self._on_destroy)
    self.set_actions.connect(self._on_set_actions)
    self.set_freq.connect(self._on_set_freq)

  @pyqtSlot(int, GuiLayout)
  def _on_create(self, hwnd:int, layout:GuiLayout):
    if hwnd not in self._overlays:
      self._overlays[hwnd] = Overlay(hwnd, layout)

  @pyqtSlot(int)
  def _on_destroy(self, hwnd:int):
    w = self._overlays.pop(hwnd, None)
    if w: w.close()

  @pyqtSlot(int, list)
  def _on_set_actions(self, hwnd:int, xs:List[str]):
    w = self._overlays.get(hwnd)
    if w: w.bus.set_actions.emit(xs)

  @pyqtSlot(int, list)
  def _on_set_freq(self, hwnd:int, xs:List[float]):
    w = self._overlays.get(hwnd)
    if w: w.bus.set_freq.emit(xs)