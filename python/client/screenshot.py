import ctypes
import threading
from typing import Optional, Tuple
from PIL import Image
import win32api
import win32con
import win32gui
import win32ui

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

CAPTUREBLT = 0x40000000
BI_RGB = 0
DIB_RGB_COLORS = 0

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_ushort),
        ("biBitCount", ctypes.c_ushort),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]

class RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", ctypes.c_ubyte),
        ("rgbGreen", ctypes.c_ubyte),
        ("rgbRed", ctypes.c_ubyte),
        ("rgbReserved", ctypes.c_ubyte),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", RGBQUAD * 1),
    ]

CreateDIBSection = ctypes.windll.gdi32.CreateDIBSection
CreateDIBSection.argtypes = [
    ctypes.c_void_p,             # HDC
    ctypes.POINTER(BITMAPINFO),  # BITMAPINFO*
    ctypes.c_uint,               # UINT (DIB color table usage)
    ctypes.POINTER(ctypes.c_void_p), # void** ppvBits
    ctypes.c_void_p,             # HANDLE hSection
    ctypes.c_uint32              # DWORD offset
]
CreateDIBSection.restype = ctypes.c_void_p

class _CachedDesktopShot:
    def __init__(self):
        self._lock = threading.Lock()
        self._dpi_inited = False
        self._hbmp = None
        self._bits_ptr = None
        self._buf = None
        self._stride = 0
        self._rect = (0, 0, 0, 0)

        self._hdesk = None
        self._desktop_dc = None
        self._srcdc = None
        self._memdc = None

        self._ensure_dpi_awareness()
        self._ensure_resources()

    def _ensure_dpi_awareness(self):
        if self._dpi_inited:
            return
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        self._dpi_inited = True

    def _current_virtual_rect(self) -> Tuple[int, int, int, int]:
        left   = win32api.GetSystemMetrics(SM_XVIRTUALSCREEN)
        top    = win32api.GetSystemMetrics(SM_YVIRTUALSCREEN)
        width  = win32api.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        return left, top, width, height

    def _free_resources(self):
        if self._memdc: self._memdc.DeleteDC()
        if self._srcdc: self._srcdc.DeleteDC()
        if self._desktop_dc and self._hdesk:
            win32gui.ReleaseDC(self._hdesk, self._desktop_dc)
        if self._hbmp: win32gui.DeleteObject(self._hbmp)
        self._memdc = self._srcdc = self._desktop_dc = self._hdesk = self._hbmp = None
        self._buf = None

    def _ensure_resources(self):
        left, top, width, height = self._current_virtual_rect()
        need_rebuild = self._hbmp is None or (self._rect[2], self._rect[3]) != (width, height)
        if not need_rebuild:
            return

        self._free_resources()
        self._rect = (left, top, width, height)

        self._hdesk = win32gui.GetDesktopWindow()
        self._desktop_dc = win32gui.GetWindowDC(self._hdesk)
        self._srcdc = win32ui.CreateDCFromHandle(self._desktop_dc)
        self._memdc = self._srcdc.CreateCompatibleDC()

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        bmi.bmiHeader.biSizeImage = 0

        bits_ptr = ctypes.c_void_p()
        hbmp = CreateDIBSection(self._srcdc.GetSafeHdc(), ctypes.byref(bmi),
                                DIB_RGB_COLORS, ctypes.byref(bits_ptr),
                                None, 0)

        win32gui.SelectObject(self._memdc.GetSafeHdc(), hbmp)

        stride = ((width * 32 + 31) // 32) * 4
        buf_size = stride * height
        buf = (ctypes.c_ubyte * buf_size).from_address(bits_ptr.value)

        self._hbmp = hbmp
        self._bits_ptr = bits_ptr
        self._stride = stride
        self._buf = buf

    def capture(self) -> Image.Image:
        with self._lock:
            self._ensure_resources()
            left, top, width, height = self._rect
            self._memdc.BitBlt(
                (0, 0), (width, height),
                self._srcdc, (left, top),
                win32con.SRCCOPY | CAPTUREBLT
            )
            img = Image.frombuffer("RGB", (width, height),
                                   self._buf, "raw", "BGRX", self._stride, 1)
            return img.copy()  # detach buffer

_cached: Optional[_CachedDesktopShot] = None
_singleton_lock = threading.Lock()

def screenshot_all() -> Image.Image:
    global _cached
    with _singleton_lock:
        if _cached is None:
            _cached = _CachedDesktopShot()
        return _cached.capture()

if __name__ == "__main__":
    img = screenshot_all()
    img.save("screenshot_all.png")
