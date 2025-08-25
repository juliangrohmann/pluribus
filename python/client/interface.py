import random
import re
import time
import pyautogui
import win32gui
import win32con
import PIL
import numpy as np
import config
import ocr
from typing import Tuple, List, Callable
from collections import Counter
from colorama import just_fix_windows_console
from PIL import ImageOps, ImageEnhance, Image
from PIL.Image import Image
from screenshot import screenshot_all
from util import round_to_str, colorize_cards

random.seed()
win_default_offset = 8
win_header_offset = 1
win_header_h = 30
false_usernames = ["Bet", "Raise", "Check", "Call", "Fold", "Post", "Post BB", "Post SB", "Muck", "Show", "Resume"]
debug = False

def _build_card_mask() -> np.ndarray:
  with open(ocr.charset_path, "r", encoding="utf-8") as f:
    charset = [line.rstrip("\n") for line in f]
  return np.array([i == 0 or c in "1234567890JKQA" for i,c in enumerate(charset)])
card_mask = _build_card_mask()

def _assert_player_pos(i:int) -> None:
  assert 0 <= i <= 5, f"Invalid Player Position: {player_pos}"

def _frac_to_pix(x:float, y:float, img_w:int, img_h:int) -> Tuple[int,int]:
  return int(round(x * img_w)), int(round(y * (img_h - win_header_h) + win_header_h))

def _crop(img:Image, box:Tuple[int,int,int,int]) -> Image:
  return img.crop((*_frac_to_pix(box[0], box[1], img.width, img.height), *_frac_to_pix(box[2], box[3], img.width, img.height)))

def _apply_color_filter(pos:int, img:Image, coords:Tuple[Tuple[float,float], ...], fun:Callable[[Tuple[int,int,int]], bool]) -> bool:
  _assert_player_pos(pos)
  return fun(img.getpixel(_frac_to_pix(*coords[pos], img.width, img.height)))

def _parse_ocr(img:Image, coords:Tuple[int,int,int,int], repl:Tuple[str, ...]=tuple(), refine:bool=False, expand:int=4, mask:np.ndarray|None=None, debug_label:str=None) -> str:
  raw = ocr.pipeline.run(cropped := _crop(img, coords), refine=refine, expand=expand, mask=mask)[0]
  if debug and debug_label is not None:
    cropped.save(f"img_debug\\{debug_label}.png")
  for s in repl: raw = raw.replace(s, '')
  return raw

def _parse_rank(img:Image, coords:Tuple[int,int,int,int], n:int=8, debug_label:str=None) -> str|None:
  cropped = _crop(img, coords)
  repeated = PIL.Image.new(cropped.mode, (cropped.width * n, cropped.height))
  for i in range(n): repeated.paste(cropped, (i * cropped.width, 0))
  counter = Counter(ocr.pipeline.run(repeated, expand=10, mask=card_mask)[0])
  if debug and debug_label is not None:
    repeated.save(f"img_debug\\{debug_label}.png")
  return 'T' if set(e[0] for e in counter.most_common(2)) == set('10') else {'0':'Q', '1':'T'}.get(most[0][0], most[0][0]) if len(most := counter.most_common(1)) else None

class PokerInterface:
  def __init__(self, wnd_handle:int, conf:dict, debug:bool=False) -> None:
    self.handle: int = wnd_handle
    self.config: config.PokerConfig = conf
    self.debug: bool = debug

  def rect(self) -> Tuple[int,int,int,int]:
    r = win32gui.GetWindowRect(self.handle)
    return r[0] + win_default_offset, r[1] + win_header_offset, r[2] - win_default_offset, r[3] - win_default_offset

  def screenshot(self) -> Image:
    return screenshot_all().crop(self.rect())

  def blinds(self, blinds:bool=False) -> Tuple[float, ...]:
    stakes = self.config.site.get_blinds(self.title())
    return tuple(v / stakes[1] for v in stakes) if blinds else stakes

  def hand(self, img:Image, pos:int) -> str|None:
    _assert_player_pos(pos)
    return hand_str if len(hand_str:=self._parse_cards(img,self.config.hole_cards_ranks[pos],self.config.hole_cards_suits[pos],debug_label=f"hole_cards_{pos}_rank")) == 4 else None

  def is_showing_hand(self, img:Image, pos:int) -> bool:
    return all(self._parse_suit(img, self.config.hole_cards_suits[pos][i]) is not None for i in range(2))

  def username(self, img:Image, pos:int) -> str|None:
    _assert_player_pos(pos)
    name = _parse_ocr(img, self.config.usernames[pos], tuple(), debug_label=f"username_{pos}")
    return name if name not in false_usernames and not name.startswith("Won ") and not name.startswith("Time ") else None

  def round(self, img:Image) -> int: return sum(bool(self._parse_suit(img, self.config.site.board_suits[i])) for i in (0, 3, 4))
  def board(self, img:Image) -> str|None: return self._parse_cards(img, self.config.site.board_ranks, self.config.site.board_suits, debug_label=f"board_rank")
  def stack_size(self, img:Image, pos:int, blinds:bool=False) -> float|None: return self._cash_by_pos(pos, img, self.config.stacks, blinds=blinds, debug_label=f"stack_size_{pos}")
  def bet_size(self, img:Image, pos:int, blinds:bool=False) -> float|None: return self._cash_by_pos(pos, img, self.config.bet_size, refine=True, blinds=blinds, debug_label=f"bet_size_{pos}")
  def pot_size(self, img:Image, blinds:bool=False) -> float|None: return self._cash(img, self.config.site.pot, ("Pot", " ", ":"), blinds, debug_label="pot_size")
  def active_seat(self, img:Image) -> int|None: return self._find_pos(img, self.config.active, self.config.site.is_active)
  def button_pos(self, img:Image) -> int|None: return self._find_pos(img, self.config.button, self.config.site.has_button)
  def is_seat_open(self, img:Image, pos:int) -> bool: return _apply_color_filter(pos, img, self.config.seats, self.config.site.is_seat_open)
  def is_sitting_out(self, img:Image, pos:int) -> bool: return (label:=_parse_ocr(img, self.config.stacks[pos], debug_label=f"sitout_{pos}")) == "Sitting Out" or label == "Waiting"
  def has_cards(self, img:Image, pos:int) -> bool: return _apply_color_filter(pos, img, self.config.cards, self.config.site.has_cards)
  def has_bet(self, img:Image, pos:int) -> bool: return _apply_color_filter(pos, img, self.config.bet_chips, self.config.site.has_bet)
  def title(self) -> str: return win32gui.GetWindowText(self.handle)
  def ante(self, blinds:bool=False) -> float: return self.config.site.get_ante(self.title()) / (self.blinds()[1] if blinds else 1.0)
  def move(self, x:int, y:int, w:int, h:int) -> None: win32gui.SetWindowPos(self.handle, win32con.HWND_TOP, x, y, w, h, 0)
  def __repr__(self) -> str: return f"<PokerTable title=\"{self.title()}\", config=\"{self.config.name}\", handle={self.handle}>"
  def __str__(self) -> str: return self.__repr__()

  def _find_pos(self, img:Image, coords:Tuple[Tuple[float,float], ...], fun:Callable[[Tuple[int,int,int]], bool]) -> int|None:
    for pos in range(self.config.n_players):
      if _apply_color_filter(pos, img, coords, fun): return pos
    return None

  def _cash(self, img:Image, coords:Tuple[int,int,int,int], repl:Tuple[str, ...]=tuple(), refine:bool=False, blinds:bool=False, debug_label:str=None) -> float|None:
    val = m.group(1) if (m:=re.match(r"(?:.*\$)?\D*(\d+\.?\d{0,2}).*", raw := _parse_ocr(img, coords, repl + (",",), refine=refine, debug_label=debug_label))) is not None else None
    return (float(val) / self.blinds()[1] if blinds else float(val)) if m is not None else 0.0 if raw.lower() == "all in" else None

  def _cash_by_pos(self, pos:int, img:Image, coords_by_pos:Tuple[Tuple[int,int,int,int], ...], refine:bool=False, blinds:bool=False, debug_label:str=None) -> float|None:
    _assert_player_pos(pos)
    return self._cash(img, coords_by_pos[pos], refine=refine, blinds=blinds, debug_label=debug_label)

  def _rgb_to_suit(self, rgb:Tuple[int,int,int]) -> str|None:
    if self.config.site.is_heart(rgb): return 'h'
    if self.config.site.is_club(rgb): return 'c'
    if self.config.site.is_diamond(rgb): return 'd'
    if self.config.site.is_spade(rgb): return 's'
    return None

  def _parse_suit(self, img:Image, coords:Tuple[int,int]) -> str|None:
    return self._rgb_to_suit(img.getpixel(_frac_to_pix(*coords, *img.size)))

  def _parse_cards(self, img:Image, rank_coords:Tuple[Tuple[int, int, int, int], ...], suit_coords:Tuple[Tuple[int,int], ...], debug_label:str=None) -> str|None:
    return ''.join(rank + suit for i in range(len(rank_coords))
                   if (suit := self._parse_suit(img, suit_coords[i])) is not None and (rank := _parse_rank(img, rank_coords[i], debug_label=f"{debug_label}_{i}")) is not None)

def is_real_window(h_wnd:int) -> bool:
  has_no_owner = win32gui.GetWindow(h_wnd, win32con.GW_OWNER) == 0
  l_ex_style = win32gui.GetWindowLong(h_wnd, win32con.GWL_EXSTYLE)
  return ((l_ex_style & win32con.WS_EX_TOOLWINDOW) == 0) == has_no_owner and win32gui.GetWindowText(h_wnd) and win32gui.IsWindowVisible(h_wnd) and win32gui.GetParent(h_wnd) == 0

def get_interfaces() -> List[PokerInterface]:
  def callback(h_wnd, wnd_list):
    if is_real_window(h_wnd): wnd_list.append(h_wnd)
  windows = []
  win32gui.EnumWindows(callback, windows)
  return [PokerInterface(h_wnd, conf) for h_wnd in windows if (conf := get_config(h_wnd)) is not None]

def get_config(handle:int) -> dict|None:
  title = win32gui.GetWindowText(handle)
  for c in config.configs:
    if c.site.get_blinds(title) is not None: return c
  return None

def run() -> None:
  just_fix_windows_console()
  global debug
  debug = True
  tables = get_interfaces()
  for table in tables: print(table)

  table_index = 0
  playing = True
  while playing:
    inp = input("\nAction: ").split(' ')
    action, args = inp[0], tuple(inp[1:])
    table = tables[table_index]
    table.debug = True
    img = table.screenshot() if len(args) == 0 else PIL.Image.open(args[0])
    img.save('img_debug/table.png')

    if action == 'update':
      tables = get_interfaces()
      for table in tables: print(table)
    elif action == 'seats':
      for i in range(table.config.n_players): print("Seat", i, "is open." if table.is_seat_open(img, i) else "is taken.")
    elif action == 'usernames':
      for i in range(table.config.n_players): print(f"Seat {i}: {table.username(img, i)}")
    elif action == 'sitout':
      for i in range(5): print(f"Seat {i} is", "sitting out." if table.is_sitting_out(img, i) else "not sitting out.")
    elif action == 'active':
      print(f"Seat {active} is active." if (active := table.active_seat(img)) else "No one is active.")
    elif action == 'cards':
      for i in range(table.config.n_players): print(f"Player {i}", ("has cards." if table.has_cards(img, i) else "folded."))
    elif action == 'invested':
      for i in range(table.config.n_players):
        if table.has_bet(img, i): print(f"Seat {i} bet: {f'${sz:.2f}' if (sz := table.bet_size(img, i) is not None) else None}")
    elif action == 'button':
      print(f"Seat {btn} has the button." if (btn := table.button_pos(img)) else "No one has the button.")
    elif action == 'hand':
      for i in range(table.config.n_players):
        if (hand := table.hand(img, i)) is not None: print(f"Seat {i} hand: {hand}")
    elif action == 'showing':
      for i in range(table.config.n_players): print(f"Seat {i} is {'' if table.is_showing_hand(img, i) else 'not '}showing a hand.")
    elif action == 'board':
      print("Board:", colorize_cards(table.board(img)))
    elif action == 'stacksize':
      for i in range(6):
        if not table.is_seat_open(img, i): print(f"Seat {i} Stack: ${table.stack_size(img, i):.2f}")
    elif action == 'pot':
      print(f"Potsize: ${table.pot_size(img):.2f}")
    elif action == 'street':
      print("Street:", round_to_str(table.round(img)))
    elif action == 'blinds':
      print("Blinds: $%.2f/$%.2f" % table.blinds())
    elif action == 'end':
      playing = False
    else:
      print("Invalid. Try again.")

if __name__ == '__main__':
  run()