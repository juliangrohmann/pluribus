"""
Implements functionality to interact with the poker UI.
"""

import random
import time
import pyautogui
import win32gui
import win32con
import PIL
from typing import Tuple, List, Callable
from PIL import ImageOps, ImageEnhance, Image
from PIL.Image import Image
from termcolor import colored
from collections import Counter
import numpy as np
import pytesseract
import config
import ocr

random.seed()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

win_default_offset = 8
win_header_offset = 1
win_header_h = 30

def build_card_mask() -> np.ndarray:
  with open(ocr.charset_path, "r", encoding="utf-8") as f:
    charset = [line.rstrip("\n") for line in f]
  return np.array([i == 0 or c in "1234567890JKQA" for i,c in enumerate(charset)])

card_mask = build_card_mask()

def assert_player_pos(i:int) -> None:
  assert 0 <= i <= 5, f"Invalid Player Position: {player_pos}"

def frac_to_pix(x:float, y:float, img_w:int, img_h:int) -> Tuple[int,int]:
  return int(round(x * img_w)), int(round(y * (img_h - win_header_h) + win_header_h))

def crop(img:Image, box:Tuple[int,int,int,int]) -> Image:
  return img.crop((*frac_to_pix(box[0], box[1], img.width, img.height), *frac_to_pix(box[2], box[3], img.width, img.height)))

class PokerTable:
  def __init__(self, wnd_handle:int, conf:dict) -> None:
    self.handle = wnd_handle
    self.config = conf

  def hand(self, pos:int, img:Image) -> str|None:
    assert_player_pos(pos)
    return hand_str if (hand_str := self._parse_cards(img, self.config.hole_cards_ranks[pos], self.config.hole_cards_suits[pos], debug_label=f"hole_cards_{pos}_rank")) else None

  def username(self, img:Image, pos:int) -> str|None:
    assert_player_pos(pos)
    name = self._parse_ocr(img, self.config.usernames[pos], tuple(), debug_label=f"username_{pos}")
    return name if name not in ["Bet", "Raise", "Check", "Call", "Fold"] else None

  def street(self, img:Image) -> str|None: return {0:"Preflop",3:"Flop",4:"Turn",5:"River"}.get(sum(bool(self._parse_suit(img,self.config.board_suits[i])) for i in range(5)), None)
  def board(self, img:Image) -> str|None: return self._parse_cards(img, self.config.board_ranks, self.config.board_suits, debug_label=f"board_rank")
  def stack_size(self, img:Image, pos:int, blinds:bool=False) -> float|None: return self._cash_amount_by_pos(pos, img, self.config.stacks, blinds=blinds, debug_label=f"stack_size_{pos}")
  def bet_size(self, img:Image, pos:int, blinds:bool=False) -> float|None: return self._cash_amount_by_pos(pos, img, self.config.bet_size, blinds=blinds, debug_label=f"bet_size_{pos}")
  def pot_size(self, img:Image, blinds:bool=False) -> float|None: return self._cash_amount(img, self.config.pot, ("Pot", " ", ":"), blinds, debug_label="pot_size")
  def active_pos(self, img:Image) -> int|None: return self._find_pos(img, self.config.active, self.config.is_active)
  def button_pos(self, img:Image) -> int|None: return self._find_pos(img, self.config.button, self.config.has_button)
  def is_seat_open(self, img:Image, pos:int) -> bool: return self._apply_color_filter(pos, img, self.config.seats, self.config.is_seat_open)
  def is_sitting_out(self, img:Image, pos:int) -> bool: return self._parse_ocr(img, self.config.stacks[pos], debug_label=f"sitout_{pos}") == "Sitting Out"
  def has_cards(self, img:Image, pos:int) -> bool: return self._apply_color_filter(pos, img, self.config.cards, self.config.has_cards)
  def has_bet(self, img:Image, pos:int) -> bool: return self._apply_color_filter(pos, img, self.config.bet_chips, self.config.has_bet)
  def screenshot(self) -> Image: return pyautogui.screenshot(region=self.rect())
  def title(self) -> str: return win32gui.GetWindowText(self.handle)
  def stakes(self) -> Tuple[float,float]: return self.config.get_stakes(self.title())
  def move(self, x:int, y:int, w:int, h:int) -> None: win32gui.SetWindowPos(self.handle, win32con.HWND_TOP, x, y, w, h, 0)
  def __repr__(self) -> str: return f"<PokerTable title=\"{self.title()}\", config=\"{self.config.name}\", handle={self.handle}>"
  def __str__(self) -> str: return self.__repr__()

  def rect(self) -> Tuple[int,int,int,int]:
    r = win32gui.GetWindowRect(self.handle)
    return r[0] + win_default_offset, r[1] + win_header_offset, r[2] - r[0] - 2 * win_default_offset, r[3] - r[1] - win_default_offset - win_header_offset

  def _apply_color_filter(self, pos:int, img:Image, coords:Tuple[Tuple[float,float], ...], fun:Callable[[Tuple[int,int,int]], bool]) -> bool:
    assert_player_pos(pos)
    if img is None: img = self.screenshot()
    return fun(img.getpixel(frac_to_pix(*coords[pos], img.width, img.height)))

  def _find_pos(self, img:Image, coords:Tuple[Tuple[float,float], ...], fun:Callable[[Tuple[int,int,int]], bool]) -> int|None:
    if img is None: img = self.screenshot()
    for pos in range(self.config.n_players):
      if self._apply_color_filter(pos, img, coords, fun): return pos
    return None

  def _parse_ocr(self, img:Image, coords:Tuple[int,int,int,int], repl:Tuple[str, ...]=tuple(), expand: int=4, mask:np.ndarray|None=None, debug_label:str=None) -> str:
    if img is None: img = self.screenshot()
    raw = ocr.pipeline.run(cropped := crop(img, coords), expand=expand, mask=mask)[0]
    if debug_label:
      cropped.save(f"img_debug\\{debug_label}.png")
    for s in repl: raw = raw.replace(s, '')
    return raw

  def _parse_rank(self, img:Image, coords:Tuple[int,int,int,int], n:int=8, debug_label:str=None) -> str:
    if img is None: img = self.screenshot()
    cropped = crop(img, coords)
    repeated = PIL.Image.new(cropped.mode, (cropped.width * n, cropped.height))
    for i in range(n): repeated.paste(cropped, (i * cropped.width, 0))
    counter = Counter(ocr.pipeline.run(repeated, expand=10, mask=card_mask)[0])
    if debug_label:
      repeated.save(f"img_debug\\{debug_label}.png")
    return 'T' if set(e[0] for e in counter.most_common(2)) == set('10') else counter.most_common(1)[0][0]

  def _cash_amount(self, img:Image, coords:Tuple[int,int,int,int], repl:Tuple[str, ...]=tuple(), blinds:bool=False, debug_label:str=None) -> float|None:
    raw = self._parse_ocr(img, coords, repl + ("$", ","), debug_label=debug_label)
    if raw.lower() == "all in": return 0.0
    try:
      amount = float(raw)
      return amount / self.get_stakes()[1] if blinds else amount
    except ValueError:
      return None

  def _cash_amount_by_pos(self, pos:int, img:Image, coords_by_pos:Tuple[Tuple[int,int,int,int], ...], blinds:bool=False, debug_label:str=None) -> float|None:
    assert_player_pos(pos)
    return self._cash_amount(img, coords_by_pos[pos], blinds=blinds, debug_label=debug_label)

  def _rgb_to_suit(self, rgb:Tuple[int,int,int]) -> str|None:
    if self.config.is_heart(rgb): return 'h'
    if self.config.is_club(rgb): return 'c'
    if self.config.is_diamond(rgb): return 'd'
    if self.config.is_spade(rgb): return 's'
    return None

  def _parse_suit(self, img:Image, coords:Tuple[int,int]) -> str|None:
    return self._rgb_to_suit(img.getpixel(frac_to_pix(*coords, *img.size)))

  def _parse_cards(self, img:Image, rank_coords:Tuple[Tuple[int, int, int, int], ...], suit_coords:Tuple[Tuple[int,int], ...], debug_label:str=None) -> str|None:
    return ''.join(self._parse_rank(img, rank_coords[i], debug_label=f"{debug_label}_{i}") + suit for i in range(len(rank_coords))
                   if (suit := self._parse_suit(img, suit_coords[i])) is not None)

def is_real_window(h_wnd:int) -> bool:
  has_no_owner = win32gui.GetWindow(h_wnd, win32con.GW_OWNER) == 0
  l_ex_style = win32gui.GetWindowLong(h_wnd, win32con.GWL_EXSTYLE)
  return ((l_ex_style & win32con.WS_EX_TOOLWINDOW) == 0) == has_no_owner and win32gui.GetWindowText(h_wnd) and win32gui.IsWindowVisible(h_wnd) and win32gui.GetParent(h_wnd) == 0

def get_tables() -> List[PokerTable]:
  def callback(h_wnd, wnd_list):
    if is_real_window(h_wnd): wnd_list.append(h_wnd)
  windows = []
  win32gui.EnumWindows(callback, windows)
  return [PokerTable(h_wnd, conf) for h_wnd in windows if (conf := get_config(h_wnd)) is not None]

def get_config(handle:int) -> dict|None:
  title = win32gui.GetWindowText(handle)
  for c in config.configs:
    if c.get_stakes(title) is not None: return c
  return None

def debug() -> None:
  tables = get_tables()
  for table in tables: print(table)

  table_index = 0
  playing = True
  while playing:
    inp = input("\nAction: ").split(' ')
    action, args = inp[0], tuple(inp[1:])
    table = tables[table_index]
    table_img = table.screenshot() if len(args) == 0 else PIL.Image.open(args[0])
    table_img.save('img_debug/table.png')

    if action == 'update':
      tables = get_tables()
      for table in tables: print(table)
    elif action == 'seats':
      for i in range(table.config.n_players): print("Seat", i, "is open." if table.is_seat_open(table_img, i) else "is taken.")
    elif action == 'usernames':
      for i in range(table.config.n_players): print(f"Seat {i}: {table.username(table_img, i)}")
    elif action == 'sitout':
      for i in range(5): print(f"Seat {i} is", "sitting out." if table.is_sitting_out(table_img, i) else "not sitting out.")
    elif action == 'active':
      print(f"Seat {active} is active." if (active := table.active_pos(table_img)) else "No one is active.")
    elif action == 'cards':
      for i in range(table.config.n_players): print(f"Player {i}", ("has cards." if table.has_cards(table_img, i) else "folded."))
    elif action == 'invested':
      for i in range(table.config.n_players):
        if table.has_bet(table_img, i): print(f"Seat {i} bet: {table.bet_size(table_img, i)}")
    elif action == 'button':
      print(f"Seat {btn} has the button." if (btn := table.button_pos(table_img)) else "No one has the button.")
    elif action == 'hand':
      for i in range(table.config.n_players):
        if (hand := table.hand(i, table_img)) is not None: print(f"Seat {i} hand: {hand}")
    elif action == 'board':
      print("Board:", table.board(table_img))
    elif action == 'stacksize':
      for i in range(6):
        if not table.is_seat_open(table_img, i): print("Seat", i, "Stack:", table.stack_size(table_img, i))
    elif action == 'pot':
      print("Potsize:", table.pot_size(table_img))
    elif action == 'street':
      print("Street:", table.street(table_img))
    elif action == 'stakes':
      print("Stakes: $%.2f/$%.2f" % table.stakes())
    elif action == 'end':
      playing = False
    else:
      print("Invalid. Try again.")

if __name__ == '__main__':
  debug()
