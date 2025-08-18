"""
Implements functionality to interact with the poker UI.
"""

import random
import time
import pyautogui
import win32gui
import win32con
import pytesseract
from typing import Tuple, List, Optional
from PIL import ImageOps, ImageEnhance
from PIL.Image import Image
from termcolor import colored

import config

random.seed()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

win_default_offset = 7
win_header_offset = 1
win_header_h = 30

class PokerTable:
  def __init__(self, wnd_handle: int, conf: dict) -> None:
    self.handle = wnd_handle
    self.config = conf

  def rect(self) -> Tuple[int, int, int ,int]:
    r = win32gui.GetWindowRect(self.handle)
    return r[0] + win_default_offset, r[1] + win_header_offset, r[2] - r[0] - 2 * win_default_offset, r[3] - r[1] - win_default_offset - win_header_offset

  def screenshot(self): return pyautogui.screenshot(region=self.rect())
  def title(self) -> str: return win32gui.GetWindowText(self.handle)
  def move(self, x, y, w, h) -> None: win32gui.SetWindowPos(self.handle, win32con.HWND_TOP, x, y, w, h, 0)
  def __str__(self) -> str: return self.__repr__()
  def __repr__(self) -> str: return f"<PokerTable title=\"{self.title()}\", config=\"{self.config["name"]}\", handle={self.handle}>"

def is_real_window(h_wnd) -> bool:
  if not win32gui.IsWindowVisible(h_wnd) or win32gui.GetParent(h_wnd) != 0:
    return False
  has_no_owner = win32gui.GetWindow(h_wnd, win32con.GW_OWNER) == 0
  l_ex_style = win32gui.GetWindowLong(h_wnd, win32con.GWL_EXSTYLE)
  if (((l_ex_style & win32con.WS_EX_TOOLWINDOW) == 0 and has_no_owner) or
      ((l_ex_style & win32con.WS_EX_APPWINDOW != 0) and not has_no_owner)):
    return win32gui.GetWindowText(h_wnd)
  return False

def get_window_list() -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
  """
  Return a list of tuples (handle, title, (x, y, width, height)) for each real window.
  """
  def callback(h_wnd, wnd_list):
    if is_real_window(h_wnd): wnd_list.append(h_wnd)

  windows = []
  win32gui.EnumWindows(callback, windows)
  return windows

def get_table_list(window_list=None) -> List[PokerTable]:
  if window_list is None: window_list = get_window_list()
  return [PokerTable(h_wnd, conf) for h_wnd in window_list if (conf := get_config(h_wnd)) is not None]

def get_config(handle) -> Optional[dict]:
  title = win32gui.GetWindowText(handle)
  for c in config.configs:
    if c['table_title'] in title: return c
  return None

def assert_player_pos(i: int) -> None:
  assert 0 <= i <= 5, f"Invalid Player Position: {player_pos}"

def position_tables(poker_tables: list):
  """
  Position the windows into the layout defined in the current profile.
  """
  assert len(poker_tables) <= 6, "Layout not implemented for > 6 tables."
  _, _, screen_w, screen_h = win32gui.GetWindowRect(win32gui.GetDesktopWindow())
  table_w, table_h = screen_w // 3, screen_h // 2
  positions = [((i // 3) * table_h, (i % 3) * table_w) for i in range(len(poker_tables))]
  for pos, table in zip(positions, poker_tables):
    table.move(pos[0], pos[1], table_w, table_h)

def seat_open(pos: int, img: Image) -> bool:
  assert_player_pos(pos)

  pos -= 1
  if pos == -1: return False

  rgb_val = img.getpixel((cc['seat'][pos][0], cc['seat'][pos][1]))
  if cc['site'] == 'gp':
    return (cc['seat'][pos][2] < rgb_val[0] < cc['seat'][pos][3] and
            cc['seat'][pos][2] < rgb_val[1] < cc['seat'][pos][3] and
            cc['seat'][pos][2] < rgb_val[2] < cc['seat'][pos][3])
  elif cc['site'] == 'bovada':
    return True if rgb_val == cc['seat_pix'][pos] else False

def has_cards(pos, img) -> bool:
  assert_player_pos(pos)
  pos -= 1
  if pos == -1: return hero_has_cards(img)
  return True if img.getpixel((cc['cards'][pos][0], cc['cards'][pos][1])) == cc['cards_pix'] else False

def button_pos(img: Image):
  for i in range(5):
    rgb_val = img.getpixel((cc['button'][i][0], cc['button'][i][1]))
    if rgb_val == cc['button_pix'][i]:
      return i + 1
  return 0

def has_bet(pos: int, img: Image) -> bool:
  assert_player_pos(pos)
  pos -= 1
  if pos == -1: return hero_has_bet(img)

  if cc['site'] == 'gp':
    return img.getpixel(cc['bet_chips'][pos]) == cc['bet_chips_pix']
  elif cc['site'] == 'bovada':
    return img.getpixel(cc['bet_chips'][pos]) != cc['bet_chips_not_pix']

def hero_has_bet(img: Image) -> bool: # TODO: remove
  if cc['site'] == 'bovada':
    return img.getpixel(cc['hero_bet_chips']) != cc['bet_chips_not_pix']
  assert False, "Unknown site"

def read_betsize(pos: int, table_idx: int) -> int:
  assert_player_pos(pos)
  pos -= 1
  img = pyautogui.screenshot(region=(
    cc['bet_amount'][pos][0] + cc['table_window'][table_idx][0],
    cc['bet_amount'][pos][1] + cc['table_window'][table_idx][1],
    cc['bet_amount'][pos][2],
    cc['bet_amount'][pos][3],
  ))
  img = ImageOps.grayscale(img)
  img = ImageOps.invert(img)
  contrast = ImageEnhance.Contrast(img)
  img = contrast.enhance(2)
  img.save('img_debug/betsize' + str(pos + 1) + '.png')

  betsize = pytesseract.image_to_string(img, config='--psm 7')
  if cc['site'] == 'bovada':
    return extract_number(betsize)
  elif cc['site'] == 'gp':
    return betsize
  assert False, "Unknown site"

def read_bet_option(table_idx: int) -> str:
  """
  Returns the bet size currently entered in the bet size selector on the poker UI
  on the i'th table.
  """
  img = pyautogui.screenshot(region=(
    cc['bet_option'][0] + cc['table_window'][table_idx][0],
    cc['bet_option'][1] + cc['table_window'][table_idx][1],
    cc['bet_option'][2],
    cc['bet_option'][3],
  ))
  img = ImageOps.grayscale(img)
  img = ImageOps.invert(img)
  contrast = ImageEnhance.Contrast(img)
  img = contrast.enhance(2)
  img.save('img_debug/bet_option.png')
  bet_option = pytesseract.image_to_string(img, config='--psm 7')
  return extract_number(bet_option)

def read_potsize(table_idx: int, img: Image) -> int:
  read_pos = cc['pot_size_pre'] if current_street(img) == 'preflop' else cc['pot_size_post']

  img = pyautogui.screenshot(region=(
    read_pos[0] + cc['table_window'][table_idx][0],
    read_pos[1] + cc['table_window'][table_idx][1],
    read_pos[2],
    read_pos[3],
  ))
  img = ImageOps.grayscale(img)
  img = ImageOps.invert(img)
  contrast = ImageEnhance.Contrast(img)
  img = contrast.enhance(2)
  img.save('img_debug/potsize.png')
  potsize = pytesseract.image_to_string(img, config='--psm 7')
  return extract_number(potsize)

def extract_number(inp_str: str) -> str:
  i = 0
  while i < len(inp_str):
    if not (inp_str[i].isdigit() or inp_str[i] == '.'):
      inp_str = inp_str[0:i] + inp_str[i + 1: len(inp_str)]
    else:
      i += 1
  return inp_str if util.is_number(inp_str) else ''

def current_street(img: Image) -> str:
  if img.getpixel(cc['card_dealt'][2]) == cc['card_dealt_pix']:
    return 'river'
  elif img.getpixel(cc['card_dealt'][1]) == cc['card_dealt_pix']:
    return 'turn'
  elif img.getpixel(cc['card_dealt'][0]) == cc['card_dealt_pix']:
    return 'flop'
  else:
    return 'preflop'

def is_active(pos: int, img: Image) -> bool:
  assert_player_pos(pos)
  pos -= 1
  return img.getpixel(cc['active'][pos]) != cc['active_not_pix'] if pos != -1 else is_hero_turn(img)

def is_hero_turn(img: Image): # TODO: Remove
  if cc['site'] == 'gp':
    return img.getpixel(cc['hero_active']) == cc['hero_active_pix']
  elif cc['site'] == 'bovada':
    return img.getpixel(cc['hero_active']) != cc['hero_active_not_pix']

def sitting_out(pos: int, img: Image) -> bool:
  if not 0 <= pos <= 5:
    print("ERROR in sitting_out(): Invalid Player Position:", pos)
    return None
  pos -= 1
  if pos == -1: return False
  rgb = img.getpixel((cc['sitting_out'][pos][0], cc['sitting_out'][pos][1]))
  return (rgb[0] == cc['sitting_out'][pos][2] and
          rgb[1] == cc['sitting_out'][pos][2] and
          rgb[2] == cc['sitting_out'][pos][2])

def color_to_suit(rgb: Tuple[int, int, int]) -> str:
  if rgb[0] - 50 > rgb[1] and rgb[0] - 50 > rgb[2]:
    return 'h'
  elif rgb[1] - 50 > rgb[0] and rgb[1] - 50 > rgb[2]:
    return 'c'
  elif rgb[2] - 50 > rgb[1] and rgb[2] - 50 > rgb[0]:
    return 'd'
  elif rgb[0] < 70 and rgb[1] < 70 and rgb[2] < 70:
    return 's'
  else:
    return '-'

def img_to_inv_cont(img: Image) -> Image:
  invert_img = ImageOps.invert(img)
  contrast = ImageEnhance.Contrast(invert_img)
  invert_img = contrast.enhance(2)
  return invert_img

def hero_has_cards(img: Image) -> bool:
  return img.getpixel(cc['hero_cards']) == cc['hero_cards_pix']

def hero_hand(img: Image, table_idx: int) -> str:
  def hero_suits():
    rgb_val_left = img.getpixel(cc['hero_suits'][0])
    rgb_val_right = img.getpixel(cc['hero_suits'][1])
    return color_to_suit(rgb_val_left) + color_to_suit(rgb_val_right)

  def hero_ranks(inp_table_id):
    if cc['site'] == 'gp':
      curr_img = pyautogui.screenshot(region=cc['hero_ranks'])
      curr_img = ImageOps.grayscale(curr_img)
      contrast = ImageEnhance.Contrast(curr_img)
      curr_img = contrast.enhance(2)
      curr_img.save('img_debug/hero_ranks.png')
      text = pytesseract.image_to_string(curr_img, config='--psm 10')
    elif cc['site'] == 'bovada':
      text = ''
      for i in range(2):
        curr_img = pyautogui.screenshot(region=(
          cc['hero_ranks'][i][0] + cc['table_window'][inp_table_id][0],
          cc['hero_ranks'][i][1] + cc['table_window'][inp_table_id][1],
          cc['hero_ranks'][i][2],
          cc['hero_ranks'][i][3],
        ))
        curr_img = ImageOps.grayscale(curr_img)
        contrast = ImageEnhance.Contrast(curr_img)
        curr_img = contrast.enhance(2)
        curr_img.save('img_debug/hero_ranks' + str(i) + '.png')
        text += pytesseract.image_to_string(curr_img, config='--psm 10')
    else:
      print("ERROR in hero_ranks(): Invalid Site Name:", cc['site'])
      text = ""

    text = text.replace(' ', '')
    text = text.replace('10', 'T')
    text = clean_card_str(text)
    return text if len(text) == 2 else None

  suits = hero_suits()
  ranks = hero_ranks(table_idx)

  # Invalid readings:
  if not (suits and ranks):
    return None

  # Canonicalize
  if card_to_index(ranks[0]) < card_to_index(ranks[1]):
    ranks = (ranks[1], ranks[0])
    suits = (suits[1], suits[0])
  return ranks[0] + suits[0] + ranks[1] + suits[1]

def clean_card_str(card_str: str) -> str:
  return [c for c in card_str if c in ['AKQJT98765432']]

def player_stacksize(pos: int, table_idx: int) -> str:
  """
  Takes an integer between 0 and 5, defining the seat (clockwise positions starting at the bottom).
  Takes an integer specifying the table to read from.
  Returns that seat's stacksize.
  """
  assert_player_pos(pos)
  pos -= 1
  if pos == -1: return hero_stacksize(table_idx) # TODO: remove
  img = pyautogui.screenshot(region=(
    cc['stack_size'][pos][0] + cc['table_window'][table_idx][0],
    cc['stack_size'][pos][1] + cc['table_window'][table_idx][1],
    cc['stack_size'][pos][2],
    cc['stack_size'][pos][3],
  ))
  img.save('img_debug/stacksize' + str(pos + 1) + '.png')
  if cc['site'] == 'gp':
    img = img_to_inv_cont(img)
  elif cc['site'] == 'bovada':
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(2)
  else:
    print("ERROR in player_stacksize(): Invalid Site Name:", cc['site'])
  text = pytesseract.image_to_string(img, config='--psm 7')
  text = extract_number(text)
  return text if text else '-1'

def hero_stacksize(i: int) -> int:
  """
  Returns hero's stacksize on the i'th table.
  """
  img = pyautogui.screenshot(region=(
    cc['hero_stack_size'][0] + cc['table_window'][i][0],
    cc['hero_stack_size'][1] + cc['table_window'][i][1],
    cc['hero_stack_size'][2],
    cc['hero_stack_size'][3],
  ))
  img.save('img_debug/stacksize0.png')
  if cc['site'] == 'gp':
    img = img_to_inv_cont(img)
  elif cc['site'] == 'bovada':
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(2)
  else:
    print("ERROR in hero_stacksize(): Invalid Site Name:", cc['site'])
  text = pytesseract.image_to_string(img, config='--psm 7')
  text = extract_number(text)
  return text if text else '-1'

def read_board(img: Image, table_idx: int, start_street: Optional[str] = None) -> str:
  """
  Returns the entire board of the i'th poker table if street = None.
  If a street is given, starts returning cards from that street from table i.
  If start_street = 'recent', only reads the most recent street.
  """
  table_street = current_street(img)
  if start_street == 'recent': start_street = table_street
  cards = {'flop': 3, 'turn': 4, 'river': 5}.get(table_street, 0)
  first_card = {'turn': 3, 'river': 4}.get(start_street, 0)
  return ''.join(board_rank(i, table_idx) + board_suits(i, img) for i in range(first_card, cards))

def board_rank(card_idx: int, table_idx: int) -> str:
  """
  Returns the rank of the i'th card on the board, where i = 0 is the first card.
  """
  img = pyautogui.screenshot(region=(
    cc['board_ranks'][card_idx][0] + cc['table_window'][table_idx][0],
    cc['board_ranks'][card_idx][1] + cc['table_window'][table_idx][1],
    cc['board_ranks'][card_idx][2],
    cc['board_ranks'][card_idx][3],
  ))
  img = ImageOps.grayscale(img)
  contrast = ImageEnhance.Contrast(img)
  img = contrast.enhance(2)
  img.save('img_debug/board_rank' + str(card_idx) + '.png')
  text = pytesseract.image_to_string(img, config='--psm 10')
  text = text.replace(' ', '')
  text = text.replace('10', 'T')
  text = clean_card_str(text)
  return text if (text.isalpha() or text.isdigit()) and len(text) <= 2 else ''

def board_suits(card_idx: int, img: Image) -> str:
  """
  Returns the suit of the i'th board card as 'c', 'h', 'd', 's'
  Returns '' if the i'th board card is not dealt.
  """
  img.save(r'img_debug\board_suits.png')
  rgb_val = img.getpixel(cc['board_suits'][card_idx])

  if rgb_val == cc['board_suits_pix'][0]:
    return 's'
  elif rgb_val == cc['board_suits_pix'][1]:
    return 'h'
  elif rgb_val == cc['board_suits_pix'][2]:
    return 'd'
  elif rgb_val == cc['board_suits_pix'][3]:
    return 'c'
  else:
    print(colored("ERROR in board_suits(). Invalid pixel RGB value:", rgb_val, 'red'))
    return ''

def get_stakes(table_title: str) -> Tuple[int, int]:
  bb_value = table_title[table_title.index('/') + 2:]
  bb_value = bb_value[:bb_value.index(' ')]
  sb_value = table_title[table_title.index('$') + 1:]
  sb_value = sb_value[:sb_value.index('/')]
  return float(sb_value), float(bb_value)

def card_to_index(card: str) -> int:
  return {'A': 12, 'K': 11, 'Q': 10, 'J': 9, 'T': 8}.get(card, int(card) - 2)

def is_number(inp_str: str) -> bool:
  pre_digit = 0
  dot = 0
  post_digit = 0
  for char in inp_str:
    if char == '.':
      dot += 1
    elif char.isdigit():
      if dot == 0:
        pre_digit += 1
      else:
        post_digit += 1
    else:
      return False
  return pre_digit >= 1 >= dot and (dot == 0 or post_digit > 0)

def debug() -> None:
  poker_tables = get_table_list()
  for table in poker_tables: print(table)
  # position_tables(poker_tables)

  table_index = 0
  playing = True
  while playing:
    action = input("Action: ")
    table_img = poker_tables[table_index].screenshot()
    table_img.save('img_debug/table.png')

    if action == 'update':
      poker_tables = get_table_list()
      for table in poker_tables: print(table)
    elif action == 'screenshot':
      table = poker_tables[table_index]
      table.screenshot().save(f"img_debug/{table.config['name']}_{table.handle}.png")
    elif action == 'seat':
      for i in range(6):
        if seat_open(i, table_img):
          print("Seat", i, "is open.")
        else:
          print("Seat", i, "is taken.")
    elif action == 'sitout':
      for i in range(5):
        print("Seat", i, "is", ("sitting out." if sitting_out(i, table_img) else "not sitting out."))
    elif action == 'waiting':
      pass
      # print(table_img.getpixel((cc['waiting'][testing][0], cc['waiting'][testing][1])))
    elif action == 'active':
      for i in range(6):
        print("Seat {num}: {active}".format(num=i, active="Active" if is_active(i, table_img) else "Inactive"))
    elif action == 'card':
      for i in range(6):
        print("Player", i, ("has cards." if has_cards(i, table_img) else "folded."))
    elif action == 'invested':
      for i in range(1, 6):
        if has_bet(i, table_img):
          amount = read_betsize(i, table_index)
          print("Seat", i, "Bet:", amount)
    elif action == 'button':
      print("Seat", button_pos(table_img), "has the button.")
    elif action == 'hero':
      print("Hero", ("has cards." if hero_has_cards(table_img) else "does not have cards."))
      print("Hero has", ("bet." if hero_has_bet(table_img) else "not bet."))
      print("Hero is", ("active." if is_hero_turn(table_img) else "inactive."))
    elif action == 'hand':
      if hero_has_cards(table_img):
        print('Hero Hand:', hero_hand(table_img, table_index))
      else:
        print("Hero does not have cards.")
    elif action == 'board':
      print("Board:", read_board(table_img, table_index))
    elif action == 'stacksize':
      for i in range(6):
        print("Seat", i, "Stack:", player_stacksize(i, table_index))
    elif action == 'pot':
      print("Potsize:", read_potsize(table_index, table_img))
    elif action == 'option':
      print("Bet Option:", read_bet_option(table_index))
    elif action == 'street':
      print("Street:", current_street(table_img))
    elif action == 'stakes':
      stakes = get_stakes(poker_tables[0][1])
      print("Stakes:", str(stakes[0]) + '/' + str(stakes[1]))
    elif action == 'end':
      playing = False
    else:
      print("Invalid. Try again.")

if __name__ == '__main__':
  debug()
