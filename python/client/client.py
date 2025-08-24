import pyautogui
import time
from typing import Tuple, List
from dataclasses import dataclass
from termcolor import colored
from PIL.Image import Image
from interface import PokerInterface, get_interfaces
from state import PokerState, round_to_str
from colorama import just_fix_windows_console

@dataclass
class PokerTable:
  state: PokerState
  interface: PokerInterface
  round: int = 0
  def __repr__(self) -> str: return f"{self.interface}\n{self.state}"
  def __str__(self) -> str: return self.__repr__()

def to_pos(seat:int, btn:int, n_players:int) -> int:
  pos = n_players + seat - btn - 1
  return pos - n_players if pos >= n_players else pos

def to_seat(pos:int, btn:int, n_players:int) -> int:
  seat = -n_players + pos + btn + 1
  return seat + n_players if seat < 0 else seat

def to_state_stacks(stacks:List[float], btn:int):
  state_stacks = [None] * len(stacks)
  for i,chips in enumerate(stacks): state_stacks[to_pos(i, btn, len(stacks))] = chips
  return state_stacks

def init_new_hand(img:Image, inter:PokerInterface, verbose:bool=False) -> PokerState|None:
  if inter.round(img) == 0 and inter.active_seat(img) is not None and (btn := inter.button_pos(img)) is not None:
    stacks = [inter.stack_size(img, pos, blinds=True) for pos in range(inter.config.n_players) if not inter.is_seat_open(img, pos) and not inter.is_sitting_out(img, pos)]
    if sum(s is None for s in stacks) == 0 and inter.pot_size(img) == sum(inter.blinds()) + inter.ante() * len(stacks):
      return PokerState(to_state_stacks(stacks, btn), inter.ante(blinds=True), len(inter.blinds()) == 3, verbose=verbose)
    else:
      return None
  else:
    return None

def update_tables(tables:List[PokerTable]) -> None:
  curr_inter = get_interfaces()
  curr_handles = set(t.handle for t in curr_inter)
  for i in range(len(tables)-1,-1,-1):
    if not tables[i].interface.handle in curr_handles:
      curr_inter.pop(i)
  prev_handles = set(t.interface.handle for t in tables)
  for inter in curr_inter:
    if not inter.handle in prev_handles:
      tables.append(PokerTable(None, inter))
      print(colored(f"New Table: {inter}", "yellow"))

def update_state(img:Image, table:PokerTable, log_level:int=0):
  tol = 1e-3
  while True:
    if table.state is None:
      table.state = init_new_hand(img, table.interface, verbose=log_level >= 1)
      if not table.state is None: print(colored(f"New hand:\n{table.state}", "yellow"))
    state, inter = table.state, table.interface
    if state is None:
      if log_level >= 2: print("Waiting for new hand...")
      return
    if (curr_round := inter.round(img)) < state.round <= 3:
      if log_level >= 2: print("Waiting for next card...")
      return
    if (button_pos := inter.button_pos(img)) is None:
      if log_level >= 0: print(colored("Failed to find button.", "red"))
      return
    if (active_seat := inter.active_seat(img)) is None:
      return
    if (query_seat := to_seat(state.active, button_pos, len(state.players))) == active_seat and curr_round == state.round:
      return
    if log_level >= 0 and curr_round > table.round:
      if (board := inter.board(img)) is not None:
        print(colored(f"{str_to_round} (Pot: {inter.pot_size(img, blinds=True)}):", "yellow"), colorize_board(board))
      else:
        print(colored("Failed to read board.", "red"))
    img.save(r"img_debug\table.png")
    if not inter.has_cards(img, query_seat):
      state.fold()
    elif not inter.has_bet(img, query_seat):
      if log_level >= 2: print(f"\tSeat {query_seat} has not bet.")
      if curr_round == state.round:
        if log_level >= 2: print(f"\tRound has not changed.")
        state.check()
      else:
        if log_level >= 2: print(f"\tRound has changed: {state.round} -> {curr_round}")
        if (chips := inter.stack_size(img, query_seat, blinds=True)) is None:
          if log_level >= 0: print(colored(f"Failed to read stack size {query_seat}.", "red"))
          return
        diff = state.players[state.active].chips - chips
        if log_level >= 2:
          print(f"\tSeat {query_seat} chips: {chips:.2f} bb\n\tSeat {query_seat} chip difference: {diff:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
        if diff > tol:
          if diff > state.max_bet + tol:
            state.bet(diff)
          else:
            state.call()
        else:
          state.check()
    else:
      if (bet_size := inter.bet_size(img, query_seat, blinds=True)) is None:
        if log_level >= 2: print("Failed to read bet size {query_seat}.")
        return
      if log_level >= 2:
        print(f"\tSeat {query_seat} has bet {bet_size:.2f} bb\n\tPreviously invested: {state.players[state.active].betsize:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
      if curr_round == state.round:
        if log_level >= 2: print(f"\tRound has not changed.")
        if bet_size > state.max_bet + tol:
          state.bet(bet_size - state.players[state.active].betsize)
        else:
          state.call()
      else:
        if log_level >= 2: print(f"\tRound has changed: {state.round} -> {curr_round}")
        if (chips := inter.stack_size(img, query_seat, blinds=True)) is None:
          if log_level >= 2: print(f"Failed to read stack size {query_seat}.")
          return
        diff = state.players[state.active].chips - chips - bet_size
        if log_level >= 2:
          print(f"\tSeat {query_seat} chips: {chips:.2f} bb\n\tSeat {query_seat} chip difference: {diff:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
        if diff > tol:
          if diff > state.max_bet + tol:
            state.bet(diff)
          else:
            state.call()
        else:
          state.check()
    if state.winner is not None or state.round > 3:
      table.state = None

def run(log_level:int=0) -> None:
  just_fix_windows_console()
  tables: List[PokerTable] = []
  while True:
    t_0 = time.time()
    screen = pyautogui.screenshot()
    update_tables(tables)
    for table in tables:
      update_state(screen.crop(table.interface.rect()), table, log_level=log_level)

    # print(f"Cycle time: {time.time() - t_0:.2f} s")

if __name__ == "__main__":
  run(log_level=2)