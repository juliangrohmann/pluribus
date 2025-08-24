import time
from typing import Tuple, List
from dataclasses import dataclass
from termcolor import colored
from colorama import just_fix_windows_console
from PIL.Image import Image
from interface import PokerInterface, get_interfaces
from state import PokerState
from screenshot import screenshot_all
from util import round_to_str, colorize_cards

@dataclass
class PokerTable:
  state: PokerState
  interface: PokerInterface
  seat_map: List[int]
  round: int = 0
  def __repr__(self) -> str: return f"{self.interface}\n{self.state}"
  def __str__(self) -> str: return self.__repr__()

def build_seat_map(valid:List[bool], btn:int):
  assert valid[btn], "Button is not in a valid seat."
  return [s for i in range(btn - len(valid) + 1, btn + 1) if valid[s := i if i >= 0 else i + len(valid)]]

def init_new_hand(img:Image, inter:PokerInterface, verbose:bool=False) -> Tuple[PokerState,List[int]]|None:
  if inter.round(img) == 0 and inter.active_seat(img) is not None and (btn := inter.button_pos(img)) is not None:
    valid = [not inter.is_seat_open(img, pos) and not inter.is_sitting_out(img, pos) for pos in range(inter.config.n_players)]
    stacks = [inter.stack_size(img, i, blinds=True) if v else None for i,v in enumerate(valid)]
    if not any(s is None for s,v in zip(stacks, valid) if v) and inter.pot_size(img) == sum(inter.blinds()) + inter.ante() * len(stacks):
      seat_map = build_seat_map(valid, btn)
      state_stacks = [stacks[seat_map[pos]] for pos in range(len(seat_map)) if valid[seat_map[pos]]]
      return PokerState(state_stacks, inter.ante(blinds=True), len(inter.blinds()) == 3, verbose=verbose), seat_map
  return None, None

def update_tables(tables:List[PokerTable]) -> None:
  curr_inter = get_interfaces()
  curr_handles = set(t.handle for t in curr_inter)
  for i in range(len(tables)-1,-1,-1):
    if not tables[i].interface.handle in curr_handles:
      curr_inter.pop(i)
  prev_handles = set(t.interface.handle for t in tables)
  for inter in curr_inter:
    if not inter.handle in prev_handles:
      tables.append(PokerTable(None, inter, None))
      print(colored(f"New Table: {inter}", "yellow"))

def print_showdown(showing:List[bool], hands:List[str|None], pot:float, board:str):
  print(colored(f"Showdown (Pot: {pot:.2f} bb, Board: {board})", "yellow"))
  for i in range(len(hands)):
    if showing[i]:
      print(colored(f"Player {i} shows", "yellow"), colorize_cards(hands[i]) if hands[i] is not None else "")

def update_state(img:Image, table:PokerTable, debug:int=0):
  tol = 1e-3
  while True:
    if table.state is None or table.state.is_terminal():
      next_state, next_seat_map = init_new_hand(img, table.interface, verbose=debug >= 1)
      if next_state is not None:
        print(colored(f"New hand:\n", "yellow") + str(next_state))
        table.state, table.seat_map = next_state, next_seat_map
        table.round = 0
    state, inter = table.state, table.interface
    if state is None:
      return
    if (curr_round := inter.round(img)) < state.round <= 3:
      if debug >= 2: print("Waiting for next card...")
      return
    if debug >= 0 and table.round < curr_round and table.round < state.round:
      if (board := inter.board(img)) is not None:
        print(colored(f"{round_to_str(curr_round)} (Pot: {state.pot:.2f}):", "yellow"), colorize_cards(board))
        table.round = curr_round
      else:
        print(colored("Failed to read board.", "red"))
        img.save(r"img_debug\fail_board.png")
    showing = [not p.folded and inter.is_showing_hand(img, table.seat_map[i]) for i,p in enumerate(state.players)]
    hands = [inter.hand(img, table.seat_map[i]) if s else None for i,s in enumerate(showing)]
    if any(showing) and any(h is None for s,h in zip(showing, hands) if s):
      if debug >= 2: print("\tFailed to read all showing hands.")
      return
    if state.is_terminal():
      if any(showing) and (pot := inter.pot_size(img, blinds=True)) is not None and (board := inter.board(img)) is not None:
        print_showdown(showing, hands, pot, board)
        if debug >= 2: print("Waiting for new hand...")
        table.state = None
      elif debug >= 2: print("Waiting for showdown...")
      return
    query_seat = table.seat_map[state.active]
    if any(showing):
      diff = state.players[state.active].chips - (chips := inter.stack_size(img, query_seat, blinds=True))
      if debug >= 2:
        print(f"\tSeat {query_seat} is showing a hand.")
        print(f"\tSeat {query_seat} chips: {chips:.2f} bb\n\tSeat {query_seat} chip difference: {diff:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
      if state.max_bet == 0.0:
        state.check()
      elif diff > tol:
        state.call()
      else:
        state.fold()
    elif not inter.has_cards(img, query_seat):
      if debug >= 2: print(f"\tSeat {query_seat} does not have cards.")
      state.fold()
    elif (active_seat := inter.active_seat(img)) is None:
      return
    elif query_seat == active_seat and curr_round == state.round:
      return
    else:
      if not inter.has_bet(img, query_seat):
        if debug >= 2: print(f"\tSeat {query_seat} has not bet.")
        if curr_round == state.round and curr_round < 3:
          if debug >= 2: print(f"\tRound has not changed.")
          state.check()
        else:
          if debug >= 2: print(f"\tRound has changed: {state.round} -> {curr_round}")
          if (chips := inter.stack_size(img, query_seat, blinds=True)) is None:
            if debug >= 0:
              print(colored(f"Failed to read stack size {query_seat}.", "red"))
              img.save(f"img_debug\\fail_stack_size_{query_seat}.png")
            return
          diff = state.players[state.active].chips - chips
          if debug >= 2:
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
          if debug >= 0:
            print(colored(f"Failed to read bet size {query_seat}.", "red"))
            img.save(f"img_debug\\fail_bet_size_{query_seat}.png")
          return
        if debug >= 2:
          print(f"\tSeat {query_seat} has bet {bet_size:.2f} bb\n\tPreviously invested: {state.players[state.active].betsize:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
        if curr_round == state.round:
          if debug >= 2: print(f"\tRound has not changed.")
          if bet_size > state.max_bet + tol:
            state.bet(bet_size - state.players[state.active].betsize)
          else:
            state.call()
        else:
          if debug >= 2: print(f"\tRound has changed: {state.round} -> {curr_round}")
          if (chips := inter.stack_size(img, query_seat, blinds=True)) is None:
            if debug >= 0:
              print(colored(f"Failed to read stack size {query_seat}.", "red"))
              img.save(f"img_debug\\fail_stack_size_{query_seat}.png")
            return
          diff = state.players[state.active].chips - chips - bet_size
          if debug >= 2:
            print(f"\tSeat {query_seat} chips: {chips:.2f} bb\n\tSeat {query_seat} chip difference: {diff:.2f} bb\n\tMax bet: {state.max_bet:.2f} bb")
          if diff > tol:
            if diff > state.max_bet + tol:
              state.bet(diff)
            else:
              state.call()
          else:
            state.check()

def run(log_level:int=0) -> None:
  just_fix_windows_console()
  cycles = []
  tables: List[PokerTable] = []
  while True:
    if log_level >= 1: t_0 = time.time()
    screen = screenshot_all()
    update_tables(tables)
    for table in tables:
      update_state(screen.crop(table.interface.rect()), table, debug=log_level)
    if log_level >= 1:
      cycles.append(time.time() - t_0)
      if len(cycles) == 1000:
        print(colored(f"Avg cycle time: {sum(cycles) / len(cycles):.3f} s", "red"))
        print(colored(f"Max cycle time: {max(cycles):.3f} s", "red"))
        print(colored(f"Min cycle time: {min(cycles):.3f} s", "red"))
        cycles.clear()

if __name__ == "__main__":
  print("Running...")
  run(log_level=2)