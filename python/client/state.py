from typing import Tuple
from termcolor import colored
from util import round_to_str

def increment(i:int, max_i:int) -> int: return i + 1 if i + 1 <= max_i else 0

class Player:
  def __init__(self, chips:int) -> None:
    self.chips: List[float] = chips
    self.betsize: float = 0
    self.folded: bool = False

  def invest(self, amount:int) -> None:
    assert not self.folded, "Attempted to invest but player already folded."
    assert self.chips >= amount, "Attempted to invest more chips than available."
    self.chips -= amount
    self.betsize += amount

  def post_ante(self, amount:int) -> None:
    self.chips -= amount

  def next_round(self) -> None:
    self.betsize = 0

  def fold(self) -> None:
    self.folded = True

class PokerState:
  def __init__(self, chips:Tuple[int, ...], ante:int, straddle:bool, verbose:bool=False) -> None:
    assert len(chips) >= 2, f"At least 2 players required."
    self.players: List[Player] = [Player(c) for c in chips]
    self.ante: float = ante
    self.straddle: bool = straddle
    self.verbose: bool = verbose
    self.pot: float = 0.0
    self.max_bet: float = 1.0
    self.bet_level: int = 1
    self.round: int = 0
    self.winner: int|None = None
    for pos in range(len(chips)):
      self.players[pos].betsize = (blind_sz := self._blind_size(pos))
      self.pot += blind_sz + self.ante
      self.max_bet = max(blind_sz, self.max_bet)
    self.active = 1 if len(self.players) == 2 else (0, 3)[len(self.players) > 3] if self.straddle else 2

  def bet(self, amount:float) -> None:
    if self.verbose: print(colored(f"{self._prefix_str(self.active, amount)} {'Bet' if self.bet_level == 0 else 'Raise to'} {amount + self.players[self.active].betsize:.2f} bb", "green"))
    self._assert_act()
    self._assert_invest(amount)
    assert amount + self.players[self.active].betsize > self.max_bet, "Attempted to but the betsize does not exceed the existing maximum bet."
    self.players[self.active].invest(amount)
    self.pot += amount
    self.max_bet = self.players[self.active].betsize
    self.bet_level += 1
    self._next_player()

  def call(self) -> None:
    amount = min(self.max_bet - self.players[self.active].betsize, self.players[self.active].chips)
    if self.verbose: print(colored(f"{self._prefix_str(self.active, amount)} Call {amount:.2f} bb", "green"))
    self._assert_act()
    self._assert_facing_bet()
    self._assert_invest(amount)
    assert self.max_bet > self.players[self.active].betsize, "Attempted call but player has already placed the maximum bet."
    self.players[self.active].invest(amount)
    self.pot += amount
    self._next_player()

  def check(self) -> None:
    if self.verbose: print(colored(f"{self._prefix_str(self.active, 0.0)} Check", "green"))
    self._assert_act()
    assert self.players[self.active].betsize == self.max_bet, "Attempted check but a unmatched bet exists."
    self._next_player()

  def fold(self) -> None:
    if self.verbose: print(colored(f"{self._prefix_str(self.active, 0.0)} Fold", "green"))
    self._assert_act()
    self._assert_facing_bet()
    assert self.players[self.active].betsize < self.max_bet, "Attempted to fold but player can check"
    self.players[self.active].fold()
    self.winner = self._find_winner()
    if self.winner is None: self._next_player()
    elif self.verbose: print(colored(f"Only player {self.winner} is remaining.", "green"))

  def __str__(self) -> str:
    ret = f"Ante={self.ante:.2f} bb, Straddle={self.straddle}\n{round_to_str(self.round)} (Pot: {self.pot:.2f} bb)\n"
    for i,p in enumerate(self.players): ret += f"{self._prefix_str(i, 0.0):<17} {('Folded' if p.folded else 'Not folded' if not self.active == i else 'Active')}\n"
    return ret

  def _prefix_str(self, pos:int, amount:float) -> str:
    return f"Player {pos} ({self.players[pos].chips - amount:>6.2f} bb):"

  def _assert_invest(self, amount:float) -> None:
    assert self.players[self.active].chips >= amount, "Not enough chips to invest."

  def _assert_facing_bet(self) -> None:
    assert self.max_bet > 0, "Attempted to face a bet but no bet exists."

  def _assert_act(self) -> None:
    assert not self.players[self.active].folded, "Attempted to act but player already folded."
    assert self.winner is None and self._find_winner() is None, "Attempted to act but there are no opponents left."

  def _next_player(self) -> None:
    once = False
    while self.players[self.active].folded or self.players[self.active].chips == 0 or not once:
      once = True
      self.active = increment(self.active, len(self.players) - 1)
      if self._is_round_complete():
        self._next_round()
        return

  def _next_round(self) -> None:
    self.round += 1
    # if self.verbose: colored(f"{round_to_str(self.round)} ({self.pot:.2f} bb):", "green")
    for p in self.players: p.next_round()
    self.active = self.max_bet = self.bet_level = 0
    if self.round < 4 and (self.players[self.active].folded or self.players[self.active].chips == 0): self._next_player()

  def _is_round_complete(self) -> bool:
    return (self.players[self.active].betsize == self.max_bet and
            (self.max_bet > 0 or self.active == 0) and
            (self.max_bet > self._big_blind_size() or self.active != self._big_blind_idx() or self.round != 0))

  def _find_winner(self) -> int|None:
    return not_folded.index(True) if sum(not_folded := [not p.folded for p in self.players]) == 1 else None

  def _big_blind_size(self) -> float:
    return 2.0 if self.straddle else 1.0

  def _blind_size(self, pos:int) -> float:
    return (0.0, 2.0)[self.straddle] if pos == 2 else 0.0 if pos > 2 else (0.5, 1.0)[pos] if len(self.players) > 2 else (1.0, 0.5)[pos]

  def _big_blind_idx(self) -> int:
    return 0 if len(self.players) == 2 else 2 if self.straddle else 1

