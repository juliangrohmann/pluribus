from typing import Tuple, Callable
from dataclasses import dataclass

def is_color(rgb: Tuple[int, int, int], idx: int, tol: int) -> bool: return sum(rgb[idx] - rgb[i] < tol for i in range(len(rgb)) if i != idx) == 0
def is_white(rgb: Tuple[int, int, int], min_val: int) -> bool: return sum(v < min_val for v in rgb) == 0

@dataclass(frozen=True)
class PokerConfig:
  name: str
  n_players: int
  table_title: str
  seats: Tuple[Tuple[float, float], ...]
  cards: Tuple[Tuple[float, float], ...]
  active: Tuple[Tuple[float, float], ...]
  is_seat_open: Callable[[Tuple[int, int, int]], bool]
  has_cards: Callable[[Tuple[int, int, int]], bool]
  is_active: Callable[[Tuple[int, int, int]], bool]

pokerstars = PokerConfig(
  name="PokerStars",
  n_players=6,
  table_title="No Limit Hold'em",
  seats=((0.5252, 0.7301), (0.0924, 0.5457), (0.1302, 0.2348), (0.4737, 0.1372), (0.8728, 0.2348), (0.9033, 0.5457)),
  cards=((0.4821, 0.6616), (0.1008, 0.4771), (0.1996, 0.1676), (0.5472, 0.0686), (0.8939, 0.1676), (0.8624, 0.4771)),
  active=((0.4351, 0.7868), (0.0558, 0.6033), (0.0877, 0.2925), (0.4351, 0.1950), (0.7813, 0.2926), (0.8132, 0.6033)),
  is_seat_open=lambda rgb: is_color(rgb, 2, 8),
  has_cards=lambda rgb: is_color(rgb, 0, 30),
  is_active=lambda rgb: is_color(rgb, 0, 20),
)

# pokerstars = {
#   "name": "PokerStars",
#   "table_title": "No Limit Hold'em",
# }
# club_wpt = {
#   "name": "Club WPT Gold",
#   "table_title": "ClubWPT GOLD"
# }
# img: 2733 x 1092
#      3483 x 1393

configs = (pokerstars, )