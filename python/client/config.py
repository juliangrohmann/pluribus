from typing import Tuple, Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class PokerConfig:
  name: str
  table_title: str
  seats: Tuple[Tuple[float, float], ...]
  is_seat_open: Callable[[int, int, int], bool]

pokerstars = PokerConfig(
  name="PokerStars",
  table_title="No Limit Hold'em",
  seats=((0, 0), (0.0597, 0.0597), (0.17715, 0.2838), ),
  is_seat_open=lambda r, g, b: b - g > 8 and b - r > 8,
)

# pokerstars = {
#   "name": "PokerStars",
#   "table_title": "No Limit Hold'em",
# }
# club_wpt = {
#   "name": "Club WPT Gold",
#   "table_title": "ClubWPT GOLD"
}
# img: 2733 x 1092
#      3483 x 1393

configs = (pokerstars)