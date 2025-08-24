import re
from typing import Tuple, Callable
from dataclasses import dataclass

def is_color(rgb: Tuple[int, int, int], idx: int, tol: int) -> bool: return sum(rgb[idx] - rgb[i] < tol for i in range(len(rgb)) if i != idx) == 0
def is_white(rgb: Tuple[int, int, int], min_val: int) -> bool: return sum(v < min_val for v in rgb) == 0
def is_exact_rgb(rgb: Tuple[int, int, int], target: Tuple[int, int, int], tol: int): return sum(abs(v - t) > tol for v,t in zip(rgb, target)) == 0
def is_uniform(rgb: Tuple[int, int, int], tol: int): max_v = max(rgb); return sum(abs(v - max_v) > tol for v in rgb) == 0

@dataclass(frozen=True)
class SiteConfig:
  board_suits: Tuple[Tuple[float,float], ...]
  board_ranks: Tuple[Tuple[float,float, float,float], ...]
  pot: Tuple[int, int, int, int]
  is_seat_open: Callable[[Tuple[int, int, int]], bool]
  has_cards: Callable[[Tuple[int, int, int]], bool]
  has_bet: Callable[[Tuple[int, int, int]], bool]
  is_active: Callable[[Tuple[int, int, int]], bool]
  has_button: Callable[[Tuple[int, int, int]], bool]
  is_spade: Callable[[Tuple[int, int, int]], bool]
  is_club: Callable[[Tuple[int, int, int]], bool]
  is_heart: Callable[[Tuple[int, int, int]], bool]
  is_diamond: Callable[[Tuple[int, int, int]], bool]
  get_blinds: Callable[[str], Tuple[float, ...] | None]
  get_ante: Callable[[str], float]


@dataclass(frozen=True)
class PokerConfig:
  name: str
  site: SiteConfig
  n_players: int
  seats: Tuple[Tuple[float, float], ...]
  cards: Tuple[Tuple[float, float], ...]
  active: Tuple[Tuple[float, float], ...]
  button: Tuple[Tuple[float, float], ...]
  hole_cards_suits: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
  hole_cards_ranks: Tuple[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]], ...]
  bet_chips: Tuple[Tuple[float, float], ...]
  bet_size: Tuple[Tuple[float, float, float, float], ...]
  stacks: Tuple[Tuple[float, float, float, float], ...]
  usernames: Tuple[Tuple[float, float, float, float], ...]

pokerstars = SiteConfig(
  board_suits=((0.3675, 0.3327), (0.4347, 0.3327), (0.5018, 0.3327), (0.5690, 0.3327), (0.6361, 0.3327)),
  board_ranks=((0.3382, 0.3327, 0.3711, 0.3823), (0.4054, 0.3327, 0.4383, 0.3823), (0.4725, 0.3327, 0.5054, 0.3823),
               (0.5397, 0.3327, 0.5725, 0.3823), (0.6068, 0.3327, 0.6397, 0.3823)),
  pot=(0.3821, 0.2955, 0.5788, 0.3204),
  is_seat_open=lambda rgb: is_color(rgb, 2, 8),
  has_cards=lambda rgb: is_color(rgb, 0, 30),
  has_bet=lambda rgb: not is_color(rgb, 1, 40) or rgb[1] >= 100,
  is_active=lambda rgb: is_color(rgb, 0, 20),
  has_button=lambda rgb: is_white(rgb, 190),
  is_spade=lambda rgb: is_exact_rgb(rgb, (108, 109, 109), 10) or is_exact_rgb(rgb, (70, 71, 71), 10),
  is_club=lambda rgb: is_exact_rgb(rgb, (112, 168, 76), 10) or is_exact_rgb(rgb, (73, 108, 51), 10),
  is_heart=lambda rgb: is_exact_rgb(rgb, (157, 70, 70), 10) or is_exact_rgb(rgb, (102, 44, 44), 10),
  is_diamond=lambda rgb: is_exact_rgb(rgb, (79, 136, 155), 10),
  get_blinds=lambda title: tuple(float(v) for v in match.group(1, 2)) if (match := re.search(r"No Limit Hold'em \$([0-9.]+)/\$([0-9.]+)", title)) else None,
  get_ante=lambda title: 0.0
)

pokerstars_6p = PokerConfig(
  name="PokerStars 6-max",
  site=pokerstars,
  n_players=6,
  seats=((0.5252, 0.7301), (0.0924, 0.5457), (0.1302, 0.2348), (0.4737, 0.1372), (0.8728, 0.2348), (0.9033, 0.5457)),
  cards=((0.4821, 0.6616), (0.1008, 0.4771), (0.1996, 0.1676), (0.5472, 0.0686), (0.8939, 0.1676), (0.8624, 0.4771)),
  active=((0.4351, 0.7868), (0.0558, 0.6033), (0.0877, 0.2925), (0.4351, 0.1950), (0.7813, 0.2926), (0.8132, 0.6033)),
  button=((0.4048, 0.6314), (0.2064, 0.4777), (0.2212, 0.3487), (0.5439, 0.2595), (0.7879, 0.3405), (0.7389, 0.5785)),
  hole_cards_suits=(((0.5245, 0.7355), (0.5952, 0.7355)),
                    ((0.0888, 0.4477), (0.1537, 0.4477)),
                    ((0.1211, 0.1377), (0.186, 0.1377)),
                    ((0.4681, 0.0401), (0.5330, 0.0401)),
                    ((0.8138, 0.1377), (0.8787, 0.1377)),
                    ((0.8463, 0.4477), (0.9112, 0.4477))),
  hole_cards_ranks=(((0.4766, 0.7173, 0.5029, 0.7603), (0.5473, 0.7173, 0.6180, 0.7603)),
                    ((0.0612, 0.4338, 0.0912, 0.4965), (0.1248, 0.4338, 0.1548, 0.4965)),
                    ((0.0936, 0.1220, 0.1236, 0.1847), (0.1572, 0.1220, 0.1872, 0.1847)),
                    ((0.4406, 0.0244, 0.4706, 0.0871), (0.5042, 0.0244, 0.5342, 0.0871)),
                    ((0.7863, 0.1220, 0.8163, 0.1847), (0.8499, 0.1220, 0.8799, 0.1847)),
                    ((0.8188, 0.4338, 0.8488, 0.4965), (0.8824, 0.4338, 0.9124, 0.4965))),
  bet_chips=((0.4424, 0.5769), (0.2405, 0.5207), (0.2509, 0.3107), (0.5313, 0.2116), (0.7286, 0.2810), (0.7594, 0.5207)),
  bet_size=((0.4595, 0.5686, 0.6295, 0.5917), (0.2600, 0.5097, 0.4300, 0.5328), (0.2690, 0.2992, 0.4390, 0.3223),
            (0.5485, 0.2017, 0.7185, 0.2248), (0.5448, 0.2694, 0.7148, 0.2925), (0.5732, 0.5080, 0.7432, 0.5345)),
  stacks=((0.4721, 0.7322, 0.5724, 0.7686), (0.0445, 0.5488, 0.1505, 0.5851), (0.0775, 0.2380, 0.1824, 0.2760),
          (0.4230, 0.1405, 0.5291, 0.1769), (0.8176, 0.2380, 0.9324, 0.2743), (0.8483, 0.5488, 0.9544, 0.5851)),
  usernames=((0.4621, 0.6852, 0.5924, 0.7216), (0.0245, 0.5018, 0.1605, 0.5381), (0.0575, 0.1910, 0.1924, 0.2290),
             (0.4030, 0.0935, 0.5391, 0.1299), (0.8076, 0.1910, 0.9424, 0.2290), (0.8383, 0.5018, 0.9744, 0.5381)),
)

configs = (pokerstars_6p, )