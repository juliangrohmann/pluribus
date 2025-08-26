import warnings
from argparse import ArgumentParser
from typing import Tuple
from tqdm import tqdm
from pokerkit import StandardHighHand, State, Street, Automation, BettingStructure, Opening, Deck

warnings.filterwarnings("ignore")

def utility_vec(stacks:Tuple[int,...], hands:Tuple[str,...], board:str, actions:Tuple[str,...]):
  state = State(
    (Automation.ANTE_POSTING,
      Automation.BET_COLLECTION,
      Automation.BLIND_OR_STRADDLE_POSTING,
      Automation.CARD_BURNING,
      Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
      Automation.HAND_KILLING,
      Automation.CHIPS_PUSHING,
      Automation.CHIPS_PULLING),
    Deck.STANDARD,
    (StandardHighHand,),
    (Street(False, (False, False), 0, False, Opening.POSITION, 50, None),
      Street(False, tuple(), 3, False, Opening.POSITION, 1, None),
      Street(False, tuple(), 1, False, Opening.POSITION, 1, None),
      Street(False, tuple(), 1, False, Opening.POSITION, 1, None)),
    BettingStructure.NO_LIMIT, True, 0, (50, 100), 0, stacks, 6)

  for a in actions:
    if a == 'D0':
      for hand in hands: state.deal_hole(hand)
    elif a == 'D1': state.deal_board(board[:6])
    elif a == 'D2': state.deal_board(board[6:8])
    elif a == 'D3': state.deal_board(board[8:10])
    elif a == 'F': state.fold()
    elif a == 'C': state.check_or_call()
    else: state.complete_bet_or_raise_to(int(a))
  return [end - start for start,end in zip(stacks, state.stacks)]

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("src", default="../../resources/testset.pokerkit")
  parser.add_argument("out", default="../../resources/utility_sidepots.pokerkit")
  args = parser.parse_args()
  with open(args.out, 'w') as out_f:
    with open(args.src, 'r') as in_f:
      for line in tqdm(in_f, total=100_000):
        tokens = line.strip().split(' ')
        it, tokens = tokens[0], tokens[1:]
        stacks = [int(t) for t in tokens[:6]]
        hands = tokens[6:12]
        board = tokens[12]
        actions = tokens[13:]
        out_f.write(' '.join(str(u) for u in utility_vec(stacks, hands, board, actions)) + '\n')
