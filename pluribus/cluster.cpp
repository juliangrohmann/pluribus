#include <string>
#include <omp/EquityCalculator.h>
#include <omp/CardRange.h>
#include <pluribus/cluster.hpp>

namespace pluribus {

std::array<float, 8> features(omp::EquityCalculator& calc, const std::string& hand, const std::string& board) {
  std::string parsed_hand;
  if(hand.length() == 4) {
    parsed_hand = std::string(1, hand[0]) + hand[2] + (hand[1] == hand[3] ? 's' : 'o');
  }
  else {
    parsed_hand = hand;
  }

  std::array<double, 8> feat = {};
  for(int i = 0; i < 8; ++i) {
    std::vector<omp::CardRange> ranges = {omp::CardRange(parsed_hand), omp::CardRange(ochs_categories[i])};
    calc.start(ranges, omp::CardRange::getCardMask(board), 0, board.length() != 0);
    calc.wait();
    feat[i] = calc.getResults().equity[0];
  }
  return feat;
}

}