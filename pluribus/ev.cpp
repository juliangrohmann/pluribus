#include <cmath>
#include <pluribus/indexing.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/ev.hpp>

namespace pluribus {

void update_stats(int x, double w, double& mean, double& w_sum, double& w_sum2, double& S) {
  // Welford's algorithm with weights: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  w_sum += w;
  w_sum2 += pow(w, 2);
  double mean_old = mean;
  mean = mean_old + (w / w_sum) * (x - mean_old);
  S = S + w * (x - mean_old) * (x - mean);
}

class LosslessActionProvider : public _ActionProvider<float> {
public:
  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<float>* bp) const override {
    auto actions = valid_actions(state, bp->get_config().action_profile);
    hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], static_cast<int>(state.get_round()));
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
    size_t base_idx = bp->get_strategy().index(state, cluster);
    auto freq = calculate_strategy(bp->get_strategy(), base_idx, actions.size());
    return actions[sample_action_idx(freq)];
  }
};

class SampledActionProvider : public _ActionProvider<Action> {
public:
  SampledActionProvider(const std::vector<int>& bias_offsets) : _bias_offsets{bias_offsets} {}

  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<Action>* bp) const override {
    hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], static_cast<int>(state.get_round()));
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
    size_t base_idx = bp->get_strategy().index(state, cluster);
    return bp->get_strategy()[base_idx + _bias_offsets[state.get_active()]];
  }
private:
  std::vector<int> _bias_offsets;
};

std::string ResultEV::to_string(int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << "EV=" << ev << ", stdDev=" << std_dev << ", stdErr=" << std_err 
      << ", iterations=" << iterations << ", time=" << milliseconds << " ms\n";
  return oss.str();
}

ResultEV MonteCarloEV::lossless(const LosslessBlueprint* bp, const PokerState& state, int i, const std::vector<PokerRange>& ranges, 
    const std::vector<uint8_t>& board) {
  LosslessActionProvider action_provider;
  return _monte_carlo_ev(state, i, ranges, board, bp->get_config().poker.n_chips, action_provider, bp);
}
ResultEV MonteCarloEV::sampled(const std::vector<Action>& biases, const SampledBlueprint* bp, const PokerState& state, int i, 
    const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) {
  BiasActionProfile bias_profile;
  const std::vector<Action>& all_biases = bias_profile.get_actions(0, 0, 0, 0);
  std::vector<int> bias_offsets;
  for(const auto& bias : biases) {
    bias_offsets.push_back(std::distance(all_biases.begin(), std::find(all_biases.begin(), all_biases.end(), bias)));
  }
  SampledActionProvider action_provider{bias_offsets};
  return _monte_carlo_ev(state, i, ranges, board, bp->get_config().poker.n_chips, action_provider, bp);
}

bool MonteCarloEV::_should_terminate(long t, double std_err, Duration dt) {
  return t >= _min_it && 
      (t >= _max_it || std_err < _std_err_target || std::chrono::duration_cast<std::chrono::milliseconds>(dt).count() > _max_ms);
}

bool any_collision(uint8_t card, const std::vector<Hand>& hands, const std::vector<uint8_t>& board) {
  for(const auto& hand : hands) {
    if(card == hand.cards()[0] || card == hand.cards()[1]) return true;
  }
  return std::find(board.begin(), board.end(), card) != board.end();
}

int villain_pos(const PokerState& state, int i) {
  for(int p = 0; p < state.get_players().size(); ++p) {
    if(p != i && !state.get_players()[p].has_folded()) return p;
  }
  throw std::runtime_error("Villain doesn't exist in state.");
}

void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) {
  int round = round_of_last_action(state);
  int n_cards = n_board_cards(round);
  std::vector<uint8_t> real_board = board.size() > n_cards ? std::vector<uint8_t>{board.begin(), board.begin() + n_cards} : board;
  std::cout << "Real round: " << round_to_str(round) << "\n";
  std::cout << "Real board: " << cards_to_str(real_board) << "\n";
  std::cout << "Hero pos: " << i << " (" << pos_to_str(i, state.get_players().size()) << ")\n";
  int ridx = 0;
  for(int p = 0; p < state.get_players().size(); ++p) {
    if(!state.get_players()[p].has_folded()) {
      std::cout << pos_to_str(p, state.get_players().size()) << ": " << ranges[ridx].n_combos() << " combos  ";
      ++ridx;
      if(ridx >= ranges.size()) break;
    }
  }
  std::cout << "\nHero combos: " << ranges[i].n_combos() << "\n"; 
  if(state.active_players() != 2) throw std::runtime_error("Expected value is only possible with two remaining players.");
  if(board.size() > n_board_cards(round)) throw std::runtime_error("Too many board cards!");
}

double node_ev(const LosslessBlueprint& bp, const PokerState& state, int i, const std::vector<Hand>& hands, const Board& board,
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    int hu = utility(state, i, Board{board}, hands, bp.get_config().poker.n_chips, eval);
    return hu;
  }
  else {
    const auto& strat = bp.get_strategy();
    hand_index_t cached_idx = indexers[state.get_active()].index(board, hands[state.get_active()], state.get_round());
    int cached_cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), cached_idx);
    int base_idx = strat.index(state, cached_cluster);
    auto actions = valid_actions(state, bp.get_config().action_profile);
    double ev = 0.0;
    for(int aidx = 0; aidx < actions.size(); ++aidx) {
      ev += strat[base_idx + aidx] * node_ev(bp, state.apply(actions[aidx]), i, hands, board, indexers, eval);
    }
    return ev;
  }
}

double enumerate_ev(const LosslessBlueprint& bp, const PokerState& state, int i, const std::vector<PokerRange>& ranges, 
    const std::vector<uint8_t>& init_board) {
  _validate_ev_inputs(state, i, ranges, init_board);
  std::vector<Board> boards;
  if(init_board.size() == 4) {
    for(uint8_t c = 0; c < MAX_CARDS; ++c) {
      if(collides(c, init_board)) continue;
      auto next_board = init_board;
      next_board.push_back(c);
      boards.push_back(Board{next_board});
    }
  }
  else if(init_board.size() == 5) {
    boards.push_back(Board{init_board});
  }
  else {
    throw std::runtime_error("Enumerate EV only supported for Turn/River.");
  }

  int pos_v = villain_pos(state, i);
  omp::HandEvaluator eval;
  std::vector<Hand> hands(ranges.size());
  double ev = 0.0;
  double total = 0.0;
  double max_combos = boards.size();
  for(const auto& r : ranges) max_combos *= r.n_combos();
  for(const auto& board : boards) {
    std::cout << "Enumerate EV: " << std::fixed << std::setprecision(1) << total / max_combos * 100 << "%\n";
    for(const auto& hh : ranges[i].hands()) {
      if(collides(hh, board)) continue;
      for(const auto& vh : ranges[pos_v].hands()) {
        if(collides(hh, vh) || collides(vh, board)) continue;
        hands[i] = hh;
        hands[pos_v] = vh;
        std::vector<CachedIndexer> indexers;
        for(int i = 0; i < hands.size(); ++i) indexers.push_back(CachedIndexer{});
        double freq = ranges[i].frequency(hh) * ranges[pos_v].frequency(vh);
        ev += freq * node_ev(bp, state, i, hands, board, indexers, eval);
        total += freq;
      }
    }
  }
  return ev / total;
}

}
