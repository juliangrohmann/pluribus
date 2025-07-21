#include <cmath>
#include <pluribus/indexing.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/ev.hpp>

namespace pluribus {

void update_stats(const int x, const double w, double& mean, double& w_sum, double& w_sum2, double& S) {
  // Welford's algorithm with weights: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  w_sum += w;
  w_sum2 += pow(w, 2);
  const double mean_old = mean;
  mean = mean_old + w / w_sum * (x - mean_old);
  S = S + w * (x - mean_old) * (x - mean);
}

std::string ResultEV::to_string(const int precision) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << "EV=" << ev << ", stdDev=" << std_dev << ", stdErr=" << std_err 
      << ", iterations=" << iterations << ", time=" << milliseconds << " ms\n";
  return oss.str();
}

ResultEV MonteCarloEV::lossless(const LosslessBlueprint* bp, const PokerState& state, const int i, const std::vector<PokerRange>& ranges,
    const std::vector<uint8_t>& board) {
  const LosslessActionProvider action_provider;
  return _monte_carlo_ev(state, i, ranges, board, bp->get_config().poker.n_chips, action_provider, bp);
}
ResultEV MonteCarloEV::sampled(const std::vector<Action>& biases, const SampledBlueprint* bp, const PokerState& state, const int i,
    const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) {
  const SampledActionProvider action_provider;
  return _monte_carlo_ev(state.apply_biases(biases), i, ranges, board, bp->get_config().poker.n_chips, action_provider, bp);
}

bool MonteCarloEV::_should_terminate(const long t, const double std_err, const Duration dt) const {
  return t >= _min_it && 
      (t >= _max_it || std_err < _std_err_target || std::chrono::duration_cast<std::chrono::milliseconds>(dt).count() > _max_ms);
}

bool any_collision(const uint8_t card, const std::vector<Hand>& hands, const std::vector<uint8_t>& board) {
  for(const auto& hand : hands) {
    if(card == hand.cards()[0] || card == hand.cards()[1]) return true;
  }
  return std::ranges::find(board, card) != board.end();
}

int villain_pos(const PokerState& state, const int i) {
  for(int p = 0; p < state.get_players().size(); ++p) {
    if(p != i && !state.get_players()[p].has_folded()) return p;
  }
  throw std::runtime_error("Villain doesn't exist in state.");
}

void _validate_ev_inputs(const PokerState& state, const int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) {
  const int round = round_of_last_action(state);
  const int n_cards = n_board_cards(round);
  const std::vector<uint8_t> real_board = board.size() > n_cards ? std::vector<uint8_t>{board.begin(), board.begin() + n_cards} : board;
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

double node_ev(const TreeStorageNode<float>* node, const SolverConfig& config, const PokerState& state, const int i, const std::vector<Hand>& hands, const Board& board,
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    const int hu = utility(state, i, Board{board}, hands, config.poker.n_chips, config.rake, eval);
    return hu;
  }
  const hand_index_t cached_idx = indexers[state.get_active()].index(board, hands[state.get_active()], state.get_round());
  const int cached_cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), cached_idx);
  const auto& actions = node->get_actions();
  double ev = 0.0;
  for(int aidx = 0; aidx < actions.size(); ++aidx) {
    ev += node->get(cached_cluster, aidx)->load() * node_ev(node->apply_index(aidx), config, state.apply(actions[aidx]), i, hands, board, indexers, eval);
  }
  return ev;
}

double enumerate_ev(const LosslessBlueprint& bp, const PokerState& state, const int i, const std::vector<PokerRange>& ranges,
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

  const int pos_v = villain_pos(state, i);
  const omp::HandEvaluator eval;
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
        for(int _ = 0; i < hands.size(); ++_) indexers.push_back(CachedIndexer{});
        const double freq = ranges[i].frequency(hh) * ranges[pos_v].frequency(vh);
        ev += freq * node_ev(bp.get_strategy(), bp.get_config(), state, i, hands, board, indexers, eval);
        total += freq;
      }
    }
  }
  return ev / total;
}

}
