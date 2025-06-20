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
  oss << std::fixed << std::setprecision(precision) << "EV=" << ev << ", stdDev=" << std_dev << ", stdErr" << std_err 
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

}
