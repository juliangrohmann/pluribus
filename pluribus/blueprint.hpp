#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

template <class T>
class Blueprint : public Strategy<T> {
public:
  Blueprint() : _freq{nullptr} {}

  const StrategyStorage<T>& get_strategy() const {
    if(_freq) return *_freq;
    throw std::runtime_error("Blueprint strategy is null.");
  }
  const BlueprintTrainerConfig& get_config() const { return _config; }
  void set_config(BlueprintTrainerConfig config) { _config = config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq);
  }

protected:
  void assign_freq(StrategyStorage<T>* freq) { _freq = std::unique_ptr<StrategyStorage<T>>{freq}; }
  std::unique_ptr<StrategyStorage<T>>& get_freq() { return _freq; }

private:
  std::unique_ptr<StrategyStorage<T>> _freq;
  BlueprintTrainerConfig _config;
};

class LosslessBlueprint : public Blueprint<float> {
public:
  void build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir = "");
  double enumerate_ev(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const;
  double monte_carlo_ev(long n, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board, bool verbose = false) const;

private:
  double node_ev(const PokerState& state, int i, const std::vector<Hand>& hands, const Board& board, int stack_size, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const;
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor);
void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

class SampledBlueprint : public Blueprint<Action> {
public:
  void build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor = 5.0f);
  double monte_carlo_ev(long n, const std::vector<Action>& biases, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board, bool verbose = false) const;
};

template <class T>
class _ActionProvider {
public:
  virtual Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<T>& bp) const = 0;
};

template <class T>
double _monte_carlo_ev(long n, const PokerState& init_state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& init_board, int stack_size, const _ActionProvider<T>& action_provider, const Blueprint<T>& bp, bool verbose) {
  _validate_ev_inputs(init_state, i, ranges, init_board);
  // std::vector<long> samples(MAX_COMBOS, 0L);
  RoundSampler sampler{ranges, init_board};
  omp::HandEvaluator eval;
  double value = 0.0;
  double total_weight = 0.0;
  auto sample = sampler.sample();
  auto t_0 = std::chrono::high_resolution_clock::now();
  for(long t = 0; t < n; ++t) {
    sampler.next_sample(sample);
    Board board{init_board};
    int board_idx = init_board.size();
    uint64_t init_mask = sample.mask;
    while(board_idx < 5) {
      uint8_t next_card = gsl_rng_uniform_int(GSLGlobalRNG::instance(), MAX_CARDS);
      uint64_t curr_mask = card_mask(next_card);
      if(!(init_mask & curr_mask)) {
        board.set_card(board_idx++, next_card);
        init_mask |= curr_mask;
      }
    }

    std::vector<CachedIndexer> indexers(ranges.size(), CachedIndexer{});
    PokerState state = init_state;
    while(!state.is_terminal()) {
      state = state.apply(action_provider.next_action(indexers[state.get_active()], state, sample.hands, board, bp));
      // if(state.get_action_history().size() == 1 && state.get_action_history().get(0) == Action::FOLD) {
        // ++samples[HoleCardIndexer::get_instance()->index(sample.hands[0])];
      // }
    }
    int u = utility(state, i, board, sample.hands, stack_size, eval);
    value += sample.weight * u;
    total_weight += sample.weight;
    if(verbose && t > 0 && t % 100'000 == 0) {
      auto t_i = std::chrono::high_resolution_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_i - t_0).count();
      std::cout << std::fixed << std::setprecision(1) << "t=" << t / 1'000'000.0 << "M, " 
                << std::setprecision(2) << "EV=" << value / t + 1 << ", "
                << std::setprecision(1) << static_cast<double>(t + 1) / (dt / 1'000.0) << "k it/sec)\n";
    }
  }

  // long sum = 0L;
  // for(long n : samples) sum += n;
  // std::cout << "Sum=" << sum << "\n";
  // for(int hidx = 0; hidx < samples.size(); ++hidx) {
  //   std::cout << HoleCardIndexer::get_instance()->hand(hidx).to_string() << ": " << std::setprecision(4) << static_cast<double>(samples[hidx]) / sum * MAX_COMBOS << "\n";
  // }
  return value / total_weight;
}

}
