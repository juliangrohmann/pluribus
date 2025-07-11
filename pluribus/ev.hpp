#pragma once
#include <limits>
#include <pluribus/poker.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

using Duration = std::chrono::high_resolution_clock::duration;

void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);
inline double standard_deviation(const double S, const double w_sum) { return pow(S / w_sum, 0.5); }
void update_stats(int x, double w, double& mean, double& w_sum, double& w_sum2, double& S); 
double enumerate_ev(const LosslessBlueprint& bp, const PokerState& state, int i, const std::vector<PokerRange>& ranges, 
    const std::vector<uint8_t>& init_board);

struct ResultEV {
  double ev;
  double std_dev;
  double std_err;
  long iterations;
  long milliseconds;

  std::string to_string(int precision = 2) const;
};

class MonteCarloEV {
public:
  MonteCarloEV* set_min_iterations(const long n) { _min_it = n; return this; }
  MonteCarloEV* set_max_iterations(const long n) { _max_it = n; return this; }
  MonteCarloEV* set_std_err_target(const double std_err) { _std_err_target = std_err; return this; }
  MonteCarloEV* set_time_limit(const double max_ms) { _max_ms = max_ms; return this; }
  MonteCarloEV* set_verbose(const bool verbose) { _verbose = verbose; return this; }
  ResultEV lossless(const LosslessBlueprint* bp, const PokerState& state, int i, const std::vector<PokerRange>& ranges, 
      const std::vector<uint8_t>& board);
  ResultEV sampled(const std::vector<Action>& biases, const SampledBlueprint* bp, const PokerState& state, int i,
      const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

private:
  bool _should_terminate(long t, double std_err, Duration dt) const;
  
  template <typename BlueprintT>
  ResultEV _monte_carlo_ev(const PokerState& init_state, const int i, const std::vector<PokerRange>& ranges,
    const std::vector<uint8_t>& init_board, const int stack_size, const ActionProvider<BlueprintT>& action_provider, const BlueprintT* bp) {
    _validate_ev_inputs(init_state, i, ranges, init_board);
    RoundSampler sampler{ranges, init_board};
    const omp::HandEvaluator eval;
    double std_err = 0.0, mean = 0.0, w_sum = 0.0, w_sum2 = 0.0, S = 0.0;
    auto sample = sampler.sample();
    const auto t_0 = std::chrono::high_resolution_clock::now();
    long t = 0;
    while(!_should_terminate(t, std_err, std::chrono::high_resolution_clock::now() - t_0)) {
      sampler.next_sample(sample);
      Board board = sample_board(init_board, sample.mask);

      std::vector indexers(ranges.size(), CachedIndexer{});
      PokerState state = init_state;
      while(!state.is_terminal() && !state.get_players()[i].has_folded()) {
        state = state.apply(action_provider.next_action(indexers[state.get_active()], state, sample.hands, board, bp));
      }
      const int u = utility(state, i, board, sample.hands, stack_size, bp->get_config().rake, eval);
      update_stats(u, sample.weight, mean, w_sum, w_sum2, S);
      std_err = pow(S / (pow(w_sum, 2) - w_sum2), 0.5);
      if(_verbose && t > 0 && t % 100'000 == 0) {
        auto t_i = std::chrono::high_resolution_clock::now();
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_i - t_0).count();
        std::cout << std::fixed << std::setprecision(1) << "t=" << t / 1'000'000.0 << "M, " 
                  << std::setprecision(2) << "EV=" << mean << ", "
                  << "stdDev=" << standard_deviation(S, w_sum) << ", "
                  << "stdErr=" << std_err << " ("
                  << std::setprecision(1) << static_cast<double>(t + 1) / (dt / 1'000.0) << "k it/sec)\n";
      }
      ++t;
    }
    const ResultEV result{mean, standard_deviation(S, w_sum), std_err, t,
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_0).count()};
    if(_verbose) std::cout << result.to_string() << "\n";
    return result;
  }

  long _min_it = 1000;
  long _max_it = std::numeric_limits<long>::max();
  double _std_err_target = 0.0;
  double _max_ms = 3'600'000.0;
  bool _verbose = false;
};

}