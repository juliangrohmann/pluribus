#pragma once
#include <limits>
#include <pluribus/poker.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

using Duration = std::chrono::high_resolution_clock::duration;

inline double standard_deviation(double S, double w_sum) { return pow(S / w_sum, 0.5); }
void update_stats(int x, double w, double& mean, double& w_sum, double& w_sum2, double& S); 

template <class T>
class _ActionProvider {
public:
  virtual Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<T>* bp) const = 0;
};

struct ResultEV {
  double ev;
  double std_dev;
  double std_err;
  long iterations;
  long milliseconds;

  std::string to_string(int precision = 2);
};

class MonteCarloEV {
public:
  MonteCarloEV* set_min_iterations(long n) { _min_it = n; return this; };
  MonteCarloEV* set_max_iterations(long n) { _max_it = n; return this; };
  MonteCarloEV* set_std_err_target(double std_err) { _std_err_target = std_err; return this; }
  MonteCarloEV* set_time_limit(double max_ms) { _max_ms = max_ms; return this; };
  MonteCarloEV* set_verbose(bool verbose) { _verbose = verbose; return this; }
  ResultEV lossless(const LosslessBlueprint* bp, const PokerState& state, int i, const std::vector<PokerRange>& ranges, 
      const std::vector<uint8_t>& board);
  ResultEV sampled(const std::vector<Action>& biases, const SampledBlueprint* bp, const PokerState& state, int i, 
      const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

private:
  bool _should_terminate(long t, double std_err, Duration dt);
  
  template <class T>
  ResultEV _monte_carlo_ev(const PokerState& init_state, int i, const std::vector<PokerRange>& ranges, 
    const std::vector<uint8_t>& init_board, int stack_size, const _ActionProvider<T>& action_provider, const Blueprint<T>* bp) {
    _validate_ev_inputs(init_state, i, ranges, init_board);
    RoundSampler sampler{ranges, init_board};
    omp::HandEvaluator eval;
    double std_err = 0.0, mean = 0.0, w_sum = 0.0, w_sum2 = 0.0, S = 0.0;
    auto sample = sampler.sample();
    auto t_0 = std::chrono::high_resolution_clock::now();
    long t = 0;
    while(!_should_terminate(t, std_err, std::chrono::high_resolution_clock::now() - t_0)) {
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
      }
      int u = utility(state, i, board, sample.hands, stack_size, eval);
      update_stats(u, sample.weight, mean, w_sum, w_sum2, S);
      std_err = pow(S / (pow(w_sum, 2) - w_sum2), 0.5);
      if(_verbose && t > 0 && t % 100'000 == 0) {
        auto t_i = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_i - t_0).count();
        std::cout << std::fixed << std::setprecision(1) << "t=" << t / 1'000'000.0 << "M, " 
                  << std::setprecision(2) << "EV=" << mean << ", "
                  << "stdDev=" << standard_deviation(S, w_sum) << ", "
                  << "stdErr=" << std_err << " ("
                  << std::setprecision(1) << static_cast<double>(t + 1) / (dt / 1'000.0) << "k it/sec)\n";
      }
      ++t;
    }
    ResultEV result{mean, standard_deviation(S, w_sum), std_err, t, 
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