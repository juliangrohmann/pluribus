#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <tqdm/tqdm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/util.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

RegretStorage::RegretStorage(int n_players, int n_chips, int ante, int n_clusters, int n_actions) : _n_clusters{n_clusters}, _n_actions{n_actions} {
  HistoryIndexer::initialize(n_players, n_chips, ante);
  std::cout << "Counting histories... ";
  size_t n_histories = HistoryIndexer::size(n_players, n_chips, ante);
  std::cout << n_histories << "\n";
  _size = n_histories * n_clusters * n_actions;
  std::cout << "Opening regret map file... " << std::flush;
  _fd = open("atomic_regret.dat", O_RDWR | O_CREAT | O_TRUNC, 0666);
  if(_fd == -1) {
    throw std::runtime_error("Failed to map file.");
  }

  size_t file_size = _size * sizeof(std::atomic<int>);
  std::cout << "Resizing file to " << file_size << " bytes... " << std::flush;
  if(ftruncate(_fd, file_size) == -1) {
    close(_fd);
    throw std::runtime_error("Failed to resize file.");
  }

  std::cout << "Mapping file... ";
  void* ptr = mmap(NULL, _size * sizeof(std::atomic<int>), PROT_READ | PROT_WRITE, MAP_PRIVATE, _fd, 0);
  if(ptr == MAP_FAILED) {
    close(_fd);
    throw std::runtime_error("Failed to map file to memory.");
  }
  _data = static_cast<std::atomic<int>*>(ptr);

  std::cout << "Initializing regrets... " << std::flush;
  #pragma omp parallel for schedule(static, 1)
  for(size_t i = 0; i < _size; ++i) {
    // std::cout << i << '\n' << std::flush;
    (_data + i)->store(0, std::memory_order_relaxed);
  }
  std::cout << "Success.\n";
}

RegretStorage::~RegretStorage() {
  if(_data) munmap(_data, _size * sizeof(std::atomic<int>));
  if(_fd != -1) close(_fd);
}

std::atomic<int>* RegretStorage::operator[](const InformationSet& info_set) {
  return _data + info_offset(info_set);
}

const std::atomic<int>* RegretStorage::operator[](const InformationSet& info_set) const {
  return _data + info_offset(info_set);
}

bool RegretStorage::operator==(const RegretStorage& other) const {
  if(_size != other._size) return false;
  for(size_t i = 0; i < _size; ++i) {
    if(*(_data + i) != *(other._data + i)) return false;
  }
  return true;
}

size_t RegretStorage::info_offset(const InformationSet& info_set) const {
  return (static_cast<size_t>(info_set.get_history_idx()) * _n_clusters + info_set.get_cluster()) * _n_actions;
}

std::vector<float> calculate_strategy(std::atomic<int>* regret_p, int n_actions) {
  int sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    sum += std::max((regret_p + a_idx)->load(), 0);
  }

  std::vector<float> freq;
  freq.reserve(n_actions);
  if(sum > 0) {
    for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
      freq.push_back(std::max((regret_p + a_idx)->load(), 0) / static_cast<double>(sum));
    }
  }
  else {
    for(int i = 0; i < n_actions; ++i) {
      freq.push_back(1.0 / n_actions);
    }
  }
  return freq;
}

int sample_action_idx(const std::vector<float>& freq) {
  std::discrete_distribution<> dist(freq.begin(), freq.end());
  return dist(GlobalRNG::instance());
}

void lcfr_discount(RegretStorage& data, double d) {
  for(auto it = data.data(); it != data.data() + data.size(); ++it) {
    it->store(it->load() * d);
  }
}

void lcfr_discount(PreflopMap& data, double d) {
  for(auto& entry : data) {
    for(int i = 0; i < entry.second.size(); ++i) {
      entry.second[i] *= d;
    }
  }
}

std::string minutes_str(int val) {
  return " (" + std::to_string(val / it_per_min) + " m)";
}

BlueprintTrainer::BlueprintTrainer(int n_players, int n_chips, int ante, long strategy_interval, long preflop_threshold, long snapshot_interval, 
                                   long prune_thresh, int prune_cutoff, int regret_floor, long lcfr_thresh, long discount_interval, long log_interval) : 
    _t{1}, _regrets{n_players, n_chips, ante, 200, 5}, _phi{}, _n_players{n_players}, _n_chips{n_chips}, _ante{ante}, 
    _strategy_interval{strategy_interval}, _preflop_threshold{preflop_threshold}, _snapshot_interval{snapshot_interval}, _prune_thresh{prune_thresh}, 
    _prune_cutoff{prune_cutoff}, _regret_floor{regret_floor}, _lcfr_thresh{lcfr_thresh}, _discount_interval{discount_interval}, _log_interval{log_interval} {
  log_state();
}

void BlueprintTrainer::mccfr_p(long T) {
  if(verbose) omp_set_num_threads(1);
  long limit = _t + T;
  long next_discount = _discount_interval;
  long next_snapshot = _preflop_threshold;
  std::cout << "Training blueprint from " << _t << " to " << limit << "\n";
  while(_t < limit) {
    long init_t = _t;
    _t = std::min(std::min(next_discount, next_snapshot), limit);
    std::cout << "Next step: " << _t << "\n";
    #pragma omp parallel for schedule(static, 1)
    for(long t = init_t; t < _t; ++t) {
      thread_local omp::HandEvaluator eval;
      thread_local Deck deck;
      thread_local Board board;
      thread_local std::vector<Hand> hands{static_cast<size_t>(_n_players)};
      if(verbose) std::cout << "============== t = " << t << " ==============\n";
      if(t % _log_interval == 0) log_metrics(t);
      for(int i = 0; i < _n_players; ++i) {
        if(verbose) std::cout << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck);
        for(Hand& hand : hands) hand.deal(deck);

        if(t <= _preflop_threshold && t % _strategy_interval == 0) {
          if(verbose) std::cout << "============== Updating strategy ==============\n";
          update_strategy(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
        }
        if(t > _prune_thresh) {
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          float q = dist(GlobalRNG::instance());
          if(q < 0.05f) {
            if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
            traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
          }
          else {
            if(verbose) std::cout << "============== Traverse MCCFR-P ==============\n";
            traverse_mccfr_p(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
          }
        }
        else {
          if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
        }
      }
    }

    if(_t == next_discount) {
      std::cout << "============== Discounting ==============\n";
      double d = (_t / _discount_interval) / (_t / _discount_interval + 1);
      std::cout << "Discount factor: " << d << "\n";
      lcfr_discount(_regrets, d);
      lcfr_discount(_phi, d);
    }
    if(_t == _preflop_threshold) {
      std::cout << "============== Saving & freezing preflop strategy ==============\n";
      std::ostringstream oss;
      oss << date_time_str() << "_preflop.bin";
      cereal_save(*this, oss.str());
    }
    else if(_t > _preflop_threshold && _t == next_snapshot) {
      std::cout << "============== Saving snapshot ==============\n";
      std::ostringstream oss;
      oss << date_time_str() << "_t" << std::fixed << std::setprecision(1) << static_cast<double>(_t) / 1'000'000 << "M.bin";
      cereal_save(*this, oss.str());
    }
    
    next_discount = next_discount + _discount_interval < _lcfr_thresh ? next_discount + _discount_interval : limit + 1;
    next_snapshot += _snapshot_interval;
  }
}

int BlueprintTrainer::traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    return utility(state, i, board, hands, eval);
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state);
    auto action_regret_p = _regrets[info_set];
    auto freq = calculate_strategy(action_regret_p, actions.size());
    std::unordered_map<Action, int> values;
    int v = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if((action_regret_p + a_idx)->load() > _prune_cutoff) {
        int v_a = traverse_mccfr_p(state.apply(a), i, board, hands, eval);
        values[a] = v_a;
        v += freq[a_idx] * v_a;
      }
    }
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(values.find(a) != values.end()) {
        int next_r = (action_regret_p + a_idx)->load() + values[a] - v;
        (action_regret_p + a_idx)->store(std::max(next_r, _regret_floor));
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    Action a = actions[sample_action_idx(freq)];
    return traverse_mccfr_p(state.apply(a), i, board, hands, eval);
  }
}

int BlueprintTrainer::traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    int u = utility(state, i, board, hands, eval);
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\tu(z) = " << u << "\n";
    }
    return u;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    std::unordered_map<Action, int> values;
    int v = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      int v_a = traverse_mccfr(state.apply(a), i, board, hands, eval);
      values[a] = v_a;
      v += freq[a_idx] * v_a;
      if(verbose) {
        std::cout << state.get_action_history().to_string() << "\n";
        std::cout << "\t u(" << action_to_str.at(a) << ") @ " << freq[a_idx] << " = " << v_a << "\n";
      }
    }
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\t u(sigma) = " << v << "\n";
    }
    auto action_regret_p = _regrets[info_set];
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      int dR = values[actions[a_idx]] - v;
      int next_r = (action_regret_p + a_idx)->load() + dR;
      (action_regret_p + a_idx)->store(std::max(next_r, _regret_floor));
      if(verbose) {
        std::cout << "\t R(" << action_to_str.at(actions[a_idx]) << ") = " << dR << "\n";
        std::cout << "\t cum R(" << action_to_str.at(actions[a_idx]) << ") = " << (action_regret_p + a_idx)->load() << "\n";
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    Action a = actions[sample_action_idx(freq)];
    return traverse_mccfr(state.apply(a), i, board, hands, eval);
  }
}

void BlueprintTrainer::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    int a_idx = sample_action_idx(freq);
    auto& phi_regrets = _phi[info_set];
    if(phi_regrets.size() == 0) {
      phi_regrets.grow_by(actions.size());
    }
    #pragma omp critical
    phi_regrets[a_idx] += 1.0f;
    update_strategy(state.apply(actions[a_idx]), i, board, hands);
  }
  else {
    for(Action action : valid_actions(state)) {
      update_strategy(state.apply(action), i, board, hands);
    }
  }
}

int BlueprintTrainer::utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const {
  if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - _n_chips + (state.get_winner() == i ? state.get_pot() : 0);
  }
  else if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - _n_chips + showdown_payoff(state, i, board, hands, eval);
  }
  else {
    throw std::runtime_error("Non-terminal state does not have utility.");
  }
}

int BlueprintTrainer::showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const {
  if(state.get_players()[i].has_folded()) return 0;
  std::vector<uint8_t> win_idxs = winners(state, hands, board, eval);
  return std::find(win_idxs.begin(), win_idxs.end(), i) != win_idxs.end() ? state.get_pot() / win_idxs.size() : 0;
}

void BlueprintTrainer::log_metrics(long t) const {
  std::cout << std::setprecision(1) << std::fixed << "t=" << t / 1'000'000.0 << "M\n";
}

void BlueprintTrainer::log_state() const {
  std::cout << "N players: " << _n_players << "\n";
  std::cout << "N chips: " << _n_chips << "\n";
  std::cout << "Ante: " << _ante << "\n";
  std::cout << "Iterations/s: " << it_per_sec << "\n";
  std::cout << "Iterations/min: " << it_per_min << "\n";
  std::cout << "Strategy interval: " << _strategy_interval << "\n";
  std::cout << "Preflop threshold: " << _preflop_threshold << minutes_str(_preflop_threshold) << "\n";
  std::cout << "Snapshot interval: " << _snapshot_interval << minutes_str(_snapshot_interval) << "\n";
  std::cout << "Prune threshold: " << _prune_thresh << minutes_str(_prune_thresh) << "\n";
  std::cout << "Prune cutoff: " << _prune_cutoff << "\n";
  std::cout << "Regret floor: " << _regret_floor << "\n";
  std::cout << "LCFR threshold: " << _lcfr_thresh << minutes_str(_lcfr_thresh) << "\n";
  std::cout << "Discount interval: " << _discount_interval << minutes_str(_discount_interval) << "\n";
  std::cout << "Log interval: " << _log_interval << minutes_str(_log_interval) << "\n";
}

}

