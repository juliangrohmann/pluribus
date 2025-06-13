#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sys/sysinfo.h>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/util.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

struct LosslessMetadata {
  BlueprintTrainerConfig config;
  tbb::concurrent_unordered_map<ActionHistory, HistoryEntry> history_map;
  std::vector<std::string> buffer_fns;
  size_t max_regrets = 0;
  int n_clusters = -1;
};

struct LosslessBuffer {
  std::vector<float> freqs;
  size_t offset;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(freqs, offset);
  }
};

LosslessMetadata build_lossless_buffers(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  std::cout << "Building lossless buffers...\n";
  LosslessMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  
  int buf_idx = 0;
  for(int bp_idx = 0; bp_idx < postflop_fns.size(); ++bp_idx) {
    auto bp = cereal_load<BlueprintTrainer>(postflop_fns[bp_idx]);
    const auto& regrets = bp.get_strategy();
    if(bp_idx == 0) {
      meta.config = bp.get_config();
      meta.n_clusters = regrets.n_clusters();
      std::cout << "Initialized blueprint config:\n";
      std::cout << "n_clusters=" << meta.n_clusters << "\n";
      std::cout << "max_actions=" << meta.config.action_profile.max_actions() << "\n";
      std::cout << meta.config.to_string();
    }

    std::vector<size_t> base_idxs;
    base_idxs.reserve(regrets.history_map().size());
    for(const auto& entry : regrets.history_map()) {
      base_idxs.push_back(entry.second.idx);
    }
    base_idxs.push_back(regrets.data().size());
    std::sort(base_idxs.begin(), base_idxs.end());

    size_t free_ram = get_free_ram();
    if(free_ram < 8 * pow(1024, 3)) {
      throw std::runtime_error("At least 8G free RAM required to build blueprint. Available (bytes): " + std::to_string(free_ram));
    }
    size_t buf_sz = static_cast<size_t>((free_ram - 8 * pow(1024, 3)) / sizeof(float));
    std::cout << "Blueprint " << bp_idx << " buffer: " << std::setprecision(2) << std::fixed << buf_sz << " elements\n";

    if(regrets.data().size() > meta.max_regrets) {
      meta.max_regrets = regrets.data().size();
      std::cout << "New max regrets: " << meta.max_regrets << "\n";
      meta.history_map = regrets.history_map();
      std::cout << "New history map size: " << meta.history_map.size() << "\n";
    }

    size_t bidx_start = 0;
    size_t bidx_end = 0;
    while(bidx_start < base_idxs.size() - 1) {
      for(; bidx_end < base_idxs.size() - 1; ++bidx_end) {
        if(base_idxs[bidx_end] - base_idxs[bidx_start] > buf_sz) break;
      }
      if(bidx_end == bidx_start) throw std::runtime_error("Failed to increase bidx_end");

      LosslessBuffer buffer;
      buffer.offset = base_idxs[bidx_start];
      buffer.freqs.resize(base_idxs[bidx_end] - base_idxs[bidx_start]);

      std::cout << "Storing buffer: base_idxs[" << bidx_start << ", " << bidx_end << "), indeces: [" 
                << base_idxs[bidx_start] << ", " << base_idxs[bidx_end] << ")\n";
      #pragma omp parallel for schedule(dynamic)
      for(size_t curr_idx = bidx_start; curr_idx < bidx_end; ++curr_idx) {
        if(curr_idx + 1 >= base_idxs.size()) throw std::runtime_error("Buffering: Indexing base indeces out of range!");
        size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
        if(n_entries % meta.n_clusters != 0) throw std::runtime_error("Buffering: Indivisible regret section!");
        size_t n_actions = n_entries / meta.n_clusters;
        if(n_actions > meta.config.action_profile.max_actions()) throw std::runtime_error("Buffering: Too many actions in storage section:" + 
            std::to_string(n_actions) + " > " + std::to_string(meta.config.action_profile.max_actions()));

        for(int c = 0; c < meta.n_clusters; ++c) {
          size_t base_idx = base_idxs[curr_idx] + c * n_actions;
          auto freq = calculate_strategy(regrets, base_idx, n_actions);
          for(int fidx = 0; fidx < freq.size(); ++fidx) {
            buffer.freqs[base_idx - base_idxs[bidx_start] + fidx] = freq[fidx];
          }
        }
      }
      bidx_start = bidx_end;

      std::string fn = "lossless_buf_" + std::to_string(buf_idx++) + ".bin";
      meta.buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
    }
  }
  return meta;
}

void LosslessBlueprint::build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  std::filesystem::path buffer_dir = buf_dir;
  LosslessMetadata meta = build_lossless_buffers(preflop_fn, postflop_fns, buf_dir);
  set_config(meta.config);
  assign_freq(new StrategyStorage<float>{meta.config.action_profile, meta.n_clusters});
  get_freq()->data().resize(meta.max_regrets);
  for(std::string buf_fn : meta.buffer_fns) {
    auto buf = cereal_load<LosslessBuffer>((buffer_dir / buf_fn).string());
    std::cout << "Accumulating " << buf_fn << ": [" << buf.offset << ", " << buf.offset + buf.freqs.size() << ")\n";
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.freqs.size(); ++idx) {
      get_freq()->operator[](buf.offset + idx).store(get_freq()->operator[](buf.offset + idx).load() + buf.freqs[idx]);
    }
  }

  std::cout << "Inserting histories...\n";
  for(const auto& entry : meta.history_map) {
    get_freq()->history_map()[entry.first] = entry.second;
  }

  std::cout << "Normalizing frequencies...\n";
  std::vector<size_t> base_idxs;
  base_idxs.reserve(meta.history_map.size());
  for(const auto& entry : meta.history_map) {
    base_idxs.push_back(entry.second.idx);
  }
  base_idxs.push_back(get_freq()->data().size());
  std::sort(base_idxs.begin(), base_idxs.end());

  for(size_t curr_idx = 0; curr_idx < base_idxs.size() - 1; ++curr_idx) {
    if(curr_idx + 1 >= base_idxs.size()) throw std::runtime_error("Renorm: Indexing base indeces out of range!");
    size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
    if(n_entries % meta.n_clusters != 0) throw std::runtime_error("Renorm: Indivisible storage section!");
    size_t n_actions = n_entries / meta.n_clusters;
    if(n_actions > get_config().action_profile.max_actions()) throw std::runtime_error("Renorm: Too many actions in storage section!");
    for(int c = 0; c < meta.n_clusters; ++c) {
      size_t base_idx = base_idxs[curr_idx] + c * n_actions;
      float total = 0.0f;
      for(size_t aidx = 0; aidx < n_actions; ++aidx) {
        total += get_freq()->operator[](base_idx + aidx);
      }
      if(total > 0) {
        for(size_t aidx = 0; aidx < n_actions; ++aidx) {
          get_freq()->operator[](base_idx + aidx).store(get_freq()->operator[](base_idx + aidx).load() / total);
        }
      }
      else {
        for(size_t aidx = 0; aidx < n_actions; ++aidx) {
          get_freq()->operator[](base_idx + aidx).store(1.0 / n_actions);
        }
      }
    }
  }
  std::cout << "Lossless blueprint built.\n";
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

double LosslessBlueprint::enumerate_ev(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const {
  _validate_ev_inputs(state, i, ranges, board);

  int pos_v = villain_pos(state, i);
  omp::HandEvaluator eval;
  std::vector<Hand> hands(ranges.size());
  double ev = 0.0;
  double total = 0.0;
  double max_combos = 1.0;
  for(const auto& r : ranges) max_combos *= r.n_combos();
  for(const auto& hh : ranges[i].hands()) {
    if(collides(hh, board)) continue;
    std::cout << "Enumerate EV: " << std::fixed << std::setprecision(1) << total / max_combos * 100 << "%\n";
    for(const auto& vh : ranges[pos_v].hands()) {
      if(collides(hh, vh) || collides(vh, board)) continue;
      hands[i] = hh;
      hands[pos_v] = vh;
      std::vector<CachedIndexer> indexers;
      for(int i = 0; i < hands.size(); ++i) indexers.push_back(CachedIndexer{3});
      double freq = ranges[i].frequency(hh) * ranges[pos_v].frequency(vh);
      ev += freq * node_ev(state, i, hands, board, get_config().poker.n_chips, indexers, eval);
      total += freq;
    }
  }
  return ev / total;
}

double LosslessBlueprint::node_ev(const PokerState& state, int i, const std::vector<Hand>& hands, const std::vector<uint8_t>& board, int stack_size, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const {
  if(board.size() < n_board_cards(state.get_round())) {
    double ev = 0.0;
    int total = 0;
    for(uint8_t card = 0; card < 52; ++card) {
      if(any_collision(card, hands, board)) continue;
      auto next_board = board;
      next_board.push_back(card);
      auto indexers_copy = indexers;
      ev += node_ev(state, i, hands, next_board, stack_size, indexers_copy, eval);
      ++total;
    }
    return ev / total;
  }
  else if(state.is_terminal()) {
    int hu = utility(state, i, Board{board}, hands, stack_size, eval);
    return hu;
  }
  else {
    const auto& strat = get_strategy();
    hand_index_t cached_idx = indexers[state.get_active()].index(board, hands[state.get_active()], state.get_round());
    int cached_cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), cached_idx);
    int base_idx = strat.index(state, cached_cluster);
    auto actions = valid_actions(state, get_config().action_profile);
    double ev = 0.0;
    for(int aidx = 0; aidx < actions.size(); ++aidx) {
      ev += strat[base_idx + aidx] * node_ev(state.apply(actions[aidx]), i, hands, board, stack_size, indexers, eval);
    }
    return ev;
  }
}

class LosslessActionProvider : public _ActionProvider<float> {
public:
  Action next_action(const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<float>& bp) const override {
    auto actions = valid_actions(state, bp.get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[state.get_active()]);
    size_t base_idx = bp.get_strategy().index(state, cluster);
    auto freq = calculate_strategy(bp.get_strategy(), base_idx, actions.size());
    return actions[sample_action_idx(freq)];
  }
};

class SampledActionProvider : public _ActionProvider<Action> {
public:
  SampledActionProvider(const std::vector<int>& bias_offsets) : _bias_offsets{bias_offsets} {}

  Action next_action(const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<Action>& bp) const override {
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[state.get_active()]);
    size_t base_idx = bp.get_strategy().index(state, cluster);
    return bp.get_strategy()[base_idx + _bias_offsets[state.get_active()]];
  }
private:
  std::vector<int> _bias_offsets;
};

double LosslessBlueprint::monte_carlo_ev(int n, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const {
  LosslessActionProvider action_provider;
  return _monte_carlo_ev(n, state, i, ranges, board, get_config().poker.n_chips, action_provider, *this);
}

double SampledBlueprint::monte_carlo_ev(int n, const std::vector<Action>& biases, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const {
  BiasActionProfile bias_profile;
  const std::vector<Action>& all_biases = bias_profile.get_actions(0, 0, 0, 0);
  std::vector<int> bias_offsets;
  for(const auto& bias : biases) {
    bias_offsets.push_back(std::distance(all_biases.begin(), std::find(all_biases.begin(), all_biases.end(), bias)));
  }
  SampledActionProvider action_provider{bias_offsets};
  return _monte_carlo_ev(n, state, i, ranges, board, get_config().poker.n_chips, action_provider, *this);
}

struct SampledMetadata {
  BlueprintTrainerConfig config;
  std::vector<std::string> buffer_fns;
  std::vector<ActionHistory> histories;
  std::vector<Action> biases;
  int n_clusters = -1;
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor) {
  std::vector<float> biased_freq;
  if(bias == Action::BIAS_FOLD || bias == Action::BIAS_CALL) {
    Action biased_action = bias == Action::BIAS_FOLD ? Action::FOLD : Action::CHECK_CALL;
    auto fold_it = std::find(actions.begin(), actions.end(), biased_action);
    if(fold_it != actions.end()) {
      size_t fold_idx = std::distance(actions.begin(), fold_it);
      for(int fidx = 0; fidx < freq.size(); ++fidx) {
        biased_freq.push_back(fidx == fold_idx ? freq[fidx] * factor : freq[fidx]);
      }
    }
    else {
      biased_freq = freq;
    }
  }
  else if(bias == Action::BIAS_RAISE) {
    for(int fidx = 0; fidx < freq.size(); ++fidx) {
      biased_freq.push_back(actions[fidx].get_bet_type() > 0 || actions[fidx] == Action::ALL_IN ? freq[fidx] * factor : freq[fidx]);
    }
  }
  else if(bias == Action::BIAS_NONE) {
    biased_freq = freq;
  }
  else {
    throw std::runtime_error("Unknown bias: " + bias.to_string());
  }
  float sum = 0.0f;
  for(float f : biased_freq) sum += f;
  for(int fidx = 0; fidx < biased_freq.size(); ++fidx) biased_freq[fidx] /= sum;
  return biased_freq;
}

Action sample_biased(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor) {
  auto b_freq = biased_freq(actions, freq, bias, factor);
  std::discrete_distribution<> dist(b_freq.begin(), b_freq.end());
  return actions[dist(GlobalRNG::instance())];
}

SampledMetadata build_sampled_buffers(const std::string& lossless_bp_fn, const std::string& buf_dir, float factor) {
  std::cout << "Building sampled buffers...\n";
  SampledMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  LosslessBlueprint bp = cereal_load<LosslessBlueprint>(lossless_bp_fn);
  meta.config = bp.get_config();
  BiasActionProfile bias_profile;
  meta.biases = bias_profile.get_actions(0, 0, 0, 150);
  meta.n_clusters = bp.get_strategy().n_clusters();

  std::cout << "Collecting histories... " << std::flush;
  meta.histories.reserve(bp.get_strategy().history_map().size());
  // size_t fold_saved = 0, bet_saved = 0;;
  PokerState init_state{meta.config.poker};
  for(const auto& entry : bp.get_strategy().history_map()) {
    PokerState state = init_state.apply(entry.first);
    if(!state.is_terminal()) {
      meta.histories.push_back(entry.first);
    }
    else {
      throw std::runtime_error("Found terminal state.");
    }
    // auto actions = valid_actions(state, bp.get_config().action_profile);
    // int bets = 0;
    // for(Action a : actions) {
    //   if(a.get_bet_type() > 0 || a == Action::ALL_IN) ++bets;
    // }
    // if(std::find(actions.begin(), actions.end(), Action::FOLD) == actions.end()) {
    //   fold_saved += 200;
    // }
    // if(bets == 0) {
    //   bet_saved += 200;
    // }
  }
  // std::cout << "Fold saved: " << fold_saved << '\n';
  // std::cout << "Bet saved: " << bet_saved << '\n';
  std::cout << "Collected " << meta.histories.size() << " (" << meta.n_clusters << " clusters).\n";

  std::cout << "Clusters=" << meta.n_clusters << ", Biases=" << meta.biases.size() << "\n";
  std::cout << "Building buffers...\n";
  std::unordered_map<ActionHistory, std::vector<Action>> buffer;
  long long min_ram = 4 * pow(1024, 3);
  int buf_idx = 0;
  for(size_t hidx = 0; hidx < meta.histories.size(); ++hidx) {
    std::vector<Action> sampled;
    sampled.reserve(meta.n_clusters * meta.biases.size());
    for(int c = 0; c < meta.n_clusters; ++c) {
      PokerState state = init_state.apply(meta.histories[hidx]);
      auto actions = valid_actions(state, bp.get_config().action_profile);
      size_t base_idx = bp.get_strategy().index(state, c);
      auto freq = calculate_strategy(bp.get_strategy(), base_idx, actions.size());
      for(Action bias : meta.biases) {
        sampled.push_back(sample_biased(actions, freq, bias, factor));
      }
    }
    buffer[meta.histories[hidx]] = sampled;

    if(get_free_ram() < min_ram || hidx == meta.histories.size() - 1) {
      std::cout << "Buffered " << buffer.size() << " histories.\n";
      if(buffer.size() == 0) throw std::runtime_error("Out of RAM but buffer is empty.");
      std::string fn = "sampled_buf_" + std::to_string(buf_idx++) + ".bin";
      meta.buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
      buffer.clear();
    }
  }
  std::cout << "Buffers built.\n";
  return meta;
}

void SampledBlueprint::build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor) {
  {
    LosslessBlueprint bp = cereal_load<LosslessBlueprint>(lossless_bp_fn);
    std::cout << "Lossless data size: " << bp.get_strategy().data().size() << "\n";
  }
  SampledMetadata meta = build_sampled_buffers(lossless_bp_fn, buf_dir, bias_factor);
  std::filesystem::path buffer_dir = buf_dir;
  assign_freq(new StrategyStorage<Action>(BiasActionProfile{}, meta.n_clusters));
  for(const auto& fn : meta.buffer_fns) {
    auto buffer = cereal_load<std::unordered_map<ActionHistory, std::vector<Action>>>((buffer_dir / fn).string());
    for(size_t hidx = 0; hidx < meta.histories.size(); ++hidx) {
      PokerState state{meta.config.poker};
      state = state.apply(meta.histories[hidx]);
      size_t base_idx = get_freq()->index(state, 0);
      auto sampled = buffer[meta.histories[hidx]];
      for(int idx = 0; idx < sampled.size(); ++idx) {
        get_freq()->operator[](base_idx + idx) = sampled[idx];
      }
    }
  }
  std::cout << "Stored histories: " << get_freq()->history_map().size() << "\n";
  std::cout << "Data size: " << get_freq()->data().size() << "\n";
  std::cout << "Sampled blueprint built.\n";
}

}
