#include <chrono>
#include <cnpy.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>
#include <cereal/types/array.hpp>
#include <hand_isomorphism/hand_index.h>
#include <omp/CardRange.h>
#include <omp/EquityCalculator.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/constants.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/util.hpp>
#include <tqdm/tqdm.hpp>

#include "debug.hpp"

namespace pluribus {

void assign_features(const std::string& hand, const std::string& board, float* data) {
  const omp::Hand board_hand = omp::Hand::empty() + omp::Hand(board);
  for(int i = 0; i < 8; ++i) {
    *(data + i) = equity(omp::Hand(hand), omp::CardRange(ochs_categories[i]), board_hand);
  }
}

std::array<double, 2> eval(const omp::Hand& hero, const omp::CardRange& vill_rng, const omp::Hand& board) {
  const omp::HandEvaluator evaluator;
  std::array<double, 2> results = {0, 0};

  const int hero_val = evaluator.evaluate(hero + board);
  const omp::Hand dead_cards = hero + board;
  for(const auto& combo : vill_rng.combinations()) {
    omp::Hand villain = omp::Hand(combo[0]) + omp::Hand(combo[1]);
    if(dead_cards.contains(villain)) {
      continue;
    }
    if(const int vill_val = evaluator.evaluate(villain + board); hero_val > vill_val) results[0] += 1;
    else if(hero_val < vill_val) results[1] += 1;
    else {
      results[0] += 0.5;
      results[1] += 0.5;
    }
  }
  return results;
}

std::array<double, 2> enumerate(const omp::Hand& hero, const omp::CardRange& villain, const omp::Hand& board) {
  if(board.count() == 5) {
    return eval(hero, villain, board);
  }
  std::array<double, 2> results = {0, 0};
  for(int idx = 0; idx < MAX_CARDS; ++idx) {
    auto card = omp::Hand(idx);
    if(hero.contains(card) || board.contains(card)) continue;
    auto tmp_res = enumerate(hero, villain, board + card);
    results[0] += tmp_res[0];
    results[1] += tmp_res[1];
  }
  return results;
}

double equity(const omp::Hand& hero, const omp::CardRange& villain, const omp::Hand& board) {
  const auto results = enumerate(hero, villain, board);
  return results[0] / (results[0] + results[1]);
}

std::string progress_str(const hand_index_t idx, const hand_index_t total, const std::chrono::steady_clock::time_point& t_0) {
  const auto dt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t_0).count();
  const double percent = static_cast<double>(idx) / total;
  std::ostringstream oss;
  oss << std::setw(11) << std::to_string(idx) << ":   "
      << std::fixed << std::setprecision(1) << std::setw(5) << (percent * 100) << "%"
      << "    Elapsed: " << std::setw(7) << std::setprecision(0) << dt << " s"
      << "    Remaining: " << std::setw(7) << 1 / percent * dt - dt << " s";
  return oss.str();
}

void map_index_to_features(const hand_index_t idx, const int round, const int card_sum, float* data) {
  uint8_t cards[7] = {};
  HandIndexer::get_instance()->unindex(idx, cards, round);
  const std::string hand = cards_to_str(cards, 2);
  const std::string board = cards_to_str(cards + 2, card_sum - 2);
  assign_features(hand, board, data);
}

void solve_features(const int round, const int card_sum, const hand_index_t total, const std::function<hand_index_t(hand_index_t)>& get_index,
    const std::string& fn, const bool verbose) {
  Logger::log("Solving features for " + std::to_string(total) + " indexes...");
  const hand_index_t chunk_size = std::max(total / omp_get_max_threads(), 1ul);
  const hand_index_t log_interval = std::max(chunk_size / 100, 1ul);
  std::vector<float> feature_map(total * 8);
  const std::chrono::steady_clock::time_point t_0 = std::chrono::steady_clock::now();

  #pragma omp parallel for schedule(static)
  for(hand_index_t i = 0; i < total; ++i) {
    const hand_index_t idx = get_index(i);
    if(verbose) {
      if(const int tid = omp_get_thread_num(); tid == 0 && idx % log_interval == 0) {
        std::ostringstream oss;
        oss << std::right << " (round " << round << ") " << progress_str(i, chunk_size, t_0);
        Logger::dump(oss);
      }
    }
    map_index_to_features(idx, round, card_sum, &feature_map[i * 8]);
  }
  Logger::log("Writing features to " + fn);
  cnpy::npy_save(fn, feature_map.data(), {total, 8}, "w");
}

void solve_features(const int round, const int card_sum, const hand_index_t start, const hand_index_t end,
    const std::string& fn, const bool verbose) {
  return solve_features(round, card_sum, end - start, [start](const int i) {return start + i; }, fn, verbose);
}

void solve_features(const int round, const int card_sum, const std::vector<hand_index_t>& indexes, const std::string& fn, const bool verbose) {
  return solve_features(round, card_sum, indexes.size(), [&indexes](const int i) {return indexes[i]; }, fn, verbose);
}

void build_ochs_features(const int round, const std::string& dir) {
  if(round < 1 || round > 3) Logger::error("Cannot build filtered OCHS features for round " + std::to_string(round) + ".");
  Logger::log("Building OCHS features: " + round_to_str(round));
  const int card_sum = n_board_cards(round) + 2;
  const size_t n_idx = HandIndexer::get_instance()->size(round);
  if(round == 3) {
    constexpr int n_batches = 10;
    const size_t batch_size = n_idx / n_batches;
    for(int batch = 0; batch < n_batches; ++batch) {
      Logger::log("Launching batch " + std::to_string(batch) + "...");
      solve_features(round, card_sum, batch * batch_size, batch == n_batches - 1 ? n_idx : (batch + 1) * batch_size,
          std::filesystem::path{dir} / (std::string("features_") + std::to_string(round) + "_b" + std::to_string(batch) + ".npy"), true);
    }
  }
  else {
    solve_features(round, card_sum, 0, n_idx, std::filesystem::path{dir} / (std::string("features_") + std::to_string(round) + ".npy"), true);
  }
}

void collect_indexes(const int i, const int round, const int max_cards, const uint64_t mask, uint8_t cards[7], std::unordered_set<hand_index_t>& indexes) {
  if(i < max_cards) {
    for(uint8_t card = i == 1 ? cards[i - 1] + 1 : 0; card < MAX_CARDS; ++card) {
      if(const auto curr_mask = card_mask(card); !(mask & curr_mask)) {
        cards[i] = card;
        collect_indexes(i == 1 ? 5 : i + 1, round, max_cards, mask | curr_mask, cards, indexes);
      }
    }
  }
  else {
    // std::cout << cards_to_str(cards, max_cards) << "\n";
    indexes.insert(HandIndexer::get_instance()->index(cards, round));
  }
}

void build_ochs_features_filtered(const int round, const std::string& dir) {
  if(round < 1 || round > 3) Logger::error("Cannot build filtered OCHS features for round " + std::to_string(round) + ".");
  Logger::log("Building filtered OCHS features: " + round_to_str(round));
  for(hand_index_t flop_idx = 0; flop_idx < NUM_DISTINCT_FLOPS; ++flop_idx) {
    std::unordered_set<hand_index_t> index_set;
    std::array<uint8_t, 7> cards{};
    FlopIndexer::get_instance()->unindex(flop_idx, cards.data() + 2);
    std::string flop = cards_to_str(cards.data() + 2, 3);
    const int card_sum = n_board_cards(round) + 2;
    if(card_sum > 7) Logger::error("Invalid card sum.");
    constexpr int n_flop_cards = 3;
    Logger::log("Collecting indexes for flop: " + flop);
    collect_indexes(0, round, card_sum, card_mask(cards.data() + 2, n_flop_cards), cards.data(), index_set);
    std::vector<hand_index_t> indexes{index_set.begin(), index_set.end()};
    std::string infix = "r" + std::to_string(round) + "_f" + std::to_string(flop_idx);
    cereal_save(indexes, std::filesystem::path{dir} / ("indexes_" + infix + ".bin"));
    Logger::log("Building OCHS features for flop: " + flop + " (" + std::to_string(indexes.size()) + " indexes)");
    solve_features(round, card_sum, indexes, std::filesystem::path{dir} / ("features_" + infix + ".npy"), false);
  }
}

std::string cluster_filename(const int round, const int n_clusters, const int split) {
  const std::string base = "clusters_r" + std::to_string(round) + "_c" + std::to_string(n_clusters);
  return base + (round == 3 ? "_p" + std::to_string(split) + ".npy": ".npy");
}

std::vector<uint16_t> load_clusters(const int round, const int n_clusters, const int split) {
  return cnpy::npy_load(cluster_filename(round, n_clusters, split)).as_vec<uint16_t>();
}

std::array<std::vector<uint16_t>, 4> init_flat_cluster_map(const int n_clusters) {
  std::cout << "Initializing flat cluster map (n_clusters=" << n_clusters << ")...\n";
  std::array<std::vector<uint16_t>, 4> cluster_map;
  cluster_map[0].resize(169);
  std::iota(cluster_map[0].begin(), cluster_map[0].end(), 0);
  for(int i = 1; i <= 3; ++i) {
    std::cout << "(Flat: " << n_clusters << " clusters) Loading round " << i << "...\n";
    cluster_map[i] = load_clusters(i, n_clusters, 1);
  }
  auto s2 = cnpy::npy_load(cluster_filename(3, n_clusters, 2)).as_vec<uint16_t>();
  const size_t s1_size = cluster_map[3].size();
  cluster_map[3].resize(cluster_map[3].size() + s2.size());
  std::ranges::copy(s2, cluster_map[3].data() + s1_size);
  std::cout << "Loaded all clusters.\n";
  return cluster_map;
}

std::unique_ptr<BlueprintClusterMap> BlueprintClusterMap::_instance = nullptr;

BlueprintClusterMap::BlueprintClusterMap() {
  _cluster_map = init_flat_cluster_map(200);
}

uint16_t RealTimeClusterMap::cluster(const int round, const hand_index_t flop_index, const hand_index_t hand_index) const {
  return _cluster_map[flop_index][round].at(hand_index);
}

uint16_t RealTimeClusterMap::cluster(const int round, const Board& board, const Hand& hand) const {
  const hand_index_t flop_index = FlopIndexer::get_instance()->index(board.cards().data());
  return cluster(round, flop_index, HandIndexer::get_instance()->index(board, hand, round));
}

RealTimeClusterMap::RealTimeClusterMap() {

}

}
