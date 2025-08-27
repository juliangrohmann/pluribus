#include <pluribus/decision.hpp>

namespace pluribus {

Action LosslessActionProvider::next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board,
    const LosslessBlueprint* bp) const  {
  const hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], state.get_round());
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
  const std::vector<Action> history = state.get_action_history().slice(bp->get_config().init_state.get_action_history().size()).get_history();
  const TreeStorageNode<float>* node = bp->get_strategy()->apply(history);
  const std::atomic<float>* base_ptr = node->get(cluster);

  const auto freq = calculate_strategy(base_ptr, node->get_value_actions().size());
  return node->get_value_actions()[sample_action_idx(freq.data(), freq.size())];
}

Action SampledActionProvider::next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const SampledBlueprint* bp) const {
  const hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], state.get_round());
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
  const std::vector<Action> history = state.get_action_history().slice(bp->get_config().init_state.get_action_history().size()).get_history();
  const TreeStorageNode<uint8_t>* node = bp->get_strategy()->apply(history);
  const uint8_t bias_offset = bp->bias_offset(state.get_biases()[state.get_active()]);
  return bp->decompress_action(node->get(cluster, bias_offset)->load());
}

}
