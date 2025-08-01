#include <pluribus/blueprint.hpp>
#include <pluribus/cereal_ext.hpp>

using namespace pluribus;

int main(int argc, char* argv[]) {
  if(argc < 4) throw std::runtime_error("straddle arg required");
  const bool straddle = strcmp(argv[3], "true") == 0;
  std::cout << "straddle=" << (straddle ? "true" : "false");
  LosslessBlueprint bp;
  cereal_load(bp, argv[1]);
  const auto tree_config = std::make_shared<TreeStorageConfig>(ClusterSpec{169, 200, 200, 200},
    ActionMode::make_blueprint_mode(bp.get_config().action_profile));
  bp.get_mutable_strategy()->set_config(tree_config);
  bp.get_mutable_strategy()->make_root();
  bp.get_mutable_config().poker.straddle = straddle;
  bp.get_mutable_config().init_state = PokerState{bp.get_mutable_config().poker};
  cereal_save(bp, argv[2]);
}
