#include <pluribus/blueprint.hpp>
#include <pluribus/cereal_ext.hpp>

using namespace pluribus;

int main(int argc, char* argv[]) {
  LosslessBlueprint bp;
  cereal_load(bp, argv[1]);
  auto tree_config = std::make_shared{ClusterSpec{169, 200, 200, 200}, ActionMode::make_blueprint_mode(bp.get_config().action_profile)};
  bp.get_mutable_strategy()->set_config(tree_config);
  bp.get_mutable_strategy()->make_root();
  cereal_save(bp, std::string{"updated_"} + argv[1]);
}
