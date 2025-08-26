#include <iostream>
#include <pluribus/blueprint.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/earth_movers_dist.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/traverse.hpp>

#include "server.hpp"

using namespace pluribus;

void traverse_strategy(RangeViewer* viewer_p, const std::string& fn, const std::string& type) {
  if(type == "--blueprint"){
    Logger::log("Traversing blueprint: " + fn);
    traverse_blueprint(viewer_p, fn);
  }
  else if(type == "--tree") {
    Logger::log("Traversing tree: " + fn);
    traverse_tree(viewer_p, fn);
  }
  else {
    Logger::log("Unknown traverse type: " + type);
  }
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
    return 1;
  }

  if(std::string command = argv[1]; command == "server") {
    // Pluribus server lossless_bp_fn sampled_bp_fn
    PluribusServer server{argv[2], argv[3]};
    server.start();
  }
  else if(command == "ochs-features") {
    // ./Pluribus ochs-features [--blueprint, --real-time] round dir
    if(argc < 5) {
      std::cout << "Missing arguments to build OCHS features.\n";
    }
    else {
      std::function<void(int, const std::string&)> fun;
      if(strcmp(argv[2], "--blueprint") == 0) fun = build_ochs_features;
      else if(strcmp(argv[2], "--real-time") == 0) fun = build_ochs_features_filtered;
      else throw std::runtime_error("Invalid OCHS feature mode: " + std::string{argv[2]});
      if(strcmp(argv[3], "all") == 0) {
        for(int round = 2; round <= 3; ++round) {
          fun(round, argv[4]);
        }
      }
      else {
        fun(atoi(argv[3]), argv[4]);
      }
    }
  }
  else if(command == "emd-matrix") {
    // ./Pluribus emd-matrix start end dir
    if(argc < 5) {
      std::cout << "Missing arguments to build EMD matrix.\n";
    }
    else {
      build_emd_preproc_cache(atoi(argv[2]), atoi(argv[3]), argv[4]);
    }
  }
  else if(command == "build-rt-cluster-map") {
    // ./Pluribus build-rt-cluster-map n_clusters dir
    if(argc < 4) {
      std::cout << "Missing arguments to build real time cluster map.\n";
    }
    else {
      build_real_time_cluster_map(atoi(argv[2]), argv[3]);
    }
  }
  else if(command == "print-clusters") {
    // ./Pluribus print-clusters --blueprint/--real-time
    if(argc < 3) {
      std::cout << "Missing arguments to print clusters.\n";
    }
    else if(strcmp(argv[2], "--blueprint") == 0) {
      print_clusters(true);
    }
    else if(strcmp(argv[2], "--real-time") == 0) {
      print_clusters(false);
    }
    else {
      std::cout << "Invalid print clusters mode: " << argv[2] << "\n";
    }
  }
  else if(command == "traverse") {
    // ./Pluribus traverse --blueprint --png out.png lossless_bp_fn
    // ./Pluribus traverse --tree --png out.png snapshot_fn
    if(argc > 5 && strcmp(argv[3], "--png") == 0) {
      PngRangeViewer viewer{argv[4]};
      traverse_strategy(&viewer, argv[5], argv[2]);
    }
    else if(argc > 3) {
      WindowRangeViewer viewer{"traverse"};
      traverse_strategy(&viewer, argv[3], argv[2]);
    }
    else {
      std::cout << "Missing arguments to traverse strategy.\n";
    }
  }
  else if(command == "blueprint") {
    // ./Pluribus blueprint preflop_snapshot_fn snapshot_dir buf_dir out_fn [--no-preflop]
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprints.\n";
    }
    else {
      bool no_preflop = argc >= 7 && strcmp(argv[6], "--no-preflop") == 0;
      LosslessBlueprint lossless_bp;
      lossless_bp.build(argv[2], get_filepaths(argv[3]), argv[4], !no_preflop);
      std::string lossless_fn = "lossless_" + std::string{argv[5]};
      cereal_save(lossless_bp, lossless_fn);
      SampledBlueprint sampled_bp;
      sampled_bp.build(lossless_fn, argv[4]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});
      lossless_bp.prune_postflop();
      cereal_save(lossless_bp, "preflop_" + std::string{argv[5]});
    }
  }
  else if(command == "blueprint-cached") {
    // ./Pluribus blueprint-cached preflop_buf_fn final_snapshot_fn buf_dir out_fn [--no-preflop]
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprint from cache.\n";
    }
    else {
      bool no_preflop = argc >= 7 && strcmp(argv[6], "--no-preflop") == 0;
      LosslessBlueprint lossless_bp;
      lossless_bp.build_cached(argv[2], argv[3], get_filepaths(argv[4]), !no_preflop);
      std::string lossless_fn = "lossless_" + std::string{argv[5]};
      cereal_save(lossless_bp, lossless_fn);
      SampledBlueprint sampled_bp;
      sampled_bp.build(lossless_fn, argv[4]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});
    }
  }
  else if(command == "blueprint-metadata") {
    // ./Pluribus blueprint-metadata metadata_fn out_fn [--no-preflop]
    if(argc < 4) {
      std::cout << "Missing arguments to build blueprint from metadata.\n";
    }
    else {
      bool no_preflop = argc >= 5 && strcmp(argv[4], "--no-preflop") == 0;
      LosslessMetadata metadata;
      cereal_load(metadata, argv[2]);
      LosslessBlueprint lossless_bp;
      lossless_bp.build_from_meta_data(metadata, !no_preflop);
      std::string lossless_fn = "lossless_" + std::string{argv[3]};
      cereal_save(lossless_bp, lossless_fn);
    }
  }
  else if(command == "sampled-blueprint") {
    // ./Pluribus sampled-blueprint lossless_bp_fn buf_dir out_fn
    if(argc < 5) {
      std::cout << "Missing arguments to build sampled blueprint.\n";
    }
    else {
      SampledBlueprint sampled_bp;
      sampled_bp.build(argv[2], argv[3]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[4]});

    }
  }
  else if(command == "preflop-blueprint") {
    // ./Pluribus preflop-blueprint lossless_bp_fn out_fn
    if(argc < 4) {
      std::cout << "Missing arguments to build preflop blueprint.\n";
    }
    else {
      LosslessBlueprint lossless_bp;
      cereal_load(lossless_bp, argv[2]);
      lossless_bp.prune_postflop();
      cereal_save(lossless_bp, argv[3]);
    }
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
