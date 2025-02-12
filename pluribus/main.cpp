#include <iostream>
#include <map>
#include <functional>
#include <cstdlib>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>

using namespace pluribus;

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
    return 1;
  }

  std::string command = argv[1];

  if(command == "cluster") {
    int round = atoi(argv[2]);
    if(round < 1 || round > 3) {
      std::cout << "1 <= round <= 3 required. Given: " << round << std::endl;
    }
    else {
      std::cout << "Clustering round " << round << "..." << std::endl;
      build_ochs_features(round);
    }
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
