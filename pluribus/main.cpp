#include <iostream>
#include <map>
#include <functional>

#include <pluribus/poker.hpp>

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
    return 1;
  }

  std::string command = argv[1];

  if(command == "cluster") {
    std::cout << "Clustering..." << std::endl;
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
