#include <string>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

void print_cards(uint8_t cards[], int n);
void print_cluster(int cluster, int round, int n_clusters);
void print_similar_boards(std::string board, int n_clusters=200);

}
