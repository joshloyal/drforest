#include <vector>

#include "drforest.h"

namespace drforest {

    Permuter::Permuter(uint num_samples, uint seed) :
            num_samples_(num_samples) {
        random_state_.seed(seed);
    }

    arma::uvec Permuter::draw() {
        arma::uvec permute = arma::regspace<arma::uvec>(0, num_samples_ - 1);
        std::shuffle(permute.begin(), permute.end(), random_state_);

        return permute;
   }

}  // namespace drforest
