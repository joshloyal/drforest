#include <random>

#include "drforest.h"

namespace drforest {

    Categorical::Categorical(uint seed) {
        random_state_.seed(seed);
    }

    uint Categorical::draw(arma::vec &weights) {
        std::discrete_distribution<uint> categorical(weights.begin(), weights.end());
        return categorical(random_state_);
    }

}  // namespace drforest
