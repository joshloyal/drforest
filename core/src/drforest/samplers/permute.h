#pragma once

#include <utility>

namespace drforest {

    // Generates bootstrap weights.
    class Permuter {
    public:
        Permuter(uint num_samples, uint seed=42);
        arma::uvec draw();
    private:
        uint num_samples_;
        std::mt19937_64 random_state_;
    };

} // namespace drforest
