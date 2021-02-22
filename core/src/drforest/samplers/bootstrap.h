#pragma once

#include <utility>

namespace drforest {

    // Generates bootstrap weights.
    class BootstrapSampler {
    public:
        BootstrapSampler(uint num_samples, uint seed=42);
        WeightVec draw();
        std::pair<arma::uvec, WeightVec> draw_indices();
    private:
        uint num_samples_;
        std::uniform_int_distribution<uint> udist_;
        std::mt19937_64 random_state_;
    };

} // namespace drforest
