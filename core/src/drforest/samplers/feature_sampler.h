#pragma once

namespace drforest {

    class FeatureSampler {
    public:
        FeatureSampler(uint num_total_features, uint seed=42);

        // Draws num_features without replacement using rejection sampling
        arma::uvec draw(uint num_features);
    private:
        uint num_total_features_;
        std::uniform_int_distribution<uint> udist_;
        std::mt19937_64 random_state_;
    };

} // namespace drforest
