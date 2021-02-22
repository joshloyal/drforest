#pragma once

namespace drforest {

    class RandomStateSampler {
    public:
        RandomStateSampler(uint num_samples, uint seed=42);
        std::vector<uint> draw();
    private:
        uint num_samples_;
        std::uniform_int_distribution<uint> udist_;
        std::mt19937_64 random_state_;
    };

} // namespace drforest
