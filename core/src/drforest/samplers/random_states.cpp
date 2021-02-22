#include <vector>

#include "drforest.h"

namespace drforest {

    RandomStateSampler::RandomStateSampler(uint num_samples, uint seed) :
            num_samples_(num_samples),
            udist_(0) {
        random_state_.seed(seed);
    }

    std::vector<uint> RandomStateSampler::draw() {
        std::vector<uint> random_states;
        random_states.resize(num_samples_);
        for (int i = 0; i < num_samples_; ++i) {
            random_states.at(i) = udist_(random_state_);
        }

        return random_states;
    }

} // namespace drforest
