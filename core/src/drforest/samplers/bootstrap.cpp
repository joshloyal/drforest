#include <vector>

#include "drforest.h"

namespace drforest {

    BootstrapSampler::BootstrapSampler(uint num_samples, uint seed) :
            num_samples_(num_samples),
            udist_(0, num_samples_ - 1) {
        random_state_.seed(seed);
    }

    WeightVec BootstrapSampler::draw() {
        WeightVec sample_weight(num_samples_, arma::fill::zeros);

        uint sample_index;
        for (int i = 0; i < num_samples_; ++i) {
            sample_index = udist_(random_state_);
            sample_weight[sample_index] += 1;
        }

        return sample_weight;
    }

    std::pair<arma::uvec, WeightVec> BootstrapSampler::draw_indices() {
        WeightVec sample_weight(num_samples_, arma::fill::zeros);
        arma::uvec sample_indices(num_samples_, arma::fill::zeros);

        uint sample_index;
        for (int i = 0; i < num_samples_; ++i) {
            sample_index = udist_(random_state_);
            sample_indices(i) = sample_index;
            sample_weight[sample_index] += 1;
        }

        return { arma::sort(sample_indices), sample_weight };
    }

}  // namespace drforest
