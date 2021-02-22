#include <vector>

#include "drforest.h"

namespace drforest {

    FeatureSampler::FeatureSampler(uint num_total_features, uint seed) :
        num_total_features_(num_total_features),
        udist_(0, num_total_features_ - 1) {
        random_state_.seed(seed);
    }

    arma::uvec FeatureSampler::draw(uint num_features) {
        // if we request more than the total number of features
        // just return all of the features.
        if (num_features > num_total_features_) {
            return arma::linspace<arma::uvec>(
                0, num_total_features_ - 1, num_total_features_);
        }

        arma::uvec feature_indices(num_features);

        // bitset to track which samples we have drawn
        std::vector<bool> bit_set(num_total_features_);
        int num_selected_features = 0;
        while (num_selected_features < num_features) {
            int feature_index = udist_(random_state_);
            if(!bit_set.at(feature_index)) {
                feature_indices(num_selected_features) = feature_index;

                // set bits
                bit_set.at(feature_index) = true;
                num_selected_features++;
            }
        }

        return feature_indices;
    }

} // namespace drforest
