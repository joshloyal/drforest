#pragma once

namespace drforest {

    // Determines non-zero samples and the associated weighted counts
    void init_sample_index(const WeightVec &sample_weight, arma::uvec &samples,
                           double &weighted_n_samples);

} // namespace drforest
