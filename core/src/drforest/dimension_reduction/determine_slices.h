#pragma once

namespace drforest {

struct SliceStats {
    arma::uvec slices;
    arma::uvec borders;
    arma::uvec counts;
};

// Determines slices used in SIR or SAVE.
//
// @returns slice_indicator, Indicates what slice an observation is in (n_samples,)
// @returns slice_counts, The number of samples in each slice (n_slices,)
SliceStats slice_y(const TargetVec &y, const uint num_slices=10);


// Determines slices used in SIR or SAVE. Assumes X, y are sorted wrt y.
// In addition, takes into account the sample_weights of X.
//
// @returns slice_indicator, Indicates what slice an observation is in (n_samples,)
// @returns slice_counts, The number of samples in each slice (n_slices,)
SliceStats slice_y_weighted(const TargetStats &y_stats,
                            const WeightVec &sample_weight,
                            const uint num_slices);

} // namespace drforest
