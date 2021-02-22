#pragma once

#include <utility>

namespace drforest {

    // Statistics about the target in the current bootstrap sample
    struct TargetStats {
        uint num_samples;    // number of samples in y
        arma::vec unique_y;  // unique values of y
        arma::uvec counts;   // number of times each y appears in the sample
    };

    // Determines the counts and distribution of y values based on
    // the current bootstrap weights.
    std::pair<TargetStats, arma::ivec>
    determine_target_stats(const arma::vec &y,
                           const arma::vec &sample_weights);

    // Responsible for recalculating target statistics in a node determined
    // by X[start:end-1].
    class TargetSplitter {
    public:
        TargetSplitter(const TargetVec &y, const WeightVec &sample_weight);
        TargetStats split_y(size_t start, size_t end);

    private:
        const TargetVec &y_;
        const WeightVec &sample_weight_;
        arma::ivec indices_;  // sample_index -> unique_y_index map.
                              // unique_y[sample_index] = y.
        TargetStats y_stats_;
    };

}  // namespace drforest
