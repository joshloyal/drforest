#pragma once

#include <utility>

namespace drforest {
    class FeatureScreener {
    public:
        FeatureScreener(const int max_features=5,
                        const int min_samples_leaf=2,
                        const int min_weight_leaf=2);

        ~FeatureScreener() {};

        arma::uvec screen_features(const arma::mat &X,
                                   const arma::vec &y,
                                   const WeightVec &sample_weight,
                                   double impurity=-1);

    private:
        int max_features_;
        int min_samples_leaf_;
        int min_weight_leaf_;
    };
} // namespace drforest
