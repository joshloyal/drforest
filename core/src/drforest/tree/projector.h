#pragma once

#include <utility>

namespace drforest {
    class DimensionReductionProjector {
    public:
        DimensionReductionProjector(
            const int num_slices, const bool use_original_features);

        ~DimensionReductionProjector() {};

        std::pair<arma::mat, arma::mat> get_directions(
                                            const arma::mat &X,
                                            const arma::vec &y,
                                            const WeightVec &sample_weight,
                                            const FeatureTypes &feat_types);

        std::pair<arma::mat, arma::mat> get_sir(
                const arma::mat &X,
                const arma::vec &y,
                const WeightVec &sample_weight);

        std::pair<arma::mat, arma::mat> get_save(
                const arma::mat &X,
                const arma::vec &y,
                const WeightVec &sample_weight);

    private:
        int num_slices_;
        bool use_original_features_;
    };
}
