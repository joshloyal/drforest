#include <utility>

#include "drforest.h"

namespace drforest {

    DimensionReductionProjector::DimensionReductionProjector(
        const int num_slices) : num_slices_(num_slices) {}

    std::pair<arma::mat, arma::mat> DimensionReductionProjector::get_directions(
                                            const arma::mat &X,
                                            const arma::vec &y,
                                            const WeightVec &sample_weight) {

        // determine DR directions.
        arma::mat Z;
        arma::mat directions;
        if (X.n_rows < X.n_cols) {
            // return raw features and perform a cart split on all of them
            Z = X;
            directions = arma::eye(X.n_cols, X.n_cols);
        } else {
            // include both and let CART decide
            auto [Z_save, dir_save] = get_save(X, y, sample_weight);
            auto [Z_sir, dir_sir] = get_sir(X, y, sample_weight);

            Z = arma::join_rows(Z_save, Z_sir);
            directions = arma::join_cols(dir_save, dir_sir);
        }

        return {Z, directions};
    }

    std::pair<arma::mat, arma::mat> DimensionReductionProjector::get_sir(
            const arma::mat &X,
            const arma::vec &y,
            const WeightVec &sample_weight) {
        // re-calculate target statistics in this node
        auto [y_stats, indices] =
            determine_target_stats(y, sample_weight);

        auto [eigvals, directions] =
            fit_sliced_inverse_regression(X,
                                          y_stats,
                                          sample_weight,
                                          num_slices_);

        return {X * directions.t(), directions};
    }

    std::pair<arma::mat, arma::mat> DimensionReductionProjector::get_save(
            const arma::mat &X,
            const arma::vec &y,
            const WeightVec &sample_weight) {
        // re-calculate target statistics in this node
        auto [y_stats, indices] =
            determine_target_stats(y, sample_weight);

        auto [eigvals, directions] =
            fit_sliced_average_variance_estimation(X,
                                                   y_stats,
                                                   sample_weight,
                                                   num_slices_);

        return {X * directions.t(), directions};
    }

}
