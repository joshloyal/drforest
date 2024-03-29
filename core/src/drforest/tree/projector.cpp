#include <utility>

#include "drforest.h"

namespace drforest {

    DimensionReductionProjector::DimensionReductionProjector(
        const int num_slices, const bool use_original_features) :
            num_slices_(num_slices),
            use_original_features_(use_original_features) {}

    std::pair<arma::mat, arma::mat> DimensionReductionProjector::get_directions(
                                            const arma::mat &X,
                                            const arma::vec &y,
                                            const WeightVec &sample_weight,
                                            const FeatureTypes &feat_types) {

        // determine DR directions.
        arma::mat Z;
        arma::mat directions;
        if (X.n_rows < X.n_cols || use_original_features_ ||
                feat_types.numeric_features.n_elem == 0) {
            // return raw features and perform a cart split on all of them
            Z = X;
            directions = arma::eye(X.n_cols, X.n_cols);
        } else if (feat_types.categorical_features.n_elem > 0) {
            // perform SDR estimation using only numeric features
            auto [Z_save, dir_save_num] = get_save(
                X.cols(feat_types.numeric_features), y, sample_weight);
            auto [Z_sir, dir_sir_num] = get_sir(
                X.cols(feat_types.numeric_features), y, sample_weight);

            // SAVE: pad directions with zeros in space of categorical features
            arma::rowvec dir_save(X.n_cols, arma::fill::zeros);
            dir_save.elem(feat_types.numeric_features) = dir_save_num.row(0);

            // SIR: pad directions with zeros in space of categorical features
            arma::rowvec dir_sir(X.n_cols, arma::fill::zeros);
            dir_sir.elem(feat_types.numeric_features) = dir_sir_num.row(0);

            // SIR and SAVE features
            Z = arma::join_rows(Z_save.col(0), Z_sir.col(0));
            directions = arma::join_cols(dir_save, dir_sir);

            // add categorical features back
            Z = arma::join_rows(Z, X.cols(feat_types.categorical_features));

            // add directions for categorical features
            for (int i = 0; i < feat_types.categorical_features.n_elem; i++) {
                arma::rowvec dir_cat(X.n_cols, arma::fill::zeros);
                dir_cat(feat_types.categorical_features(i)) = 1.;
                directions = arma::join_cols(directions, dir_cat);
            }
        } else {
            // include both and let CART decide
            auto [Z_save, dir_save] = get_save(X, y, sample_weight);
            auto [Z_sir, dir_sir] = get_sir(X, y, sample_weight);

            Z = arma::join_rows(Z_save.col(0), Z_sir.col(0));
            directions = arma::join_cols(dir_save.row(0), dir_sir.row(0));
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
