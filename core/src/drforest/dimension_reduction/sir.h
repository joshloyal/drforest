#pragma once

#include <utility>

namespace drforest {

arma::mat estimate_sir_M(const arma::mat &Z,
                         const arma::uvec &slices,
                         const arma::uvec &counts);

// Fits a Sliced Inverse Regression model
//
// @param X Training data, shape (n_samples, n_features)
// @param y Training labels, shape (n_samples,)
// @param num_slices Number of slices
arma::mat fit_sliced_inverse_regression(const DataMat &X,
                                        const TargetVec &y,
                                        const int num_slices=10);

// Fits a Weighted Sliced Inverse Regression model
//
// @param X Training data, shape (n_samples, n_features)
// @param y_stats Statistics of y values in a given node
// @param sample_weight Sample weights for each row of X.
// @param num_slices Number of slices
std::pair<arma::vec, arma::mat> fit_sliced_inverse_regression(
                                        const DataMat &X,
                                        const TargetStats &y_stats,
                                        const WeightVec &sample_weight,
                                        const int num_slices);

// Fits a Sliced Inverse Regression model with pre-precomputed slices
arma::mat fit_sliced_inverse_regression(const DataMat &X,
                                        const arma::uvec &slices,
                                        const arma::uvec &counts);

}  // namespace drforest
