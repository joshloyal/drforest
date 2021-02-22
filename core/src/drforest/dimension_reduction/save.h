#pragma once

#include <utility>

namespace drforest {

// Fits a Sliced Average Varience Estimation model
//
// @param X Training data, shape (n_samples, n_features)
// @param y Training labels, shape (n_samples,)
// @param num_slices Number of slices
arma::mat fit_sliced_average_variance_estimation(const arma::mat &X,
                                                 const arma::vec &y,
                                                 const int num_slices=10);

// Fits a Weighted Sliced Average Varience Estimation model
//
// @param X Training data, shape (n_samples, n_features)
// @param y_stats Statistics of y values in a given node
// @param sample_weight Sample weights for each row of X.
// @param num_slices Number of slices
std::pair<arma::vec, arma::mat> fit_sliced_average_variance_estimation(
                                                 const DataMat &X,
                                                 const TargetStats &y_stats,
                                                 const WeightVec &sample_weight,
                                                 const int num_slices);

// Fits a Sliced Average Variance Estimation model with pre-computed slices
arma::mat fit_sliced_average_variance_estimation(const DataMat &X,
                                                 const arma::uvec &slices,
                                                 const arma::uvec &counts);

}  // namespace drforest
