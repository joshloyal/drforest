#pragma once

#include <utility>

namespace drforest {
namespace math {

// Standardize each column of a feature matrix, X.
//
// Centers each column by its mean and scales by its standard deviation.
//
// @param X Feature matrix, shape (n_samples, n_features)
arma::mat standardize(const arma::mat &X, bool with_std=true);

// Center each column of a feature matrix, X.
//
// Centers each column by subtracting of a weighted mean of X.
// The weights are assumed to add up to X.n_rows.
//
// @param X Feature matrix, shape (n_samples, n_features)
DataMat center(const DataMat &X, const WeightVec &sample_weight);

arma::rowvec weighted_mean(const DataMat &X,
                           const WeightVec &sample_weight);

// Calculates the column-wise standard deviations of a matrix.
// Columns with zero standard deviation are imputed with `impute_val`.
arma::rowvec imputed_stddev(const arma::mat &X, double impute_val=1);

// Whitens a matrix using a QR decomposition.
//
// Returns Z and R where Z = sqrt(num_samples) * Q is a whitened version of
// the data.
std::pair<arma::mat, arma::mat> whiten(const DataMat &X);

// Whitens a matrix using a QR decomposition.
//
// Returns Z and R where Z = sqrt(num_samples) * Q is a whitened version of
// the data. Sample weights correspond to bootstrap sample weights
std::pair<arma::mat, arma::mat> whiten(const DataMat &X,
                                       const WeightVec &sample_weight);

}  // namespace math
}  // namespace drforest
