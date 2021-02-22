#include <math.h>

#include "drforest.h"

namespace drforest {

arma::mat construct_slice_means(const arma::mat &Z,
                                const arma::uvec &slices,
                                const arma::uvec &counts) {
    uint num_slices = counts.n_elem;
    uint num_features = Z.n_cols;

    arma::mat Z_means = arma::zeros<arma::mat>(num_slices, num_features);
    for (int i = 0; i < Z.n_rows; ++i) {
        Z_means.row(slices[i]) += Z.row(i);
    }

    // normalize by sqrt(slice_counts)
    arma::vec scale = arma::conv_to<arma::vec>::from(counts);
    Z_means.each_col() /= arma::sqrt(scale);

    return Z_means;
}


// dermine the M matrix from pre-whitened data
arma::mat estimate_sir_M(const arma::mat &Z,
                         const arma::uvec &slices,
                         const arma::uvec &counts) {
    // create slice matrix / covariance matrix
    arma::mat Z_means = construct_slice_means(Z, slices, counts);
    arma::mat M = (Z_means.t() * Z_means) / Z.n_rows;

    return M;
}

arma::mat fit_sliced_inverse_regression(const DataMat &X,
                                        const TargetVec &y,
                                        const int num_slices) {
    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X);

    // create slice matrix / covariance matrix
    auto [slices, borders, counts] = slice_y(y, num_slices);
    arma::mat M = estimate_sir_M(Z, slices, counts);

    // eigen decomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M);

    // transform eigenvalues back to feature space and
    // normalise to unit norm
    arma::mat directions = arma::solve(
        arma::trimatu(std::sqrt(X.n_rows) * R), arma::fliplr(eigvec));
    directions = arma::normalise(directions).t();

    return directions;
}

arma::mat construct_slice_means(const arma::mat &Z,
                                const arma::uvec &slices,
                                const arma::uvec &counts,
                                const WeightVec &sample_weight) {
    uint num_slices = counts.n_elem;
    uint num_features = Z.n_cols;

    arma::mat Z_means = arma::zeros<arma::mat>(num_slices, num_features);
    arma::vec weighted_counts(counts.n_elem, arma::fill::zeros);
    for (int i = 0; i < Z.n_rows; ++i) {
        if (sample_weight(i) != 0) {
            Z_means.row(slices[i]) += Z.row(i) * sample_weight(i);
            weighted_counts(slices[i]) += sample_weight(i);
        }
    }

    // normalize by sqrt(slice_counts)
    Z_means.each_col() /= arma::sqrt(weighted_counts);

    return Z_means;
}

std::pair<arma::vec, arma::mat> fit_sliced_inverse_regression(
                                        const DataMat &X,
                                        const TargetStats &y_stats,
                                        const WeightVec &sample_weight,
                                        const int num_slices) {
    double n_rows = sum(sample_weight);

    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X, sample_weight);

    // create slice matrix / covariance matrix
    auto [slices, borders, counts] = slice_y_weighted(y_stats, sample_weight,
                                                      num_slices);
    arma::mat Z_means = construct_slice_means(Z, slices, counts, sample_weight);
    arma::mat M = (Z_means.t() * Z_means) / n_rows;

    // eigen decomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M);

    // transform eigenvalues back to feature space and
    // normalise to unit norm
    arma::mat directions;
    if (X.n_rows >= X.n_cols) {
        directions = arma::solve(
            arma::trimatu(std::sqrt(n_rows) * R), arma::fliplr(eigvec));
    } else {
        directions = arma::solve(
            std::sqrt(n_rows) * R, arma::fliplr(eigvec));
    }
    directions = arma::normalise(directions).t();

    return {arma::reverse(eigval), directions};
}


arma::mat fit_sliced_inverse_regression(const DataMat &X,
                                        const arma::uvec &slices,
                                        const arma::uvec &counts) {
    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X);

    // create slice matrix / covariance matrix
    arma::mat M = estimate_sir_M(Z, slices, counts);

    // eigen decomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M);

    // transform eigenvalues back to feature space and
    // normalise to unit norm
    arma::mat directions;
    if (X.n_rows >= X.n_cols) {
        directions = arma::solve(
            arma::trimatu(std::sqrt(X.n_rows) * R), arma::fliplr(eigvec));
    } else {
        directions = arma::solve(
            std::sqrt(X.n_rows) * R, arma::fliplr(eigvec));
    }
    directions = arma::normalise(directions).t();

    return directions;
}

}  // namespace drforest
