#include "drforest.h"

namespace drforest {

arma::mat construct_slice_covariance(const arma::mat &Z,
                                     const arma::uvec &slices,
                                     const arma::uvec &counts) {
    // construct the slice covariance matrix
    uint num_samples = Z.n_rows;
    uint num_features = Z.n_cols;
    arma::mat M = arma::zeros<arma::mat>(num_features, num_features);
    for (int i = 0; i < counts.n_rows; ++i) {
        int num_slice = counts(i);

        // extract and center entries in a slice
        arma::uvec slice_indices = arma::find(slices == i);
        arma::mat Z_slice = Z.rows(slice_indices);
        Z_slice.each_row() -= arma::mean(Z_slice);

        // slice covariance matrix
        arma::mat V_slice = (Z_slice.t() * Z_slice) / num_slice;
        arma::mat M_slice = arma::eye<arma::mat>(arma::size(V_slice)) -
            V_slice;

        double slice_ratio = (double) num_slice / (double) num_samples;
        M += slice_ratio * (M_slice * M_slice);
    }

    return M;
}

arma::mat fit_sliced_average_variance_estimation(const DataMat &X,
                                                 const TargetVec &y,
                                                 const int num_slices) {
    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X);

    // create slice covariance matrix
    auto [slices, borders, counts] = slice_y(y, num_slices);
    arma::mat M = construct_slice_covariance(Z, slices, counts);

    // eigen decomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M);

    // transform eigenvalues back to feature space and
    // normalise to unit norm
    arma::mat directions = arma::solve(
        arma::trimatu(std::sqrt(X.n_rows) * R), arma::fliplr(eigvec));
    directions = normalise(directions).t();

    return directions;
}

arma::mat construct_slice_covariance(const arma::mat &Z,
                                     const arma::uvec &slices,
                                     const arma::uvec &counts,
                                     const WeightVec &sample_weight) {
    // construct the slice covariance matrix
    // XXX: Assumes sample weights sum to n_samples (i.e. bootstrap indicator)
    //      if sample weights do not, then we need to count number of non-zeros.
    //uint num_samples = Z.n_rows;
    uint num_samples = arma::sum(sample_weight);
    uint num_features = Z.n_cols;
    arma::mat M = arma::zeros<arma::mat>(num_features, num_features);
    for (int i = 0; i < counts.n_rows; ++i) {
        // extract entries in a slice
        arma::uvec slice_indices = arma::find(
            (slices == i) && (sample_weight != 0));
        arma::mat Z_slice = Z.rows(slice_indices);
        arma::vec slice_sample_weight = sample_weight.rows(slice_indices);

        // normalizing factor (sum of the weights)
        int num_slice = arma::sum(slice_sample_weight);

        // slice covariance matrix
        Z_slice = drforest::math::center(Z_slice, slice_sample_weight);
        Z_slice.each_col() %= arma::sqrt(slice_sample_weight);
        arma::mat V_slice = (Z_slice.t() * Z_slice) / num_slice;
        arma::mat M_slice = arma::eye<arma::mat>(arma::size(V_slice)) -
            V_slice;

        double slice_ratio = (double) num_slice / (double) num_samples;
        M += slice_ratio * (M_slice * M_slice);
    }

    return M;
}

std::pair<arma::vec, arma::mat> fit_sliced_average_variance_estimation(
                                                 const DataMat &X,
                                                 const TargetStats &y_stats,
                                                 const WeightVec &sample_weight,
                                                 const int num_slices) {
    double n_rows = sum(sample_weight);

    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X, sample_weight);

    // create slice covariance matrix
    auto [slices, borders, counts] = slice_y_weighted(y_stats, sample_weight,
                                                      num_slices);
    arma::mat M = construct_slice_covariance(Z, slices, counts, sample_weight);

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
    directions = normalise(directions).t();

    return {arma::reverse(eigval), directions};
}

arma::mat fit_sliced_average_variance_estimation(const DataMat &X,
                                                 const arma::uvec &slices,
                                                 const arma::uvec &counts) {
    double n_rows = X.n_rows;

    // Center and Whiten feature matrix using a QR decomposition
    auto [Z, R] = drforest::math::whiten(X);

    // create slice covariance matrix
    arma::mat M = construct_slice_covariance(Z, slices, counts);

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
    directions = normalise(directions).t();

    return directions;
}

} // namespace drforest
