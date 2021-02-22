#include "drforest.h"

namespace drforest {

arma::mat construct_M_dr(const arma::mat &Z,
                         const arma::uvec &slices,
                         const arma::uvec &counts,
                         const WeightVec &sample_weight) {
    // XXX: Assumes sample weights sum to n_samples (i.e. bootstrap)
    //      If sample weights do not, then count number of non-zero weights!
    uint num_samples = arma::sum(sample_weight);
    uint num_features = Z.n_cols;

    arma::mat Mz = arma::zeros<arma::mat>(num_features, num_features);
    arma::mat Mzzt = arma::zeros<arma::mat>(num_features, num_features);
    for (int i = 0; i < counts.n_rows; ++i) {
        // extract and center entries in a slice
        arma::uvec slice_indices = arma::find((slices == i) &&
                                              (sample_weight != 0));
        arma::mat Z_slice = Z.rows(slice_indices);
        arma::vec slice_sample_weight = sample_weight.rows(slice_indices);

        int num_slice = arma::sum(slice_sample_weight);
        int p_slice = (double) num_samples / (double) num_slice;

        // extract empirical moments
        arma::vec Ez = arma::mean(Z_slice).t();
        arma::mat Ezzt = (Z_slice.t() * Z_slice) / num_slice;

        // first moment matrix
        Mz += p_slice * (Ez * Ez.t());

        // second moment matrix
        Mzzt += p_slice * (Ezzt * Ezzt);
    }

    arma::mat M = 2 * Mzzt + 2 * (Mz * Mz) + 2 * arma::trace(Mz) * Mz;
    M -= 2 * arma::eye(num_features, num_features);

    return M;
}

std::pair<arma::vec, arma::mat> fit_directional_regression(
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
    arma::mat M = construct_M_dr(Z, slices, counts, sample_weight);

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

} // namespace drforest
