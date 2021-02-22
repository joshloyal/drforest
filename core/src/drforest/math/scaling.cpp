#include "drforest.h"

namespace drforest {
namespace math {

// Replaces zeros with ones to avoid division by zero.
void _handle_zeros_in_scale(arma::rowvec &scale, double impute_val = 1) {
    scale.for_each(
        [impute_val](arma::rowvec::elem_type &val) {
            if(val == 0.0) val = impute_val; }
    );
}


// Calculates the column-wise standard deviations of a matrix.
// Columns with zero standard deviation are imputed with `impute_val`.
arma::rowvec imputed_stddev(const arma::mat &X, double impute_val) {
    arma::rowvec scale = arma::stddev(X);
    _handle_zeros_in_scale(scale, impute_val);

    return scale;
}


arma::mat standardize(const arma::mat &X, bool with_std) {
    // subtract column means
    arma::mat out = X.each_row() - arma::mean(X);

    // scale by column stddev
    if (with_std) {
        out.each_row() /= imputed_stddev(X);
    }

    return out;
}

// Sample weights are assumed to add up to X.n_rows
arma::rowvec weighted_mean(const DataMat &X,
                           const WeightVec &sample_weight) {

    arma::mat X_weighted = X.each_col() % sample_weight;

    return arma::sum(X_weighted) / arma::sum(sample_weight);
}


DataMat center(const DataMat &X, const WeightVec &sample_weight) {
    // subtract column means
    return X.each_row() - weighted_mean(X, sample_weight);
}

std::pair<arma::mat, arma::mat> whiten(const DataMat &X) {

    // Center and Whiten feature matrix using a QR decomposition
    arma::mat Q, R, Z;
    arma::qr_econ(Q, R, standardize(X, false));
    Z = std::sqrt(X.n_rows) * Q;

    return {Z, R};
}

std::pair<arma::mat, arma::mat> whiten(const DataMat &X,
                                       const WeightVec &sample_weight) {

    // Center and Whiten feature matrix using a QR decomposition
    arma::mat Q, R, Z;

    arma::mat X_center = center(X, sample_weight);

    // QR = W^{1/2} * (X - mean(X))
    X_center.each_col() %= arma::sqrt(sample_weight);
    arma::qr_econ(Q, R, X_center);

    // Rescale back to unweighted space
    Q.each_col() /= arma::sqrt(sample_weight);

    // Whitened Matrix
    Z = std::sqrt(arma::sum(sample_weight)) * Q;

    return {Z, R};
}

}  // namespace math
}  // namespace drforest
