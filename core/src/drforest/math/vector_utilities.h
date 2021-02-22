#pragma once

namespace drforest {
namespace math {

template<typename T>
arma::Col<T> rep_vec(uint n, T value) {
    arma::Col<T> new_vec(n);
    new_vec.fill(value);

    return new_vec;
}

// Determines the unique values and the number of times they occur in a
// one dimensional vector.
template <typename T>
std::pair<arma::Col<T>, arma::uvec> unique_counts(const arma::Col<T> &y) {
    arma::Col<T> unique = arma::unique(y);

    arma::uvec counts(unique.n_elem);
    for(int i = 0; i < unique.n_elem; ++i) {
        counts(i) = (uint)std::count(y.begin(), y.end(), unique(i));
    }

    return {unique, counts};
}

}  // namespace drforest
}  // namespace math
