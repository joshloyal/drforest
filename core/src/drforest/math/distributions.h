#pragma once

namespace drforest {
namespace math {

// Generates samples from a bernoulli distribution with success probability
// `proba`.
template<typename VecType>
VecType rbernoulli(const int n_samples, const double proba=0.5,
                   const uint seed=123) {
    std::mt19937_64 random_state(seed);
    std::bernoulli_distribution distribution(proba);

    VecType samples(n_samples);
    for(auto i = 0; i < n_samples; ++i) {
        samples[i] = distribution(random_state);
    }

    return samples;
}

// Samples from a vector with replacement. Each value is equally likely.
template<typename VecType>
VecType sample(VecType values, const size_t num_samples, const uint seed=123) {
    arma::arma_rng::set_seed(seed);
    VecType out(num_samples);

    // generate random indices
    arma::ivec indices = arma::randi(num_samples,
                                     arma::distr_param(0, values.n_elem - 1));

    // fill out with samples
    for(int i = 0; i < num_samples; ++i) {
        out(i) = values(indices(i));
    }

    return out;
}

}  // namespace math
}  // namespace drforest
