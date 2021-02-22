#include <memory>
#include <stack>
#include <iostream>

#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"


std::pair<arma::mat, arma::vec> linear_model(int n_samples, uint seed=123) {
    arma::arma_rng::set_seed(seed);

    // generate gaussian features
    arma::mat X(n_samples, 10, arma::fill::randn);

    // generate linear signal
    arma::vec beta{3, 2, 1, 0, 0, 2, 0, 0, 0, 1};
    arma::vec y =  X * beta + arma::vec(n_samples, arma::fill::randn);

    return {X, y};
}


// smoke test the screener identifies features with signal
TEST_CASE("Test screener on a sparse linear signal", "[screener]") {

    auto [X, y] = linear_model(500, 3234);
    auto sample_weight = arma::vec(y.n_rows, arma::fill::ones);
    drforest::FeatureScreener screener(5, 2, 2);
    auto feature_ids = screener.screen_features(X, y, sample_weight);

    REQUIRE(feature_ids.n_rows == 5);

    arma::uvec sorted_ids = arma::sort(feature_ids);
    arma::uvec expected{0, 1, 2, 5, 9};
    drforest::testing::assert_vector_equal(sorted_ids, expected);
}
