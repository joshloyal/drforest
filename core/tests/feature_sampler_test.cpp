#include "catch.hpp"

#include "drforest.h"
#include "asserts.h"


TEST_CASE("Simple feature sampler", "[sampler]") {
    size_t num_samples = 5;
    size_t num_total_features = 10;
    uint seed = 123;

    drforest::FeatureSampler sampler(num_total_features, seed);
    arma::uvec indices = sampler.draw(num_samples);

    REQUIRE(indices.n_elem == num_samples);

    for (int i = 0; i < indices.n_elem; ++i) {
        REQUIRE(indices(i) < num_total_features);
    }
}

TEST_CASE("Two draws have different features", "[sampler]") {
    size_t num_samples = 5;
    size_t num_total_features = 10;
    uint seed = 123;

    drforest::FeatureSampler sampler(num_total_features, seed);
    arma::uvec indices_one = sampler.draw(num_samples);
    arma::uvec indices_two = sampler.draw(num_samples);

    REQUIRE(indices_one.n_elem == num_samples);
    REQUIRE(indices_two.n_elem == num_samples);

    // this seeds works out so each element is different
    for (int i = 0; i < num_samples; ++i) {
        REQUIRE(indices_one(i) != indices_two(i));
    }
}


TEST_CASE("Request more features", "[sampler]") {
    size_t num_samples = 20;
    size_t num_total_features = 10;
    uint seed = 123;

    drforest::FeatureSampler sampler(num_total_features, seed);
    arma::uvec indices = sampler.draw(num_samples);

    REQUIRE(indices.n_elem == 10);

    arma::uvec expected = arma::linspace<arma::uvec>(
        0, num_total_features - 1, num_total_features);
    drforest::testing::assert_vector_equal(indices, expected);
}
