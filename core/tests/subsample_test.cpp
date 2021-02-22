#include "catch.hpp"

#include "drforest.h"
#include "datasets.h"
#include "asserts.h"


TEST_CASE("Subsampler Sampler", "[subsampler]") {
    size_t num_samples = 500;
    uint seed = 123;

    drforest::SubSampler subsampler(num_samples, 5, seed);
    arma::vec weights_one = subsampler.draw();

    // weights should sum to the number of samples
    REQUIRE(arma::sum(weights_one) == 5);

    // subsequent calls should generate different samples
    arma::vec weights_two = subsampler.draw();
    REQUIRE(arma::sum(weights_two) == 5);
    REQUIRE(arma::any(weights_two - weights_one));
}
