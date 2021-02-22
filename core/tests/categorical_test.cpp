#include "catch.hpp"

#include "drforest.h"
#include "asserts.h"


TEST_CASE("Categorical Sampler", "[categorical]") {
    uint seed = 123;
    arma::vec weights{2, 1, 1};

    drforest::Categorical categorical(seed);

    uint index = categorical.draw(weights);
    REQUIRE(index == 0);
}
