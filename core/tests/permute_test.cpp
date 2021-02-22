#include "catch.hpp"

#include "drforest.h"
#include "datasets.h"
#include "asserts.h"


TEST_CASE("Permutation", "[permute]") {
    size_t num_samples = 10;
    uint seed = 123;

    drforest::Permuter permuter(num_samples, seed);
    arma::uvec permute = permuter.draw();
    std::cout << permute << std::endl;

    std::cout << permuter.draw() << std::endl;
}
