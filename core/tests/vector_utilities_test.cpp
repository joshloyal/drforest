#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"


TEST_CASE("Count Unique Values", "[unique]") {
    arma::vec a{3.2, 2, 2, 4, 4, 6, 6, 2, 3.2, 4.29};
    auto [unique, counts] = drforest::math::unique_counts(a);

    arma::vec expected{2, 3.2, 4, 4.29, 6};
    drforest::testing::assert_vector_equal(unique, expected);

    arma::uvec expected_counts{3, 2, 2, 1, 2};
    drforest::testing::assert_vector_equal(counts, expected_counts);
}
