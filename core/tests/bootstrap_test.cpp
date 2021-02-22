#include "catch.hpp"

#include "drforest.h"
#include "datasets.h"
#include "asserts.h"


TEST_CASE("Boostrap Sampler", "[sampler]") {
    size_t num_samples = 500;
    uint seed = 123;

    drforest::BootstrapSampler bootstrap(num_samples, seed);
    arma::vec weights_one = bootstrap.draw();

    // weights should sum to the number of samples
    REQUIRE(arma::sum(weights_one) == num_samples);

    // subsequent calls should generate different samples
    arma::vec weights_two = bootstrap.draw();
    REQUIRE(arma::sum(weights_two) == num_samples);
    REQUIRE(arma::any(weights_two - weights_one));
}

TEST_CASE("Target Statistics", "[target-stats]") {
    // generate a toy dataset with only three values
    arma::vec y = drforest::math::sample(arma::vec{1, 2, 3}, 500);
    y = arma::sort(y);
    auto [unique_y, counts] = drforest::math::unique_counts(y);

    // draw a bootsrap
    uint seed = 42;
    drforest::BootstrapSampler bootstrap(y.n_elem, seed);
    arma::vec sample_weight = bootstrap.draw();

    // recalculate y stats
    auto [y_stats, indices] = drforest::determine_target_stats(
        y, sample_weight);

    // samples not drawn from the bootstrap should have negative indices
    arma::uvec negative_indices = arma::find(indices == -1);
    arma::uvec unsampled_indices = arma::find(sample_weight == 0);
    drforest::testing::assert_vector_equal(negative_indices,
                                           unsampled_indices);

    // check that counts line up
    arma::uvec borders = arma::cumsum(counts);
    uint expected_count = arma::sum(sample_weight.rows(0, borders(0) - 1));
    REQUIRE(expected_count == y_stats.counts(0));

    expected_count = arma::sum(sample_weight.rows(borders(0), borders(1) - 1));
    REQUIRE(expected_count == y_stats.counts(1));

    expected_count = arma::sum(sample_weight.rows(borders(1), borders(2) - 1));
    REQUIRE(expected_count == y_stats.counts(2));

    drforest::testing::assert_vector_equal(y_stats.unique_y, unique_y);
}

TEST_CASE("Target Statistics Actual Bootstrap", "[target-stats-boot]") {
    // A better test is to compare with the actual bootstrap
    auto [X, y] = drforest::testing::load_athletes();
    y = arma::sort(y);

    // actually materialize a bootstrap sample
    uint num_samples = y.n_elem;
    arma::uvec sample_indices = arma::linspace<arma::uvec>(0, num_samples - 1);
    arma::uvec bootstrap = drforest::math::sample(sample_indices, num_samples);
    arma::vec y_boot = arma::sort(y.rows(bootstrap));

    // expected target statistics
    auto [unique_y, counts] = drforest::math::unique_counts(y_boot);

    // calculate equivalent sample weights
    arma::vec sample_weight(num_samples, arma::fill::zeros);
    for (int i = 0; i < num_samples; ++i) {
        sample_weight(bootstrap(i)) += 1;
    }

    auto [y_stats, indices] = drforest::determine_target_stats(
        y, sample_weight);

    // check counts and unique_y are the same
    drforest::testing::assert_vector_equal(y_stats.unique_y, unique_y);
    drforest::testing::assert_vector_equal(y_stats.counts, counts);

    // check indices are correct
    for (int i = 0; i < indices.n_elem; ++i) {
        if (indices(i) == -1) {
            REQUIRE(sample_weight(i) == 0);
        } else {
            REQUIRE(y_stats.unique_y(indices(i)) == y(i));
        }
    }
}
