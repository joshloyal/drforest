#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

TEST_CASE("Sliced Inverse Regression", "[sir]") {
    auto [X, y] = drforest::testing::load_athletes();
    arma::mat expected_directions =
        drforest::testing::load_athletes_directions();

    arma::mat directions =
        drforest::fit_sliced_inverse_regression(X, y, 11);

    // we can only compare magnitudes since some directions are flipped
    arma::mat abs_directions = arma::abs(directions);
    arma::mat abs_expected = arma::abs(expected_directions);
    drforest::testing::assert_matrix_allclose(abs_directions, abs_expected);
}

TEST_CASE("Weighted Sliced Inverse Regression", "[sir]") {
    // Test that the weighted version matches the unweighted version
    auto [X, y] = drforest::testing::load_athletes();

    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::WeightVec sample_weight(X.n_rows, arma::fill::ones);

    // calculate y stats
    auto [y_stats, indices] = drforest::determine_target_stats(
        y, sample_weight);

    arma::mat expected_directions =
        drforest::testing::load_athletes_directions();

    auto [eigvals, directions] =
        drforest::fit_sliced_inverse_regression(X, y_stats, sample_weight, 11);

    // make sure eigvales are in descending order
    REQUIRE(eigvals(0) > eigvals(1));

    // we can only compare magnitudes since some directions are flipped
    arma::mat abs_directions = arma::abs(directions);
    arma::mat abs_expected = arma::abs(expected_directions);
    drforest::testing::assert_matrix_allclose(abs_directions, abs_expected);
}

TEST_CASE("Weighted Sliced Inverse Regression Boostrap", "[sir]") {
    // Test that the weighted version matches the unweighted version
    // using an actual bootstrap sample
    auto [X, y] = drforest::testing::load_athletes();

    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::BootstrapSampler bootstrap(X.n_rows);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();

    // calculate y stats
    auto [y_stats, indices] = drforest::determine_target_stats(
        y, sample_weight);

    auto [eigvals, directions] =
        drforest::fit_sliced_inverse_regression(X, y_stats, sample_weight, 11);

    // make sure eigvales are in descending order
    REQUIRE(eigvals(0) > eigvals(1));

    // actually create the bootstrap
    X = X.rows(sample_indices);
    y = y.rows(sample_indices);
    arma::mat directions_bootstrap =
        drforest::fit_sliced_inverse_regression(X, y, 11);

    // the directions should be the same up to some sign changes
    arma::mat abs_directions = arma::abs(directions);
    arma::mat abs_expected = arma::abs(directions_bootstrap);
    drforest::testing::assert_matrix_allclose(abs_directions, abs_expected);
}

// Same as the previous except we subsample rows
TEST_CASE("Weighted Sliced Inverse Regression Boostrap Subsample",
          "[subsample sir]") {
    auto [X, y] = drforest::testing::load_athletes();

    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::BootstrapSampler bootstrap(X.n_rows);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();


    // sub-sample by splitting in half
    arma::mat X_sub = X.rows(0, 99);
    arma::vec y_sub = y.rows(0, 99);
    sample_weight = sample_weight.rows(0, 99);

    // remove rows with zero weight since we do this in the tree building code
    arma::uvec nonzero = arma::find(sample_weight != 0.0);
    X_sub = X_sub.rows(nonzero);
    y_sub = y_sub.rows(nonzero);
    sample_weight = sample_weight(nonzero);

    // calculate y stats
    auto [y_stats, indices] = drforest::determine_target_stats(
        y_sub, sample_weight);
    auto [eigvals, directions] =
        drforest::fit_sliced_inverse_regression(X_sub, y_stats,
                                                sample_weight, 10);

    // make sure eigvales are in descending order
    REQUIRE(eigvals(0) > eigvals(1));

    // actually create the bootstrap
    X = X.rows(sample_indices);
    y = y.rows(sample_indices);
    arma::uvec index = arma::find(sample_indices == 99); // first 100 samples
    X = X.rows(0, index(index.n_elem-1));
    y = y.rows(0, index(index.n_elem-1));
    arma::mat directions_bootstrap =
        drforest::fit_sliced_inverse_regression(X, y, 10);

    // the directions should be the same
    drforest::testing::assert_matrix_allclose(directions, directions_bootstrap);
}
