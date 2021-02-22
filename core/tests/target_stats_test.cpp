#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

TEST_CASE("Unweighted Target Split", "[target-splitter]") {
    drforest::TargetVec y{1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::WeightVec sample_weight(9, arma::fill::ones);

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(2, 6);

    REQUIRE(y_stats.num_samples == 4);

    drforest::testing::assert_vector_equal(y_stats.unique_y,
                                           arma::vec{3, 4, 5, 6});

    drforest::testing::assert_vector_equal(y_stats.counts,
                                           arma::uvec{1, 1, 1, 1});
}

TEST_CASE("Unweighted Target Split Not all Unique", "[target-splitter]") {
    drforest::TargetVec y{1, 1, 1, 1, 2, 2, 2, 3, 3};
    drforest::WeightVec sample_weight(9, arma::fill::ones);

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(2, 8);

    REQUIRE(y_stats.num_samples == 6);

    drforest::testing::assert_vector_equal(y_stats.unique_y,
                                           arma::vec{1, 2, 3});

    drforest::testing::assert_vector_equal(y_stats.counts,
                                           arma::uvec{2, 3, 1});
}

TEST_CASE("Unweighted Target Split Beginning", "[target-splitter]") {
    drforest::TargetVec y{1, 1, 1, 1, 2, 2, 2, 3, 3};
    drforest::WeightVec sample_weight(9, arma::fill::ones);

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(0, 5);

    REQUIRE(y_stats.num_samples == 5);

    drforest::testing::assert_vector_equal(y_stats.unique_y,
                                           arma::vec{1, 2});

    drforest::testing::assert_vector_equal(y_stats.counts,
                                           arma::uvec{4, 1});
}


TEST_CASE("Unweighted Target Split Full Array", "[target-splitter]") {
    drforest::TargetVec y{1, 1, 1, 1, 2, 2, 2, 3, 3};
    drforest::WeightVec sample_weight(9, arma::fill::ones);

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(0, 9);

    REQUIRE(y_stats.num_samples == 9);

    drforest::testing::assert_vector_equal(y_stats.unique_y,
                                           arma::vec{1, 2, 3});

    drforest::testing::assert_vector_equal(y_stats.counts,
                                           arma::uvec{4, 3, 2});
}

TEST_CASE("Weighted Target Split Full Array", "[target-splitter]") {
    arma::vec y = arma::vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(0, 9);

    REQUIRE(y_stats.num_samples == 9);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{2, 3, 5, 6, 7, 8});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{2, 1, 1, 2, 2, 1});
}

TEST_CASE("Weighted Target Split", "[target-splitter]") {
    arma::vec y = arma::vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(2, 7);

    REQUIRE(y_stats.num_samples == 6);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{3, 5, 6, 7});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{1, 1, 2, 2});
}

TEST_CASE("Weighted Target Split Repeated Full Array", "[target-splitter]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(0, 9);

    REQUIRE(y_stats.num_samples == 9);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{1, 2, 3, 4});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{3, 1, 4, 1});
}

TEST_CASE("Weighted Target Split Repeated", "[target-splitter]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(2, 7);

    REQUIRE(y_stats.num_samples == 6);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{1, 2, 3});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{1, 1, 4});
}

TEST_CASE("Weighted Target Split Beginning", "[target-splitter]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(0, 4);

    REQUIRE(y_stats.num_samples == 3);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{1});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{3});
}

TEST_CASE("Weighted Target Split End", "[target-splitter]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(4, 9);

    REQUIRE(y_stats.num_samples == 6);

    drforest::testing::assert_vector_equal(
        y_stats.unique_y, arma::vec{2, 3, 4});

    drforest::testing::assert_vector_equal(
        y_stats.counts, arma::uvec{1, 4, 1});
}

TEST_CASE("Weighted Split Athletes", "[target-stats]") {
    auto [X, y] = drforest::testing::load_athletes();
    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::BootstrapSampler bootstrap(X.n_rows);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();

    size_t start = 20;
    size_t end = 110;
    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(start, end);

    REQUIRE(arma::sum(sample_weight.rows(start, end - 1)) == y_stats.num_samples);

    // determine unique y values in the sample
    arma::vec y_slice = y.rows(start, end - 1);
    arma::vec sample_weight_slice = sample_weight.rows(start, end - 1);
    auto [expected_y_stats, indices] =
        drforest::determine_target_stats(y_slice, sample_weight_slice);

    drforest::testing::assert_vector_equal(expected_y_stats.unique_y,
                                           y_stats.unique_y);
    drforest::testing::assert_vector_equal(expected_y_stats.counts,
                                           y_stats.counts);
}

TEST_CASE("Weighted Split Athletes Begin", "[target-stats]") {
    auto [X, y] = drforest::testing::load_athletes();
    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::BootstrapSampler bootstrap(X.n_rows);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();

    size_t start = 0;
    size_t end = 79;
    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(start, end);

    REQUIRE(arma::sum(sample_weight.rows(start, end - 1)) == y_stats.num_samples);

    // determine unique y values in the sample
    arma::vec y_slice = y.rows(start, end - 1);
    arma::vec sample_weight_slice = sample_weight.rows(start, end - 1);
    auto [expected_y_stats, indices] =
        drforest::determine_target_stats(y_slice, sample_weight_slice);

    drforest::testing::assert_vector_equal(expected_y_stats.unique_y,
                                           y_stats.unique_y);
    drforest::testing::assert_vector_equal(expected_y_stats.counts,
                                           y_stats.counts);
}

TEST_CASE("Weighted Split Athletes End", "[target-stats]") {
    auto [X, y] = drforest::testing::load_athletes();
    arma::uvec sorted_index = arma::stable_sort_index(y);
    X = X.rows(sorted_index);
    y = y.rows(sorted_index);

    drforest::BootstrapSampler bootstrap(X.n_rows);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();

    size_t start = 23;
    size_t end = y.n_rows;
    drforest::TargetSplitter splitter(y, sample_weight);
    drforest::TargetStats y_stats = splitter.split_y(start, end);

    REQUIRE(arma::sum(sample_weight.rows(start, end - 1)) == y_stats.num_samples);

    // determine unique y values in the sample
    arma::vec y_slice = y.rows(start, end - 1);
    arma::vec sample_weight_slice = sample_weight.rows(start, end - 1);
    auto [expected_y_stats, indices] =
        drforest::determine_target_stats(y_slice, sample_weight_slice);

    drforest::testing::assert_vector_equal(expected_y_stats.unique_y,
                                           y_stats.unique_y);
    drforest::testing::assert_vector_equal(expected_y_stats.counts,
                                           y_stats.counts);
}
