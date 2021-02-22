#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

TEST_CASE("Only a single slice", "[slice]") {
    // test only one value creates a single slice
    arma::vec y = arma::ones<arma::vec>(100);
    auto [slice_indicator, borders, slice_counts] = drforest::slice_y(y);

    // expect everything to be in the same slice
    arma::uvec expected_indicator = arma::zeros<arma::uvec>(100);
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);

    arma::uvec expected_counts{100};
    drforest::testing::assert_vector_equal(slice_counts, expected_counts);
}

TEST_CASE("Simple target with 100 unique values / even slices.", "[slice]" ) {
    arma::vec y = arma::shuffle( arma::linspace(0, 99, 100) );

    auto [slice_indicator, borders, counts] = drforest::slice_y(y);

    // all slices should have 10 counts
    arma::uvec expected_counts = drforest::math::rep_vec(
        10, (arma::uword)10);
    drforest::testing::assert_vector_equal(counts, expected_counts);

    // continguous indices counting up from 10
    arma::uvec y_order = arma::stable_sort_index(y);
    arma::uvec expected_indicator = arma::zeros<arma::uvec>(100);
    expected_indicator.elem(y_order.subvec(10, 19)).fill(1);
    expected_indicator.elem(y_order.subvec(20, 29)).fill(2);
    expected_indicator.elem(y_order.subvec(30, 39)).fill(3);
    expected_indicator.elem(y_order.subvec(40, 49)).fill(4);
    expected_indicator.elem(y_order.subvec(50, 59)).fill(5);
    expected_indicator.elem(y_order.subvec(60, 69)).fill(6);
    expected_indicator.elem(y_order.subvec(70, 79)).fill(7);
    expected_indicator.elem(y_order.subvec(80, 89)).fill(8);
    expected_indicator.elem(y_order.subvec(90, 99)).fill(9);
    drforest::testing::assert_vector_equal(slice_indicator,
                                           expected_indicator);
}

TEST_CASE( "Target with three values evenly distributed", "[slice]" ) {
    arma::vec y = arma::ones<arma::vec>(30);
    y.subvec(10, 19).fill(2.0);
    y.subvec(20, 29).fill(3.0);

    auto [slice_indicator, borders, counts] = drforest::slice_y(y, 3);

    arma::uvec expected_counts{10, 10, 10};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_indicator = arma::zeros<arma::uvec>(30);
    expected_indicator.subvec(10, 19).fill(1);
    expected_indicator.subvec(20, 29).fill(2);
    drforest::testing::assert_vector_equal(slice_indicator,
                                           expected_indicator);
}

TEST_CASE("Target with three values unevenly distributed", "[slice]") {
    arma::vec y{1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 1.0};
    auto [slice_indicator, borders, counts] = drforest::slice_y(y, 3);

    arma::uvec expected_counts{4, 1, 2};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_indicator{0, 1, 2, 0, 0, 2, 0};
    drforest::testing::assert_vector_equal(slice_indicator,
                                           expected_indicator);
}

TEST_CASE("Target with uneven slices within y", "[slice]") {
    arma::vec y = arma::ones<arma::vec>(43);
    y.subvec(13, 17).fill(2);
    y.subvec(18, 42).fill(3);

    auto [slice_indicator, borders, counts] = drforest::slice_y(y, 3);

    arma::uvec expected_counts{13, 5, 25};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_indicator = arma::zeros<arma::uvec>(43);
    expected_indicator.subvec(13, 17).fill(1);
    expected_indicator.subvec(18, 42).fill(2);
    drforest::testing::assert_vector_equal(slice_indicator,
                                           expected_indicator);
}

TEST_CASE("Slices match on the athletes", "[slice]") {
    auto [X, y] = drforest::testing::load_athletes();

    auto [slice_indicator, borders, counts] = drforest::slice_y(y, 11);

    // counts per slice match the DR package
    arma::uvec expected_counts{
        18, 18, 18, 18, 18, 19, 18, 19, 23, 18, 15};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    // slice indicators match the DR package
    arma::uvec expected = arma::uvec({
        6, 5, 4, 4, 2, 3, 5, 1, 3, 3, 8, 6, 1, 7, 5, 4, 3, 4, 6,
        4, 6, 6, 4, 3, 7, 6, 5, 4, 1, 2, 2, 5, 5, 6, 6, 2, 1, 3,
        2, 4, 4, 2, 3, 4, 2, 5, 4, 5, 3, 1, 3, 4, 3, 5, 5, 3, 3,
        3, 4, 2, 5, 5, 6, 5, 4, 2, 2, 2, 5, 8, 6, 6, 7, 5, 8, 1,
        3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 2, 2, 4, 4, 1, 2, 1, 2,
        1, 1, 1, 1, 1, 5, 8, 9, 10, 9, 8, 8, 9, 9, 10, 7, 9, 11, 9,
        9, 9, 11, 9, 10, 10, 1, 10, 10, 10, 10, 11, 10, 9, 11, 8, 10, 10, 11,
        11, 11, 9, 10, 8, 9, 8, 4, 7, 7, 8, 11, 10, 9, 6, 7, 8, 6, 5,
        4, 8, 3, 7, 7, 9, 9, 11, 9, 9, 11, 7, 9, 8, 7, 6, 6, 6, 7,
        7, 6, 6, 5, 11, 11, 11, 10, 7, 8, 9, 7, 9, 8, 8, 9, 7, 11, 9,
        9, 11, 8, 10, 10, 10, 7, 10, 8, 7, 6, 8,
    }) - 1;
    drforest::testing::assert_vector_equal(slice_indicator, expected);
}

TEST_CASE("Weighted Slices match on the athletes", "[slice]") {
    auto [X, y] = drforest::testing::load_athletes();
    drforest::WeightVec sample_weight(y.n_elem, arma::fill::ones);
    y = arma::sort(y);

    auto [y_stats, indices] = drforest::determine_target_stats(
        y, sample_weight);

    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 11);

    // counts per slice match the DR package
    arma::uvec expected_counts{
        18, 18, 18, 18, 18, 19, 18, 19, 23, 18, 15};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    // slice indicators match the DR package
    arma::uvec expected = arma::uvec({
        6, 5, 4, 4, 2, 3, 5, 1, 3, 3, 8, 6, 1, 7, 5, 4, 3, 4, 6,
        4, 6, 6, 4, 3, 7, 6, 5, 4, 1, 2, 2, 5, 5, 6, 6, 2, 1, 3,
        2, 4, 4, 2, 3, 4, 2, 5, 4, 5, 3, 1, 3, 4, 3, 5, 5, 3, 3,
        3, 4, 2, 5, 5, 6, 5, 4, 2, 2, 2, 5, 8, 6, 6, 7, 5, 8, 1,
        3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 2, 2, 4, 4, 1, 2, 1, 2,
        1, 1, 1, 1, 1, 5, 8, 9, 10, 9, 8, 8, 9, 9, 10, 7, 9, 11, 9,
        9, 9, 11, 9, 10, 10, 1, 10, 10, 10, 10, 11, 10, 9, 11, 8, 10, 10, 11,
        11, 11, 9, 10, 8, 9, 8, 4, 7, 7, 8, 11, 10, 9, 6, 7, 8, 6, 5,
        4, 8, 3, 7, 7, 9, 9, 11, 9, 9, 11, 7, 9, 8, 7, 6, 6, 6, 7,
        7, 6, 6, 5, 11, 11, 11, 10, 7, 8, 9, 7, 9, 8, 8, 9, 7, 11, 9,
        9, 11, 8, 10, 10, 10, 7, 10, 8, 7, 6, 8,
    }) - 1;
    expected = arma::sort(expected);
    drforest::testing::assert_vector_equal(slice_indicator, expected);
}

TEST_CASE("Boostrap Homogen. Slices - Weights all One", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    drforest::WeightVec sample_weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 2, 2, 2, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 2, 3, 4, 5};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{0, 0, 0, 1, 1, 2, 2, 3, 3, 4};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{3, 2, 2, 3};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 3, 5, 7, 10};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 1, 1, 2, 2, 3, 3, 3};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Homoegen. Slices - Weighted Zero at End", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    drforest::WeightVec sample_weight{2, 1, 0, 0, 1, 3, 2, 0, 1, 0};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 1, 5, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 2, 3, 4};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{0, 0, -1, -1, 1, 2, 2, -1, 3, -1};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{4, 6};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 4, 10};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Homogen. Slices - Weighted Zero at Start", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 0, 3, 2, 0, 0, 2};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 5, 2};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 3, 5};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 0, -1, -1, 1, 1, -1, -1, 2};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{3, 5, 2};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 3, 8, 10};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Homogen. Slices - Only Missing Beginning", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 0, 1, 1};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 1, 4, 1, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 2, 3, 4, 5};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 0, -1, 1, 2, 2, -1, 3, 4};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{4, 4, 2};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 4, 8, 10};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 0, 0, 1, 1, 2, 2, 2};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Homogen. Slices - Odd Samples Zero First", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 0, 1};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 1, 4, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 2, 3, 5};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 0, -1, 1, 2, 2, -1, 3};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{4, 5};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 4, 9};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 0, 0, 1, 1, 1, 1};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Homogen. Slices - Odd Samples Zero End", "[slice]") {
    arma::vec y = arma::vec {1, 1, 1, 2, 2, 3, 3, 4, 5};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{3, 1, 4, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{1, 2, 3, 4};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 0, -1, 1, 2, 2, 3, -1};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    // make "homogeneous" slices
    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 10);

    expected_counts = arma::uvec{4, 5};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 4, 9};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 0, 0, 1, 1, 1, 1};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Hetero. Slices - Odd Samples Two Slices", "[slice]") {
    arma::vec y = arma::vec {1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{2, 1, 1, 2, 2, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{2, 3, 5, 6, 7, 8};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 1, -1, 2, 3, 4, 5, -1};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 2);

    expected_counts = arma::uvec{4, 5};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 4, 9};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 0, 0, 1, 1, 1, 1};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Boostrap Hetero. Slices - Odd Samples Three Slices", "[slice]") {
    arma::vec y = arma::vec {1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::WeightVec sample_weight{0, 2, 1, 0, 1, 2, 2, 1, 0};

    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    // sanity check on weighted target statistics
    arma::uvec expected_counts{2, 1, 1, 2, 2, 1};
    drforest::testing::assert_vector_equal(y_stats.counts, expected_counts);

    arma::vec expected_unique{2, 3, 5, 6, 7, 8};
    drforest::testing::assert_vector_equal(y_stats.unique_y, expected_unique);

    arma::ivec expected_indices{-1, 0, 1, -1, 2, 3, 4, 5, -1};
    drforest::testing::assert_vector_equal(indices, expected_indices);

    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 3);

    expected_counts = arma::uvec{3, 3, 3};
    drforest::testing::assert_vector_equal(counts, expected_counts);

    arma::uvec expected_borders{0, 3, 6, 9};
    drforest::testing::assert_vector_equal(borders, expected_borders);

    arma::uvec expected_indicator{0, 0, 0, 1, 1, 1, 2, 2, 2};
    drforest::testing::assert_vector_equal(slice_indicator, expected_indicator);
}

TEST_CASE("Weighted Slices Match on Athletes - Actual Bootstrap", "[slice]") {
    auto [X, y] = drforest::testing::load_athletes();
    y = arma::sort(y);

    uint seed = 42;
    drforest::BootstrapSampler bootstrap(y.n_elem, seed);
    auto [sample_indices, sample_weight] = bootstrap.draw_indices();

    // apply the weighted version of sliceing
    auto [y_stats, indices] =
        drforest::determine_target_stats(y, sample_weight);

    auto [slice_indicator, borders, counts] =
        drforest::slice_y_weighted(y_stats, sample_weight, 11);

    // actually materialize the bootstrap sample and apply the slicing
    y = y.rows(sample_indices);
    auto [slice_indicator_boot, borders_boot, counts_boot] =
        drforest::slice_y(y, 11);

    // counts in each slice should be equal
    drforest::testing::assert_vector_equal(counts, counts_boot);
}
