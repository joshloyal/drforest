#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

/**
 * Makes sure that the samples array is initialized properly.
 */
TEST_CASE("DimensionReductionSplitter Init", "[splitter]") {
    arma::mat X = arma::mat(10, 3, arma::fill::randn);
    arma::vec y = arma::linspace<arma::vec>(0, 10, 10);
    arma::vec sample_weight{0, 1, 1, 0, 1, 0, 0, 1, 1, 0};

    drforest::DimensionReductionSplitter splitter(X, y, sample_weight);

    arma::uvec expected_samples{1, 2, 4, 7, 8};
    const arma::uvec samples = splitter.get_node_samples();
    drforest::testing::assert_vector_equal(samples, expected_samples);
}

/**
 * Check that the splitting rule is correct on the following toy dataset:
 *
 *  ------------------
 *  | X | y | weight |
 *  ------------------
 *  | 2 | 3 |   1    |
 *  | 0 | 3 |   1    |
 *  | 1 | 4 |   1    |  <-- best split (convention is <= in left node)
 *  | 4 | 6 |   1    |
 *  | 3 | 7 |   1    |
 *  | 5 | 5 |   1    |
 *  ------------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  1  |   134.0     |
 *  |  2  |   139.0     |
 *  |  3  |   141.33    |  <--- best improvement
 *  |  4  |   136.0     |
 *  |  5  |   130.8     |
 *  ---------------------
 */
TEST_CASE("split_node unweighted toy dataset", "[splitter]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec X{2, 0, 1, 4, 3, 5};
    arma::vec sample_weight(6, arma::fill::ones);

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight);
    splitter.reset_node(0, splitter.get_num_samples());

    // this should be the variance of the whole sample
    double impurity = splitter.node_impurity();
    drforest::testing::almost_equal(impurity, 2.222, 1e-3);

    // split the node. Note that the SDR direction is beta = 1
    drforest::SplitRecord split = splitter.split_node(impurity);

    // For one feature beta = 1.0
    drforest::testing::assert_vector_equal(split.direction, arma::vec{1.0});

    // make sure we pick the correct split point
    REQUIRE(split.threshold == 2.5);
    REQUIRE(split.pos == 3);
    REQUIRE(!split.is_leaf);

    // This is the variance of the samples in the left node (3, 3, 4)
    drforest::testing::almost_equal(split.impurity_left, 0.222, 1e-3);

    // This is the variance of the samples in the right node (6, 7, 5)
    drforest::testing::almost_equal(split.impurity_right, 0.666, 1e-3);

    // Since there are equal number of samples in each node the improvement
    // is impurity - 0.5 * left - 0.5 * right
    drforest::testing::almost_equal(split.improvement, 1.777, 1e-3);

    // finally make sure samples is rearranged properly
    arma::uvec expected{0, 1, 2, 3, 4, 5};
    drforest::testing::assert_vector_equal(splitter.get_node_samples(),
                                            expected);
}


/**
 * Check that the splitting rule is correct on the following toy dataset
 *
 *  ------------------
 *  | X | y | weight |
 *  ------------------
 *  | 2 | 3 |   2    |
 *  | 0 | 3 |   0    |
 *  | 1 | 4 |   3    |
 *  | 4 | 6 |   1    |
 *  | 3 | 7 |   2    |
 *  | 5 | 5 |   0    |
 *  ------------------
 *
 *  The reduced dataset is

 *  ------------------
 *  | X | y | weight |
 *  ------------------
 *  | 2 | 3 |   2    |
 *  | 1 | 4 |   3    |  <--- the split should occur here (X <= 2)
 *  | 4 | 7 |   1    |
 *  | 3 | 5 |   2    |
 *  ------------------
 *
 *  The criterion should be

 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  1  |   178.166   |
 *  |  2  |   185.133   |  <--- best improvement
 *  |  3  |   171.285   |
 *  ---------------------
 */
TEST_CASE("split_node with weights", "[splitter]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec X{2, 0, 1, 4, 3, 5};
    arma::vec sample_weight{2, 0, 3, 1, 2, 0};

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight);
    splitter.reset_node(0, splitter.get_num_samples());

    // this should be the weighted variance of the whole sample
    double impurity = splitter.node_impurity();
    drforest::testing::almost_equal(impurity, 2.4375);

    // split the node. Note that the SDR direction is beta = 1
    drforest::SplitRecord split = splitter.split_node(impurity);

    // For one feature beta = 1.0
    drforest::testing::assert_vector_equal(split.direction, arma::vec{1.0});

    // make sure we pick the correct split point
    REQUIRE(split.threshold == 2.5);
    REQUIRE(split.pos == 2);
    REQUIRE(!split.is_leaf);

    // This is the variance of the samples in the left node (3, 4)
    drforest::testing::almost_equal(split.impurity_left, 0.24);

    // This is the variance of the samples in the right node (7, 5)
    drforest::testing::almost_equal(split.impurity_right, 0.222, 1e-3);

    // The impurity improvement is
    // impurity - (5/8) * left - (3/8) * right
    drforest::testing::almost_equal(split.improvement, 2.204, 1e-3);

    // finally make sure samples is rearranged properly
    arma::uvec expected{0, 2, 3, 4};
    drforest::testing::assert_vector_equal(splitter.get_node_samples(),
                                            expected);
}

/**
 * Check that the splitting rule is correct on the following toy dataset:
 *
 *  ----------------------------
 *  |  id |  X  |  y  | weight |
 *  ----------------------------
 *  |  0  |  0  |  3  |    1   |
 *  |  1  |  2  |  2  |    1   |
 *  |  2  |  2  |  3  |    1   |  <--- start here
 *  |  3  |  0  |  3  |    1   |
 *  |  4  |  1  |  4  |    1   |  <--- best cut point
 *  |  5  |  4  |  6  |    1   |
 *  |  6  |  3  |  7  |    1   |
 *  |  7  |  5  |  5  |    1   |  <--- end here
 *  |  8  |  5  |  7  |    1   |
 *  |  9  |  3  |  6  |    1   |
 *  ----------------------------
 *
 *  Ordered in terms of X
 *
 *  ----------------------------
 *  |  id |  X  |  y  | weight |
 *  ----------------------------
 *  |  0  |  0  |  3  |    1   |
 *  |  1  |  2  |  2  |    1   |
 *  |  3  |  0  |  3  |    1   |  <--- start here
 *  |  4  |  1  |  4  |    1   |
 *  |  2  |  2  |  3  |    1   |  <--- best cut point
 *  |  6  |  3  |  7  |    1   |
 *  |  5  |  4  |  6  |    1   |
 *  |  7  |  5  |  5  |    1   |  <--- end here
 *  |  8  |  5  |  7  |    1   |
 *  |  9  |  3  |  6  |    1   |
 *  ----------------------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  4  |   134.75    |
 *  |  5  |   141.33    |  <--- best improvement
 *  |  6  |   132.75    |
 *  ---------------------
 */
TEST_CASE("split_node subset unweighted toy dataset", "[splitter]") {
    arma::vec y{0, 2, 3, 3, 4, 6, 7, 5, 7, 6};
    arma::vec X{3, 2, 2, 0, 1, 4, 3, 5, 5, 3};
    arma::vec sample_weight(10, arma::fill::ones);

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight);

    // reset node so that each split has less than 1 sample at the children
    splitter.reset_node(2, splitter.get_num_samples() - 2);

    // check within node impurity (var([3, 4, 3, 7, 6, 5]))
    double impurity = splitter.node_impurity();
    drforest::testing::almost_equal(impurity, 2.222, 1e-3);

    drforest::SplitRecord split = splitter.split_node(impurity);

    // make sure we pick the correct split point
    REQUIRE(split.threshold == 2.5);
    REQUIRE(split.pos == 5);
    REQUIRE(!split.is_leaf);

    // This is the variance of the samples in the left node (3, 3, 4)
    drforest::testing::almost_equal(split.impurity_left, 0.222, 1e-3);

    // This is the variance of the samples in the right node (6, 7, 5)
    drforest::testing::almost_equal(split.impurity_right, 0.666, 1e-3);

    // Since there are equal number of samples in each node the improvement
    // is (6/10) * (impurity - 0.5 * left - 0.5 * right)
    drforest::testing::almost_equal(split.improvement, 1.06665, 1e-3);

    //// finally make sure samples is rearranged properly
    arma::uvec expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    drforest::testing::assert_vector_equal(splitter.get_node_samples(),
                                           expected);
}

/**
 * Check that the splitting rule is correct on a subset of the following
 * toy dataset:
 *  --------------------------
 *  | id  | X   | y | weight |
 *  --------------------------
 *  | 0   | 3   | 1 |   1    |
 *  | 1   | 2   | 2 |   2    |
 *  | 2   | 2   | 3 |   0    |
 *  | 3   | 0   | 2 |   1    |   <--- start here
 *  | 4   | 1   | 3 |   2    |
 *  | 5   | 4   | 4 |   3    |
 *  | 6   | 3   | 6 |   0    |
 *  | 7   | 5   | 7 |   2    |
 *  | 8   | 5   | 5 |   1    |   <--- end here
 *  | 9   | 3   | 5 |   2    |
 *  | 10  | 6   | 6 |   1    |
 *  --------------------------
 *
 *  The reduced dataset is
 *
 *  ------------------
 *  | X | y | weight |
 *  ------------------
 *  | 3 | 1 |   1    |
 *  | 2 | 2 |   2    |
 *  | 0 | 2 |   1    |  <--- start here
 *  | 1 | 3 |   2    |
 *  | 4 | 4 |   3    |  <--- best split pos (samples[0:5] in split)
 *  | 5 | 7 |   2    |
 *  | 5 | 5 |   1    |  <--- end here
 *  | 3 | 5 |   2    |
 *  | 6 | 6 |   1    |
 *  ------------------
 *
 *  Subset sorted in terms of X
 *
 *  ------------------
 *  | X | y | weight |
 *  ------------------
 *  | 0 | 2 |   1    |  <--- start here
 *  | 1 | 3 |   2    |
 *  | 4 | 4 |   3    |  <--- best pos [2, 3, 4] [7, 5]
 *  | 5 | 7 |   2    |
 *  | 5 | 5 |   1    |  <--- end here
 *  ------------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  4  |   181.5     |
 *  |  5  |   187.0     |
 *  ---------------------
 *
 *  Note that pos refers to one pass the last sample in the node
 *  due to slice notation.
 */
TEST_CASE("split_node larger subset with weights", "[splitter]") {
    // make dataset
    arma::vec y{1, 2, 3, 2, 3, 4, 6, 7, 5, 5, 6};
    arma::vec X{3, 2, 2, 0, 1, 4, 3, 5, 5, 3, 6};
    arma::vec sample_weight{1, 2, 0, 1, 2, 3, 0, 2, 1, 2, 1};

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight);

    // reset node so that each split has less than 1 sample at the children
    splitter.reset_node(2, splitter.get_num_samples() - 2);

    // check within node impurity (var([3, 4, 3, 7, 6, 5]))
    double impurity = splitter.node_impurity();
    drforest::testing::almost_equal(impurity, 2.666, 1e-3);

    // perform the split
    drforest::SplitRecord split = splitter.split_node(impurity);

    // make sure we pick the correct split point
    REQUIRE(split.threshold == 4.5);
    REQUIRE(split.pos == 5);
    REQUIRE(!split.is_leaf);

    // This is the variance of the samples in the left node (2, 3, 4)
    drforest::testing::almost_equal(split.impurity_left, 0.555, 1e-3);

    // This is the variance of the samples in the right node (6, 7, 5)
    drforest::testing::almost_equal(split.impurity_right, 0.888, 1e-3);

    // is (9/15) * (impurity - (6/9) * left - (3/9) * right)
    drforest::testing::almost_equal(split.improvement, 1.2, 1e-3);

    // finally make sure samples is rearranged properly
    arma::uvec expected{0, 1, 3, 4, 5, 7, 8, 9, 10};
    drforest::testing::assert_vector_equal(splitter.get_node_samples(),
                                           expected);
}

/**
 * Test that we do not return a valid split if we do not meet the minimum
 * sample requirements at each node.
 */
TEST_CASE("split_node does not split if < min_samples_leaf", "[splitter]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec X{2, 0, 1, 4, 3, 5};
    arma::vec sample_weight(6, arma::fill::ones);

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight, 1);

    // reset node so that each split has less than 1 sample at the children
    splitter.reset_node(1, splitter.get_num_samples() - 2);

    // split away
    double impurity = splitter.node_impurity();
    drforest::SplitRecord split = splitter.split_node(impurity);

    REQUIRE(split.is_leaf);
}


/**
 * Test we do not return splits if they do not meet the minimum
 * weighted samples requirement.
 */
TEST_CASE("split_node does not split if < weighted_n_leaf", "[splitter]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec X{2, 0, 1, 4, 3, 5};
    arma::vec sample_weight{2, 0, 3, 1, 2, 0};

    // initialize the splitter
    drforest::DimensionReductionSplitter splitter(X, y, sample_weight, 1, 2, 4);
    splitter.reset_node(1, splitter.get_num_samples());

    // split away
    double impurity = splitter.node_impurity();
    drforest::SplitRecord split = splitter.split_node(impurity);

    REQUIRE(split.is_leaf);
}
