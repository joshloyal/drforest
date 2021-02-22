#include <algorithm>

#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

/**
 * This is an unweighted dataset where y only takes two values.
 * The criterion should identify the cut point as the value splitting
 * the two values (10 / 20).
 */
TEST_CASE("Obvious cut-point", "[criterion]") {
    arma::vec y = arma::join_cols(10 * arma::vec(100, arma::fill::ones),
                                  20 * arma::vec(100, arma::fill::ones));
    drforest::WeightVec sample_weight(y.n_rows, arma::fill::ones);

    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);

    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);

    criterion.reset_node(0, y.n_rows);

    arma::vec impurities(y.n_rows - 1);
    for(int i = 1; i < y.n_rows; ++i) {
        criterion.update(i);
        impurities(i - 1) = criterion.proxy_impurity_improvement();
    }

    // the maximum improvement should be right at the mid-point
    REQUIRE(impurities.index_max() == 99);
    REQUIRE(impurities.max() == 50000);  // hand calculated max

    // check impurity improvement increases until 99 and then decreases
    // afterward
    arma::uvec first_decrease = arma::find(arma::diff(impurities) < 0, 1);
    REQUIRE(first_decrease(0) == 99);
}


/**
 * Check that the criterion is correct on the following toy dataset:
 *
 *  --------------
 *  | y | weight |
 *  --------------
 *  | 3 |   1    |
 *  | 3 |   1    |
 *  | 4 |   1    |
 *  | 6 |   1    |
 *  | 7 |   1    |
 *  | 5 |   1    |
 *  --------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  1  |   134.0     |
 *  |  2  |   139.0     |
 *  |  3  |   141.33    |
 *  |  4  |   136.0     |
 *  |  5  |   130.8     |
 *  ---------------------
 */
TEST_CASE("Simple MSE Test No Weights", "[criterion]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec sample_weight(6, arma::fill::ones);

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 6);
    REQUIRE(samples.n_rows == 6);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(0, samples.n_rows);

    // pos = 1
    criterion.update(1);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    134.0);

    // pos = 2
    criterion.update(2);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    139.0);

    // pos = 3
    criterion.update(3);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    141.333, 1e-3);

    // pos = 4
    criterion.update(4);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    136.0);

    // pos = 5
    criterion.update(5);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    130.8);
}


/**
 * Same as the previous test except we do not start at the beginning of
 * the array.
 *
 *  --------------
 *  | y | weight |
 *  --------------
 *  | 3 |   1    |
 *  | 3 |   1    | <--- start here
 *  | 4 |   1    |
 *  | 6 |   1    | <--- end here
 *  | 7 |   1    |
 *  | 5 |   1    |
 *  --------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  2  |   59.0      |
 *  |  3  |   60.5      |
 *  ---------------------
 */
TEST_CASE("MSE Subset No Weights", "[criterion]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec sample_weight(6, arma::fill::ones);

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 6);
    REQUIRE(samples.n_rows == 6);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(1, samples.n_rows - 2);

    // pos = 2
    criterion.update(2);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    59.0);

    // pos = 3
    criterion.update(3);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    60.5);
}

/**
 * Check that the splitting rule is correct on the following toy dataset
 * and subset. The subset is bigger in this case.
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
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  4  |   139.0     |
 *  |  5  |   141.33    |  <--- best improvement
 *  |  6  |   136.0     |
 *  ---------------------
 */
TEST_CASE("criterion larger subset unweighted toy dataset", "[criterion]") {
    arma::vec y{0, 2, 3, 3, 4, 6, 7, 5, 7, 6};
    arma::vec X{3, 2, 2, 0, 1, 4, 3, 5, 5, 3};
    arma::vec sample_weight(10, arma::fill::ones);

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 10);
    REQUIRE(samples.n_rows == 10);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(2, samples.n_rows - 2);

    // pos = 4
    criterion.update(4);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    139.0, 1e-3);

    // pos = 5
    criterion.update(5);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    141.333, 1e-3);

    // pos = 6
    criterion.update(6);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    136.0, 1e-3);
}

/**
 * Check that the criterion is correct on the following toy dataset:
 *
 *  --------------
 *  | y | weight |
 *  --------------
 *  | 3 |   0    |
 *  | 3 |   2    |
 *  | 4 |   3    |
 *  | 6 |   0    |
 *  | 7 |   2    |
 *  | 5 |   1    |
 *  --------------
 *
 *  The reduced dataset is

 *  --------------
 *  | y | weight |
 *  --------------
 *  | 3 |   2    |
 *  | 4 |   3    |
 *  | 7 |   2    |
 *  | 5 |   1    |
 *  --------------
 *
 *  The criterion should be

 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  1  |   178.166   |
 *  |  2  |   185.133   |
 *  |  3  |   171.285   |
 *  ---------------------
 */
TEST_CASE("Simple MSE Test With Weights", "[criterion]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec sample_weight{0, 2, 3, 0, 2, 1};

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 8.0);
    REQUIRE(samples.n_rows == 4);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(0, samples.n_rows);

    // pos = 1
    criterion.update(1);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    178.166, 1e-3);

    // pos = 2
    criterion.update(2);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    185.133, 1e-3);

    // pos = 3
    criterion.update(3);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    171.285, 1e-3);
}

/**
 * Same as the previous test except we do not start at the beginning of
 * the array.
 *
 *  --------------
 *  | y | weight |
 *  --------------
 *  | 3 |   2    |
 *  | 4 |   3    |  <-- node start
 *  | 7 |   2    |
 *  | 5 |   1    |  <-- node end
 *  --------------
 *
 *  The criterion should be
 *
 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  2  |   168.333   |
 *  |  3  |   160.2     |
 *  ---------------------
 */
TEST_CASE("MSE Subset With Weights", "[criterion]") {
    // make dataset
    arma::vec y{3, 3, 4, 6, 7, 5};
    arma::vec sample_weight{0, 2, 3, 0, 2, 1};

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 8.0);
    REQUIRE(samples.n_rows == 4);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(1, samples.n_rows);

    // pos = 2
    criterion.update(2);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    168.333, 1e-3);

    //// pos = 3
    criterion.update(3);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    160.2);
}


/**
 * Check that the criterion is correct on the following augmented toy dataset.
 * It is enlarged in order for the subset to be non-trivial.
 *
 *  --------------
 *  | y | weight |
 *  --------------
 *  | 1 |   1    |
 *  | 2 |   2    |
 *  | 3 |   0    |
 *  | 2 |   1    |   <--- start here
 *  | 3 |   2    |
 *  | 4 |   3    |
 *  | 6 |   0    |
 *  | 7 |   2    |
 *  | 5 |   1    |   <--- end here
 *  | 5 |   2    |
 *  | 6 |   1    |
 *  --------------
 *
 *  The reduced dataset is

 *  --------------
 *  | y | weight |
 *  --------------
 *  | 1 |   1    |
 *  | 2 |   2    |
 *  | 2 |   1    |  <--- start here
 *  | 3 |   2    |
 *  | 4 |   3    |  <--- best cut (<= 4)
 *  | 7 |   2    |
 *  | 5 |   1    |  <--- end here
 *  | 5 |   2    |
 *  | 6 |   1    |
 *  --------------
 *
 *  The criterion should be

 *  ---------------------
 *  | pos | improvement |
 *  ---------------------
 *  |  4  |   181.5     |
 *  |  5  |   187.0     |
 *  |  6  |   181.5     |
 *  ---------------------
 */
TEST_CASE("Simple MSE Test Larger Subset With Weights", "[criterion]") {
    // make dataset
    arma::vec y{1, 2, 3, 2, 3, 4, 6, 7, 5, 5, 6};
    arma::vec sample_weight{1, 2, 0, 1, 2, 3, 0, 2, 1, 2, 1};

    // initialize sample index used for sweeping through the criterion
    arma::uvec samples;
    double weighted_n_samples;
    drforest::init_sample_index(sample_weight, samples, weighted_n_samples);
    REQUIRE(weighted_n_samples == 15.0);
    REQUIRE(samples.n_rows == 9);

    // reset and indicate we are looking at every sample
    drforest::MeanSquaredError criterion(y, sample_weight, samples,
                                         weighted_n_samples);
    criterion.reset_node(2, samples.n_rows - 2);

    // pos = 4
    criterion.update(4);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    181.5, 1e-3);

    // pos = 5
    criterion.update(5);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    187.0, 1e-3);

    // pos = 6
    criterion.update(6);
    drforest::testing::almost_equal(criterion.proxy_impurity_improvement(),
                                    169.5, 1e-3);
}
