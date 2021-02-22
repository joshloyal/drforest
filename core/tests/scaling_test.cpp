#include "drforest.h"

#include <math.h>

#include "asserts.h"
#include "catch.hpp"


TEST_CASE("Standardizing a Matrix", "[standardize]") {

    arma::mat X = arma::randn<arma::mat>(100, 2);

    SECTION( "Results in columns with zero mean / std" ) {
        arma::mat X_scaled = drforest::math::standardize(X);

        arma::mat means = arma::mean(X_scaled);
        REQUIRE(arma::all(arma::vectorise(arma::abs(means)) < 1e-10));

        arma::mat stddevs = arma::stddev(X_scaled);
        REQUIRE(arma::all(arma::vectorise(arma::abs(stddevs - 1.0)) < 1e-10));
    }

    SECTION( "Does not scale zero variance features" ) {
        // set the first column to all ones
        X.col(0) = arma::ones<arma::colvec>(X.n_rows);

        arma::mat X_scaled = drforest::math::standardize(X);

        // means should still be roughly zero
        arma::mat means = arma::mean(X_scaled);
        REQUIRE(arma::all(arma::vectorise(arma::abs(means)) < 1e-10));

        arma::mat stddevs = arma::stddev(X_scaled);

        // first column is not scaled
        REQUIRE(stddevs(0)  == 0.0);

        // this column is still scaled properly
        drforest::testing::almost_equal(stddevs(1), 1.0);
    }
}

TEST_CASE("Weighted Mean", "[weighted-mean]") {
    drforest::DataMat X("1, 2, 3;4, 5, 6;7, 8, 9");
    drforest::WeightVec sample_weight{2, 0, 1};

    arma::vec X_mean = drforest::math::weighted_mean(X, sample_weight).t();
    arma::vec expected{3, 4, 5};

    drforest::testing::assert_vector_equal(X_mean, expected);
}

TEST_CASE("Weighted Center", "[weighted-center]") {
    drforest::DataMat X("1, 2, 3;4, 5, 6;7, 8, 9");
    drforest::WeightVec sample_weight{2, 0, 1};

    drforest::DataMat X_center = drforest::math::center(X, sample_weight);

    drforest::DataMat X_expected("-2, -2, -2; 1, 1, 1; 4, 4, 4");
    drforest::testing::assert_matrix_equal(X_center, X_expected);
}
