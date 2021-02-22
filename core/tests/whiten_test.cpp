#include "drforest.h"

#include <math.h>

#include "datasets.h"
#include "asserts.h"
#include "catch.hpp"


TEST_CASE("Bootstrap vs. Weighted Full Sample", "[whiten]") {
    auto [X, y] = drforest::testing::load_athletes();

    drforest::BootstrapSampler bootstrap(y.n_elem);
    auto [indices, sample_weight] = bootstrap.draw_indices();

    // remove rows with zero weight since we do this in the tree building code
    arma::uvec nonzero = arma::find(sample_weight != 0.0);
    arma::mat X_weighted = X.rows(nonzero);
    sample_weight = sample_weight(nonzero);

    // perform weighted whitening
    auto [Z_weighted, R_weighted] = drforest::math::whiten(X_weighted,
                                                           sample_weight);

    // materialize the bootstrap sample and whiten the sample directly
    // with no sample weights
    X = X.rows(indices);
    auto [Z_full, R_full] = drforest::math::whiten(X);

    // to compare extract the unique rows (this skips the last element)
    arma::uvec unique_indices = arma::find(arma::diff(indices) > 0);
    arma::uvec last_elem{indices(indices.n_rows-1)};
    unique_indices = arma::join_cols(unique_indices, last_elem);
    Z_full = Z_full.rows(unique_indices);

    // these matrices should be equal up to some sign changes
    Z_full = arma::abs(Z_full);
    Z_weighted = arma::abs(Z_weighted);
    drforest::testing::assert_matrix_allclose(Z_full, Z_weighted);

    // R matrices should also match (the diagonals are not unique by sign)
    R_full = arma::abs(R_full);
    R_weighted = arma::abs(R_weighted);
    drforest::testing::assert_matrix_allclose(R_full, R_weighted);
}


// The difference here is that in a sub-sample the sample weights
// do not add up to the number of rows
TEST_CASE("Bootstrap vs. Weighted Sub-Sample", "[whiten]") {
    auto [X, y] = drforest::testing::load_athletes();

    drforest::BootstrapSampler bootstrap(y.n_elem);
    auto [indices, sample_weight] = bootstrap.draw_indices();


    // take the first 100 rows
    arma::mat X_sub = X.rows(0, 99);
    sample_weight = sample_weight.rows(0, 99);

    // remove rows with zero weight since we do this in the tree building code
    arma::uvec nonzero = arma::find(sample_weight != 0.0);
    arma::mat X_weighted = X_sub.rows(nonzero);
    sample_weight = sample_weight(nonzero);

    // perform weighted whitening
    auto [Z_weighted, R_weighted] = drforest::math::whiten(X_weighted,
                                                           sample_weight);

    // materialize the bootstrap sample and whiten the sample directly
    // with no sample weights
    X = X.rows(indices);
    arma::uvec index = arma::find(indices == 99); // first 100 samples
    indices = indices.rows(0, index(0));
    X = X.rows(0, index(0));
    auto [Z_full, R_full] = drforest::math::whiten(X);

    // to compare extract the unique rows (this skips the last element)
    arma::uvec unique_indices = arma::find(arma::diff(Z_full.col(0)) != 0);
    arma::uvec last_elem{Z_full.n_rows-1};
    unique_indices = arma::join_cols(unique_indices, last_elem);
    Z_full = Z_full.rows(unique_indices);

    // these matrices should be equal up to some sign changes
    Z_full = arma::abs(Z_full);
    Z_weighted = arma::abs(Z_weighted);
    drforest::testing::assert_matrix_allclose(Z_full, Z_weighted);

    // R matrices should also match (the diagonals are not unique by sign)
    R_full = arma::abs(R_full);
    R_weighted = arma::abs(R_weighted);
    drforest::testing::assert_matrix_allclose(R_full, R_weighted);
}
