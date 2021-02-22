#include <memory>
#include <stack>

#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"


// smoke test the builder matches the first dimension of the bootstrap
// and weighted sample
TEST_CASE("Try out builder", "[builder]") {
    auto [X, y] = drforest::testing::load_athletes();

    // draw bootstrap sample
    drforest::BootstrapSampler bootstrap(y.n_elem);
    auto [indices, sample_weight] = bootstrap.draw_indices();

    auto tree_one = drforest::build_dimension_reduction_tree(
        X, y, sample_weight, -1, 10, 1);
    arma::mat directions = arma::abs(tree_one->get_directions());

    // actually materialize the bootstrap (we load the dataset again
    // because it was sorted in-place).
    auto [X2, y2] = drforest::testing::load_athletes();
    arma::mat X_boot = X2.rows(indices);
    arma::vec y_boot = y2.rows(indices);
    sample_weight = arma::vec(y2.n_elem, arma::fill::ones);
    auto tree_two = drforest::build_dimension_reduction_tree(
        X_boot, y_boot, sample_weight, -1, 10, 1);

    arma::mat directions_boot = arma::abs(tree_two->get_directions());

    drforest::testing::assert_matrix_allclose(directions, directions_boot);
}
