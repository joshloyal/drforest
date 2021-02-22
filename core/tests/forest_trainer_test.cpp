#include <memory>

#include "catch.hpp"

#include "drforest.h"
#include "asserts.h"
#include "datasets.h"


TEST_CASE("Forest Trainer", "[forest]") {
    //auto [X, y] = drforest::testing::load_athletes();
    auto [X, y] = drforest::testing::load_quadratic();
    //auto [X, y] = drforest::testing::make_checkerboard();
    std::cout << X.n_rows << " " << X.n_cols << std::endl;

    drforest::RandomForestTrainer trainer(
        100, -1, 10, -1,  2, true, -1, 123);

    std::shared_ptr<drforest::RandomForest> forest = trainer.train(X, y);

    std::shared_ptr<drforest::Tree> tree = forest->get_tree(1);

    arma::vec preds = tree->predict(X);

    std::cout << arma::mean(arma::square(arma::mean(y) - y)) << std::endl;
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    preds = forest->predict(X, -1);
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    preds = forest->predict(X, -1);
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    forest = trainer.train(X, y);
    preds = forest->predict(X, -1);
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    preds = forest->get_oob_predictions();
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    std::cout << forest->get_oob_error() << std::endl;

    //std::cout << forest->estimate_sufficient_dimension(X, -1) << std::endl;
}

TEST_CASE("Forest Trainer Permuted", "[permuted_forest]") {
    auto [X, y] = drforest::testing::load_quadratic();
    //auto [X, y] = drforest::testing::make_checkerboard();
    std::cout << X.n_rows << " " << X.n_cols << std::endl;

    drforest::RandomForestTrainer trainer(
        100, -1, 10, -1,  2, true, -1, 123);

    std::shared_ptr<drforest::RandomForest> forest = trainer.train(X, y);
    arma::vec preds =  forest->predict(X, -1);
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;

    std::shared_ptr<drforest::RandomForest> forest_perm = trainer.train_permuted(X, y, 1);
    preds =  forest_perm->predict(X, -1);
    std::cout << arma::mean(arma::square(preds - y)) << std::endl;
}
