#include "catch.hpp"

#include "drforest.h"

#include "asserts.h"
#include "datasets.h"

#include <iostream>


//TEST_CASE("Tree Data Structure Tests", "[tree]") {
//    drforest::Tree tree(10);
//
//    arma::vec beta = arma::ones<arma::vec>(10);
//
//    // root node
//    size_t node_id  = tree.add_node(drforest::kTreeUndefined, true, false, beta, 3, 10);
//    REQUIRE(node_id == 0);
//
//    drforest::testing::assert_vector_equal(tree.get_node(0).beta, beta);
//
//    REQUIRE(tree.get_node(0).threshold == 3);
//    REQUIRE(tree.get_node(0).num_node_samples == 10);
//
//    // add a left node
//    node_id = tree.add_node(node_id, true, true, beta, 3, 5);
//    REQUIRE(node_id == 1);
//    drforest::Node root_node  = tree.get_node(0);
//    REQUIRE(root_node.left_child == node_id);
//    REQUIRE(tree.get_node(1).left_child == drforest::kTreeLeaf);
//
//    node_id = tree.add_node(0, false, false, beta, 3, 5);
//    tree.add_node(node_id, true, true, beta, 5, 10);
//    tree.add_node(node_id, false, true, beta, 3, 23);
//
//    std::vector<int> left_children = tree.get_children_left();
//    //for(auto child_id : left_children) {
//    //    std::cout << child_id << std::endl;
//    //}
//
//    std::vector<int> right_children = tree.get_children_right();
//    //for(auto child_id : right_children) {
//    //    std::cout << child_id << std::endl;
//    //}
//
//    //std::cout << tree.get_directions() << std::endl;
//}

//TEST_CASE("Tree Depth Two Tree", "[tree_depth_two]") {
//    auto [X, y] = drforest::testing::load_athletes();
//    arma::mat directions =
//        drforest::fit_sliced_inverse_regression(X, y, 11);
//    arma::vec beta = directions.row(1).t();
//
//    drforest::Tree tree(X.n_cols);
//    size_t node_id = tree.add_node(drforest::kTreeUndefined, true, false, beta, 3.4, X.n_rows);
//    tree.directions_
//
//    tree.add_node(node_id, true, true, drforest::kTreeUndefined, 91);
//    tree.add_node(node_id, false, true, drforest::kTreeUndefined, 111);
//}
