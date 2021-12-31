#include <cmath>
#include <stack>
#include <memory>

#include "drforest.h"

namespace drforest {

    TreeBuilder::TreeBuilder(std::shared_ptr<NodeSplitter> splitter,
                             int max_depth,
                             size_t min_samples_leaf) :
                splitter_(splitter),
                max_depth_(0),
                min_samples_leaf_(min_samples_leaf),
                min_weight_leaf_(min_samples_leaf) {
        max_depth_ = (max_depth > 0) ? (size_t)max_depth : std::pow(2, 31) - 1;
    }

    std::shared_ptr<Tree> TreeBuilder::build(const arma::mat &X,
                                             const arma::vec &y,
                                             const arma::vec &sample_weight) {

        // usefel variables
        bool first = true;

        size_t start;
        size_t end;
        uint64_t depth;
        uint64_t parent;
        uint64_t max_depth;
        bool is_left;
        bool is_leaf;
        double impurity;

        SplitRecord split;
        uint64_t node_id;
        uint64_t n_node_samples;
        double node_value;
        double weighted_n_node_samples;

        // intantiate the tree
        std::shared_ptr<Tree> tree = std::make_shared<Tree>();

        // Build the tree in a depth first (LIFO) fashion using a stack
        std::stack<BuildRecord> stack;
        stack.emplace(0, splitter_->get_num_samples(), 0,
                      kTreeUndefined, false, INFINITY);
        while (!stack.empty()) {
            // extract values from top of the stack
            start = stack.top().start;
            end = stack.top().end;
            depth = stack.top().depth;
            parent = stack.top().parent;
            is_left = stack.top().is_left;
            impurity = stack.top().impurity;
            stack.pop();

            n_node_samples = end - start;
            weighted_n_node_samples = splitter_->reset_node(start, end);

            is_leaf = (depth >= max_depth_ ||
                       n_node_samples < 2 * min_samples_leaf_ ||
                       impurity <= 0);

            if (first) {
                impurity = splitter_->node_impurity();
                first = false;
            }

            // not a leaf node so split!
            if (!is_leaf) {
                split = splitter_->split_node(impurity);
                is_leaf = (is_leaf || split.pos >= end);
            }

            // add the node to the tree structure
            node_value = splitter_->node_value();
            node_id = tree->add_node(parent, depth, is_left, is_leaf,
                                     split.direction, split.threshold,
                                     impurity, node_value, n_node_samples,
                                     weighted_n_node_samples);

            if (!is_leaf) {
                // add right child onto the stack
                stack.emplace(split.pos, end, depth + 1, node_id, false,
                              split.impurity_right);

                // add left child onto the stack
                stack.emplace(start, split.pos, depth + 1, node_id, true,
                              split.impurity_left);

            }
        }

        return tree;
    }

    std::shared_ptr<Tree> build_dimension_reduction_tree(
            arma::mat &X, arma::vec &y,
            arma::vec &sample_weight,
            arma::uvec &numeric_features, arma::uvec &categorical_features,
            int max_features, int num_slices, int max_depth,
            size_t min_samples_leaf, bool use_original_features,
            bool presorted, uint seed) {
        // sort X, y in terms of y.
        if (!presorted) {
            arma::uvec y_order = arma::stable_sort_index(y);
            y = y.rows(y_order);
            X = X.rows(y_order);
            sample_weight = sample_weight.rows(y_order);
        }
        FeatureInfo feat_info(numeric_features, categorical_features);

        // use the DR spliting rule
        std::shared_ptr<drforest::NodeSplitter> splitter =
            std::make_shared<drforest::DimensionReductionSplitter>(
                X, y, sample_weight, feat_info, max_features,
                min_samples_leaf, min_samples_leaf,
                num_slices, use_original_features, seed);

        TreeBuilder builder(splitter, max_depth, min_samples_leaf);

        return builder.build(X, y, sample_weight);
    }

} // namespace drforest
