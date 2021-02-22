#include <vector>
#include <algorithm>
#include <map>

#include "drforest.h"

namespace drforest {

    Tree::Tree() :
        node_count_(0),
        max_depth_(0),
        directions_() {}

    uint64_t Tree::add_node(uint64_t parent_id, uint64_t depth,
                            bool is_left, bool is_leaf,
                            arma::vec &direction, double threshold,
                            double impurity, double value,
                            uint64_t num_node_samples,
                            double weighted_n_node_samples) {
        uint64_t node_id = node_count_;

        Node node;
        node.node_id = node_id;
        node.depth = depth;
        node.impurity = impurity;
        node.num_node_samples = num_node_samples;
        node.weighted_n_node_samples = weighted_n_node_samples;

        // update max depth if necessary
        if (node.depth > max_depth_) {
            max_depth_ = node.depth;
        }

        if (parent_id != kTreeUndefined) {  // if not the root node
            if (is_left) {
                nodes_.at(parent_id).left_child = node_id;
            } else {
                nodes_.at(parent_id).right_child = node_id;
            }
        }

        if (is_leaf) {
            node.left_child = kTreeLeaf;
            node.right_child = kTreeLeaf;
            node.threshold = kTreeUndefined;
            node.value = value;
            node.feature = kTreeUndefined;
        } else {
            // left_child and right_child will be set later
            node.threshold = threshold;
            node.value = value;
            node.feature = directions_.n_cols;
            directions_.insert_cols(node.feature, std::move(direction));
        }

        nodes_.push_back(node);
        node_count_++;

        return node_id;
    }

    arma::uvec Tree::apply(const arma::mat &X) const {
        Node node;
        arma::rowvec X_sample;
        arma::vec X_sdr;
        arma::uvec out(X.n_rows);
        for (arma::uword i = 0; i < X.n_rows; ++i) {
            X_sample = X.row(i);
            node = get_node(0);  // root node
            while (node.left_child != kTreeLeaf) {
                X_sdr = X_sample * directions_.col(node.feature);
                if (X_sdr(0) <= node.threshold) {
                    node = get_node(node.left_child);
                } else{
                    node = get_node(node.right_child);
                }
            }
            out(i) = node.node_id;
        }

        return out;
    }

    arma::vec Tree::predict(const arma::mat &X) const {
        Node node;
        arma::rowvec X_sample;
        arma::vec X_sdr;
        arma::vec out(X.n_rows);
        for (arma::uword i = 0; i < X.n_rows; ++i) {
            X_sample = X.row(i);
            node = get_node(0);  // root node
            while (node.left_child != kTreeLeaf) {
                X_sdr = X_sample * directions_.col(node.feature);
                if (X_sdr(0) <= node.threshold) {
                    node = get_node(node.left_child);
                } else{
                    node = get_node(node.right_child);
                }
            }
            out(i) = node.value;
        }

        return out;
    }

    arma::umat Tree::decision_path(const arma::mat &X) const {
        size_t num_samples = X.n_rows;
        size_t num_nodes = nodes_.size();
        arma::umat out(num_samples, num_nodes, arma::fill::zeros);

        // loop through samples and determine their path through the tree
        Node node;
        arma::vec X_sdr;
        for (arma::uword i = 0; i < num_samples; ++i) {
            node = get_node(0);  // root node
            while (node.left_child != kTreeLeaf) {
                // record the node that the sample passes through
                out(i, node.node_id) = 1;

                // determine the next node
                X_sdr = X.row(i) * directions_.col(node.feature);
                if (X_sdr(0) <= node.threshold) {
                    node = get_node(node.left_child);
                } else{
                    node = get_node(node.right_child);
                }
            }

            // also record the leaf node
            out(i, node.node_id) = 1;
        }

        return out;
    }

    std::pair<arma::uvec, arma::uvec>
    Tree::get_leaf_slices(const arma::mat &X) const {
        // Determine leaf node id for each sample
        arma::uvec leaf_ids = apply(X);

        // Find the y values at each leaf node as well as the number of
        // samples at each node
        auto [unique_ids, counts] = drforest::math::unique_counts(leaf_ids);
        arma::vec values(unique_ids.n_elem);
        for (int i = 0; i < unique_ids.n_elem; ++i) {
            values(i) = nodes_.at(unique_ids(i)).value;
        }
        arma::uvec value_order = arma::sort_index(values);

        // Build a map from leaf_id to ordered index
        std::map<uint, uint> encoder;
        std::transform(unique_ids.begin(), unique_ids.end(),
                       value_order.begin(),
                       std::inserter(encoder, encoder.end()),
                       std::make_pair<uint const&, uint const&>);

        // re-arrange the leaf_ids and the counts in ascending order by leaf
        // value
        counts = counts.rows(value_order);
        for(int i = 0; i < leaf_ids.n_elem; ++i) {
            leaf_ids(i) = encoder[leaf_ids(i)];
        }

        return { leaf_ids, counts };
    }

    arma::mat Tree::estimate_sufficient_dimensions(
            const arma::mat &X,
            DimensionReductionAlgorithm dr_algo) const {
        auto [leaf_ids, counts] = get_leaf_slices(X);

        // apply the appropriate dimension reduction algorithm
        if (dr_algo == DimensionReductionAlgorithm::SIR) {
            return fit_sliced_inverse_regression(X, leaf_ids, counts);
        } else {
            return fit_sliced_average_variance_estimation(X, leaf_ids, counts);
        }
    }

    arma::mat Tree::estimate_M(const arma::mat &X, const arma::mat &Z) const {
        auto [leaf_ids, counts] = get_leaf_slices(X);
        return estimate_sir_M(Z, leaf_ids, counts);
    }

    arma::uvec Tree::generate_oob_indices(bool return_unsorted_indices) const {
        uint num_samples = row_order_.n_rows;

        BootstrapSampler bootstrap(num_samples, random_state_);
        arma::vec sample_weight = bootstrap.draw();

        arma::uvec oob_indices = arma::find(sample_weight == 0.0);
        if (return_unsorted_indices) {
            return row_order_.rows(oob_indices);
        }
        return oob_indices;
    }

}  // namespace drforest
