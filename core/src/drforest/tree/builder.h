#pragma once

namespace drforest {

    // Book keeping for the current node
    class BuildRecord {
    public:
        BuildRecord(size_t start_, size_t end_, size_t depth_,
                    size_t parent_, bool is_left_, double impurity_) :
            start(start_), end(end_), depth(depth_), parent(parent_),
            is_left(is_left_), impurity(impurity_) {}

        size_t start;
        size_t end;
        size_t depth;
        size_t parent;
        bool is_left;
        double impurity;
    };

    class TreeBuilder {
    public:
        TreeBuilder(std::shared_ptr<NodeSplitter> splitter,
                    int max_depth=-1,
                    size_t min_samples_leaf=2);

        std::shared_ptr<Tree> build(const arma::mat &X, const arma::vec &y,
                                    const arma::vec &sample_weight);
    private:
        std::shared_ptr<NodeSplitter> splitter_;
        size_t max_depth_;
        size_t min_samples_leaf_;
        size_t min_weight_leaf_;
    };

    // Builds a tree using the data and given hyperparameters
    std::shared_ptr<Tree> build_dimension_reduction_tree(
            arma::mat &X, arma::vec &y, arma::vec &sample_weight,
            int max_features=-1, int num_slices=10, int max_depth=-1,
            size_t min_samples_leaf=2, bool use_original_features=false, uint seed=42);

}  // namespace drforest
