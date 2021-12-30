#pragma once

#include <memory>

namespace drforest {

    class RandomForestTrainer {
    public:
        RandomForestTrainer(size_t num_trees=50,
                            int max_features=-1,
                            size_t num_slices=10, int max_depth=-1,
                            size_t min_samples_leaf=2,
                            bool oob_error=false,
                            int num_threads=1,
                            uint seed=42);

        std::shared_ptr<RandomForest> train(
            arma::mat &X, arma::vec &y, FeatureInfo &feat_info);
        std::shared_ptr<RandomForest> train_permuted(
            arma::mat &X, arma::vec &y, FeatureInfo &feat_info,
            uint feature_id);

    private:
        // hyper-parameters
        size_t num_trees_;
        int max_features_;
        size_t num_slices_;
        int max_depth_;
        size_t min_samples_leaf_;
        bool oob_error_;
        int num_threads_;
        uint seed_;
    };

    std::shared_ptr<RandomForest> train_random_forest(
            arma::mat &X, arma::vec &y,
            arma::uvec &numeric_features, arma::uvec &categorical_features,
            size_t num_trees=50,
            int max_features=-1,
            size_t num_slices=10, int max_depth=-1,
            size_t min_samples_leaf=2,
            bool oob_error=false, int num_threads=1, uint seed=42);

    std::shared_ptr<RandomForest> train_permuted_random_forest(
            arma::mat &X, arma::vec &y,
            arma::uvec &numeric_features, arma::uvec &categorical_features,
            uint feature_id,
            size_t num_trees=50,
            int max_features=-1,
            size_t num_slices=10, int max_depth=-1,
            size_t min_samples_leaf=2,
            bool oob_error=false, int num_threads=1, uint seed=42);

} // namespace drforest
