#include <memory>
#include <vector>

#include "pthread.h"

#include "drforest.h"

namespace drforest {

    RandomForestTrainer::RandomForestTrainer(
            size_t num_trees,
            int max_features,
            size_t num_slices,
            int max_depth, size_t min_samples_leaf,
            bool oob_error, int num_threads, uint seed) :
            num_trees_(num_trees),
            max_features_(max_features),
            num_slices_(num_slices), max_depth_(max_depth),
            min_samples_leaf_(min_samples_leaf),
            oob_error_(oob_error),
            seed_(seed) {
        // determine number of threads to run
        num_threads_ = get_num_threads(num_threads);
    }

    std::shared_ptr<RandomForest> RandomForestTrainer::train(arma::mat &X,
                                                             arma::vec &y,
                                                             FeatureInfo &feat_info) {
        // begin by sorting the data
        arma::uvec y_order = arma::stable_sort_index(y);
        y = y.rows(y_order);
        X = X.rows(y_order);

        // counters
        int t;

        // initialize vector to hold the trees
        std::vector<std::shared_ptr<Tree>> trees;
        trees.resize(num_trees_);

        // set the random states for each tree
        RandomStateSampler sampler(num_trees_, seed_);
        std::vector<uint> random_states = sampler.draw();

        #pragma omp parallel for schedule(static) num_threads(num_threads_) private(t)
        for (t = 0; t < num_trees_; t++) {
            // draw bootstrap
            BootstrapSampler bootstrap(X.n_rows, random_states.at(t));
            arma::vec sample_weight = bootstrap.draw();

            // initialize a splitter
            std::shared_ptr<drforest::NodeSplitter> splitter =
                std::make_shared<drforest::DimensionReductionSplitter>(
                    X, y, sample_weight, feat_info, max_features_,
                    min_samples_leaf_, min_samples_leaf_,
                    num_slices_, false, random_states.at(t));

            // build the tree
            TreeBuilder builder(splitter, max_depth_, min_samples_leaf_);
            trees.at(t) = builder.build(X, y, sample_weight);

            // set the random state for latter oob estimates
            trees.at(t)->set_random_state(random_states.at(t));
            trees.at(t)->set_index_lookup(y_order);
        }

        // create the forest
        auto forest = std::make_shared<RandomForest>(trees, X, y);

        // calculate oob statistics (if requested)
        if (oob_error_) {
            arma::vec oob_preds(X.n_rows, arma::fill::zeros);
            arma::vec n_predictions(X.n_rows, arma::fill::zeros);

            for (auto &tree : forest->get_trees()) {
                arma::uvec unsampled_indices =
                    tree->generate_oob_indices(false);

                arma::vec tree_preds = tree->predict(X.rows(unsampled_indices));

                oob_preds.rows(
                    y_order.rows(unsampled_indices)) += tree_preds;
                n_predictions.rows(
                    y_order.rows(unsampled_indices)) += 1.0;
            }

            // set counts of any samples that had no oob predictions to 1
            if (arma::any(n_predictions == 0)) {
                arma::uvec zero_preds = arma::find(n_predictions == 0);
                n_predictions.rows(zero_preds) += 1.0;
            }

            oob_preds /= n_predictions;
            double oob_error = arma::mean(arma::square(y - oob_preds));

            forest->set_oob_predictions(std::move(oob_preds));
            forest->set_oob_error(oob_error);
        }

        return forest;
    }

    std::shared_ptr<RandomForest> RandomForestTrainer::train_permuted(
            arma::mat &X, arma::vec &y, FeatureInfo &feat_info,
            uint feature_id) {
        // begin by sorting the data
        arma::uvec y_order = arma::stable_sort_index(y);
        y = y.rows(y_order);

        // counters
        int t;

        // initialize vector to hold the trees
        std::vector<std::shared_ptr<Tree>> trees;
        trees.resize(num_trees_);

        // set the random states for each tree
        RandomStateSampler sampler(num_trees_, seed_);
        std::vector<uint> random_states = sampler.draw();

        #pragma omp parallel for schedule(static) num_threads(num_threads_) private(t)
        for (t = 0; t < num_trees_; t++) {
            // permute feature
            Permuter permuter(X.n_rows, random_states.at(t));
            arma::uvec permute = permuter.draw();

            // create a copy here...
            // (this is memory intensive when running parallel on large data)
            arma::mat X_null(X);
            arma::vec X_col = X.col(feature_id);
            X_null.col(feature_id) = X_col.rows(permute);

            // sort X array
            X_null = X_null.rows(y_order);

            // draw bootstrap
            BootstrapSampler bootstrap(X_null.n_rows, random_states.at(t));
            arma::vec sample_weight = bootstrap.draw();

            // initialize a splitter
            std::shared_ptr<drforest::NodeSplitter> splitter =
                std::make_shared<drforest::DimensionReductionSplitter>(
                    X_null, y, sample_weight, feat_info, max_features_,
                    min_samples_leaf_, min_samples_leaf_,
                    num_slices_, false, random_states.at(t));

            // build the tree
            TreeBuilder builder(splitter, max_depth_, min_samples_leaf_);
            trees.at(t) = builder.build(X_null, y, sample_weight);

            // set the random state for latter oob estimates
            trees.at(t)->set_random_state(random_states.at(t));
            trees.at(t)->set_index_lookup(y_order);
        }

        // create the forest
        auto forest = std::make_shared<RandomForest>(trees, X, y);

        return forest;
    }

    std::shared_ptr<RandomForest> train_random_forest(
            arma::mat &X, arma::vec &y,
            arma::uvec &numeric_features, arma::uvec &categorical_features,
            size_t num_trees, int max_features,
            size_t num_slices, int max_depth,
            size_t min_samples_leaf, bool oob_error,
            int num_threads, uint seed) {

        FeatureInfo feat_info(numeric_features, categorical_features);

        RandomForestTrainer trainer(num_trees, max_features,
                                    num_slices, max_depth, min_samples_leaf,
                                    oob_error, num_threads, seed);

        return trainer.train(X, y, feat_info);
    }

    std::shared_ptr<RandomForest> train_permuted_random_forest(
            arma::mat &X, arma::vec &y, uint feature_id,
            arma::uvec &numeric_features, arma::uvec &categorical_features,
            size_t num_trees, int max_features,
            size_t num_slices, int max_depth,
            size_t min_samples_leaf, bool oob_error,
            int num_threads, uint seed) {

        FeatureInfo feat_info(numeric_features, categorical_features);

        RandomForestTrainer trainer(num_trees, max_features,
                                    num_slices, max_depth, min_samples_leaf,
                                    oob_error, num_threads, seed);

        return trainer.train_permuted(X, y, feat_info, feature_id);
    }


}  // namespace drforest
