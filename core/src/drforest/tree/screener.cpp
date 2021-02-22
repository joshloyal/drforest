#include <utility>

#include "drforest.h"

namespace drforest {

    FeatureScreener::FeatureScreener(const int max_features,
                                     const int min_samples_leaf,
                                     const int min_weight_leaf) :
        max_features_(max_features),
        min_samples_leaf_(min_samples_leaf),
        min_weight_leaf_(min_weight_leaf) {}

    arma::uvec FeatureScreener::screen_features(
            const arma::mat &X, const arma::vec &y,
            const WeightVec &sample_weight, double impurity) {

        // initialize sample index trackers
        arma::uvec samples;
        double weighted_n_samples;
        init_sample_index(sample_weight, samples, weighted_n_samples);

        // initialize splitting criterion
        MeanSquaredError criterion(
            y, sample_weight, samples, weighted_n_samples);
        criterion.reset_node(0, samples.n_rows);

        // record best impurity improvement of each feature
        arma::vec importances(X.n_cols);

        // loop over all features and determine best impurity improvement
        uint p, best_p;
        double current_proxy_improvement, best_proxy_improvement;
        for (int i = 0; i < X.n_cols; ++i) {
            // keep track of impurity improvements
            current_proxy_improvement =
                -std::numeric_limits<double>::infinity();
            best_proxy_improvement =
                -std::numeric_limits<double>::infinity();

            // start by sorting samples along feature i
            //arma::uvec sample_order = arma::stable_sort_index(X.col(i));
            arma::uvec sample_order = arma::stable_sort_index(X.col(i));
            samples.rows(0, samples.n_rows - 1) = sample_order;

            // Evaluate all splits
            criterion.reset();
            p = 0;
            best_p = 0;
            while (p < samples.n_rows) {
                p += 1;

                // Reject if min_samples_leaf_ is not garunteed
                if ( (p < min_samples_leaf_) ||
                        ((samples.n_rows - p) < min_samples_leaf_) ) {
                    continue;
                }


                criterion.update(p);

                // Reject if min_samples_leaf is not satisfied
                if ((criterion.weighted_n_left() < min_weight_leaf_) ||
                        (criterion.weighted_n_right() < min_weight_leaf_)) {
                    continue;
                }

                current_proxy_improvement =
                    criterion.proxy_impurity_improvement();
                //current_proxy_improvement = criterion.impurity_improvement(impurity);
                if (current_proxy_improvement > best_proxy_improvement) {
                    best_proxy_improvement = current_proxy_improvement;
                    best_p = p;
                }
            }

            // save impurity at best split point
            criterion.reset();
            criterion.update(best_p);
            importances(i) = criterion.impurity_improvement(impurity);
        }

        // NOTE: another option is to set a cut-off of the normalize scores
        //       importances / sum(importances). For example 0.75 * median
        //       or 0.75 * mean.
        arma::uvec feature_ids = arma::stable_sort_index(importances, 1);
        return feature_ids.rows(0, max_features_ - 1);

    }
} // namespace drforest
