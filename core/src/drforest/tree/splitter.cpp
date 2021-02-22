#include <algorithm>
#include <memory>

#include "drforest.h"

namespace drforest {

    DimensionReductionSplitter::DimensionReductionSplitter(
            const DataMat &X,
            const TargetVec &y,
            const WeightVec &sample_weight,
            const int max_features,
            const int min_samples_leaf,
            const int min_weight_leaf,
            const int num_slices,
            const uint seed) :
            X_(X),
            y_(y),
            sample_weight_(sample_weight),
            max_features_(max_features),
            min_samples_leaf_(min_samples_leaf),
            min_weight_leaf_(min_weight_leaf),
            projector_(num_slices),
            start_(0),
            sampler_(X.n_cols, seed),
            screener_(max_features, min_samples_leaf, min_weight_leaf) {
        // make sure max features does not exceed the number of features
        if (max_features_ < 0 || max_features_ > X.n_cols) {
            max_features_ = X.n_cols;
        }

        init_sample_index(sample_weight, samples_, weighted_n_samples_);
        num_samples_ = samples_.n_rows;
        end_ = num_samples_;

        criterion_ = drforest::make_unique<MeanSquaredError>(
            y_, sample_weight_, samples_, weighted_n_samples_);
    }

    double DimensionReductionSplitter::reset_node(size_t start, size_t end) {
        start_ = start;
        end_ = end;

        // reset criterion statistics as well
        criterion_->reset_node(start, end);

        return criterion_->weighted_n_node_samples();
    }

    SplitRecord DimensionReductionSplitter::split_node(double impurity) {
        // extract samples in the current node
        arma::uvec node_samples = samples_.rows(start_, end_ - 1);
        DataMat  node_X = X_.rows(node_samples);
        WeightVec node_sample_weight = sample_weight_.rows(node_samples);
        TargetVec node_y = y_.rows(node_samples);

        // sub-sample features going into SIR / SAVE
        arma::uvec feature_ids;
        if (max_features_ < X_.n_cols) {
            if (impurity <= 0) {
                // zero variance node so random sample features
                feature_ids = sampler_.draw(max_features_);
            } else {
                feature_ids = screener_.screen_features(
                    node_X, node_y, node_sample_weight, impurity);
            }
            node_X = node_X.cols(feature_ids);
        }

        // determine directions for splitting
        auto [Z, directions] =  projector_.get_directions(node_X,
                                                          node_y,
                                                          node_sample_weight);

        // CART search over best direction
        SplitRecord split = find_best_split(Z, node_samples, impurity);

        // pad directions with zeros if necessary
        arma::rowvec new_direction(X_.n_cols, arma::fill::zeros);

        if (max_features_ < X_.n_cols) {
            new_direction.elem(feature_ids) = directions.row(split.feature);
        } else {
            new_direction = directions.row(split.feature);
        }
        split.direction = new_direction.t();

        return split;
    }

    SplitRecord DimensionReductionSplitter::find_best_split(
            arma::mat &Z,
            arma::uvec &node_samples,
            double impurity) {

        // initialize records to track the best split
        SplitRecord best, current;
        best.is_leaf = true;
        current.is_leaf = false;
        best.feature = 0;
        double current_proxy_improvement =
            -std::numeric_limits<double>::infinity();
        double best_proxy_improvement =
            -std::numeric_limits<double>::infinity();

        arma::vec Z_col;
        uint p;
        uint p_feature;
        for (int i = 0; i < Z.n_cols; ++i) {
            // record the current feature
            current.feature = i;

            // start by sorting samples along the SDR direction `Z`.
            Z_col = Z.col(i);
            arma::uvec sample_order = arma::stable_sort_index(Z_col);
            Z_col = Z_col.rows(sample_order);
            samples_.rows(start_, end_ - 1) = node_samples.rows(sample_order);

            // Evaluate all splits
            criterion_->reset();
            p = start_;
            p_feature = 0;  // feature array is indexed 0...(start_ - end_ - 1)
            while (p < end_) {
                p += 1;
                p_feature += 1;
                if (p < end_) {
                    current.pos = p;

                    // Reject if min_samples_leaf_ is not garunteed
                    if ( ((current.pos - start_) < min_samples_leaf_) ||
                            ((end_ - current.pos) < min_samples_leaf_) ) {
                        continue;
                    }

                    criterion_->update(current.pos);

                    // Reject if min_samples_leaf is not satisfied
                    if ((criterion_->weighted_n_left() < min_weight_leaf_) ||
                            (criterion_->weighted_n_right() < min_weight_leaf_)) {
                        continue;
                    }

                    current_proxy_improvement =
                        criterion_->proxy_impurity_improvement();
                    if (current_proxy_improvement > best_proxy_improvement) {
                        best_proxy_improvement = current_proxy_improvement;

                        // mid-point split is more stable. why?
                        current.threshold =
                            Z_col(p_feature - 1) / 2 + Z_col(p_feature) / 2;
                        best = current;
                    }
                }
            }
        }

        // Save the best split
        if (best.pos < end_) { // if best.pos == end_ then this is a leaf
            // sort samples_ in terms of the best feature
            Z_col = Z.col(best.feature);
            arma::uvec sample_order = arma::stable_sort_index(Z_col);
            samples_.rows(start_, end_ - 1) = node_samples.rows(sample_order);

            // reorganize samples_ so that
            // sample_[start:best.pos] + sample_[best.pos:end] are
            // ordered in y

            // sort beginning of array
            std::sort(samples_.begin() + start_,
                      samples_.begin() + best.pos);

            // sort end of array
            std::sort(samples_.begin() + best.pos,
                      samples_.begin() + end_);

            // update and store impurity statistics
            criterion_->reset();
            criterion_->update(best.pos);

            best.improvement = criterion_->impurity_improvement(impurity);
            criterion_->children_impurity(best.impurity_left,
                                          best.impurity_right);
        }

        return best;
    }

} // namespace drforest
