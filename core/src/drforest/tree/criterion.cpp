#include <cmath>

#include "drforest.h"

namespace drforest {

    MeanSquaredError::MeanSquaredError(const TargetVec &y,
                                       const WeightVec &sample_weight,
                                       const arma::uvec &samples,
                                       double weighted_n_samples)
            : y_(y),
            sample_weight_(sample_weight),
            samples_(samples),
            weighted_n_samples_(weighted_n_samples),
            sum_total_(0.), sq_sum_total_(0.), weighted_n_node_samples_(0.) {}

    void MeanSquaredError::reset_node(size_t start, size_t end) {
        // set new position in samples
        start_ = start;
        end_ = end;

        // reset the statistics
        sum_total_ = 0.;
        sq_sum_total_ = 0.;
        weighted_n_node_samples_ = 0.;

        // start by calculating a few invariants
        n_node_samples_ = end_ - start_;

        uint i;
        double w;
        double y_i;
        double w_y_i;
        for (int p = start_; p < end_; ++p) {
            i = samples_(p);
            w = sample_weight_(i);
            y_i = y_(i);
            w_y_i = w * y_i;
            sum_total_ += w_y_i;
            sq_sum_total_ += w_y_i * y_i;
            weighted_n_node_samples_ += w;
        }

        // set pos_ = start_
        reset();
    }

    void MeanSquaredError::reset() {
        // Sets pos_ = start_, so every sample is in the right node
        sum_left_ = 0.0;
        weighted_n_left_ = 0.0;

        sum_right_ = sum_total_;
        weighted_n_right_ = weighted_n_node_samples_;

        pos_ = start_;
    }

    void MeanSquaredError::reverse_reset() {
        sum_left_ = sum_total_;
        weighted_n_left_ = weighted_n_node_samples_;

        sum_right_ = 0.0;
        weighted_n_right_ = 0.0;

        pos_ = end_;
    }

    void MeanSquaredError::update(size_t new_pos) {
        // Trick from sklearn.
        //
        // Given that
        //
        //  sum_left[x] + sum_right[x] = sum_total[x]
        //
        // and sum_total is known, we are going to update
        // sum_left from the direction that requires the least amount
        // of computations, i.e. from pos to new_pos or from end to new_pos
        uint i;
        double w;
        double y_i;
        if ( (new_pos - pos_) <= (end_ - new_pos) ) {
            for (int p = pos_; p < new_pos; ++p) {
                i = samples_(p);
                w = sample_weight_(i);
                y_i = y_(i);
                sum_left_ += w * y_i;
                weighted_n_left_ += w;
            }
        } else {
            reverse_reset();

            for (int p = end_ - 1; p >= new_pos; --p) {
                i = samples_(p);
                w = sample_weight_(i);
                y_i = y_(i);
                sum_left_ -= w * y_i;
                weighted_n_left_ -= w;
            }
        }

        // calculate the right statistics using the sum_total invariant
        weighted_n_right_ = weighted_n_node_samples_ - weighted_n_left_;
        sum_right_ = sum_total_ - sum_left_;

        pos_ = new_pos;
    }

    double MeanSquaredError::node_impurity() {
        double impurity = sq_sum_total_ / weighted_n_node_samples_;
        impurity -= std::pow(sum_total_ / weighted_n_node_samples_, 2);

        return impurity;
    }

    double MeanSquaredError::impurity_improvement(double impurity) {
        double impurity_left;
        double impurity_right;
        children_impurity(impurity_left, impurity_right);

        return ((weighted_n_node_samples_ / weighted_n_samples_) *
                (impurity - (weighted_n_right_ /
                             weighted_n_node_samples_ * impurity_right)
                          - (weighted_n_left_ /
                             weighted_n_node_samples_ * impurity_left)));

    }

    double MeanSquaredError::proxy_impurity_improvement() {
        double proxy_impurity_left = sum_left_ * sum_left_;
        double proxy_impurity_right = sum_right_ * sum_right_;

        return (proxy_impurity_left / weighted_n_left_ +
                proxy_impurity_right / weighted_n_right_);
    }

    void MeanSquaredError::children_impurity(double &impurity_left,
                                             double &impurity_right) {
        uint i;
        double w;
        double y_i;
        double sq_sum_left = 0.0;
        double sq_sum_right;
        for (int p = start_; p < pos_; ++p) {
            i = samples_(p);
            w = sample_weight_(i);
            y_i = y_(i);
            sq_sum_left += w * y_i * y_i;
        }

        sq_sum_right = sq_sum_total_ - sq_sum_left;

        impurity_left = sq_sum_left / weighted_n_left_;
        impurity_right = sq_sum_right / weighted_n_right_;

        impurity_left -= std::pow(sum_left_ / weighted_n_left_, 2);
        impurity_right -= std::pow(sum_right_ / weighted_n_right_, 2);
    }

}  // namespace drforest
