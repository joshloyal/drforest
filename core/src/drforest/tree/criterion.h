#pragma once

#include <utility>

namespace drforest {

    class Criterion  {
    public:
        virtual ~Criterion() {}

        virtual void reset_node(size_t start, size_t end) = 0;

        virtual void reset() = 0;

        virtual void update(size_t new_pos) = 0;

        virtual double node_impurity() = 0;

        virtual double impurity_improvement(double impurity) = 0;

        virtual double proxy_impurity_improvement() = 0;

        virtual void children_impurity(double &impurity_left,
                                       double &impurity_right) = 0;

        virtual double node_value() = 0;

        virtual const double& weighted_n_left() const = 0;

        virtual const double& weighted_n_right() const = 0;

        virtual double weighted_n_node_samples() const = 0;
    };

    // The criterion computes the impurity of a node and the reduction
    // of impurity of a split on that node. It also computes the output
    // statistics found at the leaf node, i.e. the mean for MSE.
    //
    // Right now only MSE is implemented which uses the reduction in variance:
    //
    //      var = \sum^i_n (y_i - y_bar) ^ 2
    //          = (\sum^i_n y_i ^ 2) - n_samples * y_bar ^ 2
    //
    // This class is very similar to the Criterion class in scikit-learn
    class MeanSquaredError : public Criterion {
    public:
        MeanSquaredError(const TargetVec &y, const WeightVec &sample_weight,
                         const arma::uvec &samples, double weighted_n_samples);

        ~MeanSquaredError() {};

        // Reset statistics for splitting of a new node
        void reset_node(size_t start, size_t end);

        // reset criterion at pos=start.
        void reset();

        // reset criterion at pos=end.
        void reverse_reset();

        // update left and right statistics by moving samples[pos:new_pos]
        // to the left node aka the split <= Xf[new_pos - 1]
        void update(size_t new_pos);

        double node_impurity();

        double impurity_improvement(double impurity);

        double proxy_impurity_improvement();

        void children_impurity(double &impurity_left, double &impurity_right);

        double node_value() { return sum_total_ / weighted_n_node_samples_; };

        const double& weighted_n_left() const {
            return weighted_n_left_;
        }

        const double& weighted_n_right() const {
            return weighted_n_right_;
        }

        const arma::uvec& get_samples() const {
            return samples_;
        }

        double weighted_n_node_samples() const {
            return weighted_n_node_samples_;
        }

    private:
        const TargetVec &y_;
        const WeightVec &sample_weight_;

        // Used to keep track of samples in node
        const arma::uvec &samples_;
        size_t start_;
        size_t end_;
        size_t pos_;

        // invariante statistics used for calculating reduction in variance
        double sum_total_;
        double sq_sum_total_;
        uint  n_node_samples_;
        double weighted_n_samples_;
        double weighted_n_node_samples_;

        // statistics for the proposed left and right node
        double sum_left_;
        double  weighted_n_left_;
        double sum_right_;
        double  weighted_n_right_;
    };

}
