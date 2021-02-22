#pragma once

#include <memory>

namespace drforest {
    // Record of the current split
    struct SplitRecord {
        arma::vec direction;    // SDR direction we are splitting on
        size_t feature;         // Index of feature / sdr direction
        size_t pos;             // Split samples array at this position
                                // samples[start:pos], samples[pos:end]
        double threshold;       // Threshold to split at (Xbeta <= threshold)
        double improvement;     // Impurity improvement given parent node.
        double impurity_left;   // Impurity improvement of the left split.
        double impurity_right;  // Impurity improvement of the right split.

        bool is_leaf;           // boolean indicating whether no split was
                                // found. This indicates a leaf node.
        bool used_sdr;          // Whether the split was performed on an
                                // sdr dimension or a normal feature
    };

    // Abstract Splitter
    class NodeSplitter {
    public:
        virtual ~NodeSplitter() {}

        // Reset the node statistics at a given position.
        virtual double reset_node(size_t start, size_t end) = 0;

        // Heavy lifting to actually determine the best split
        virtual SplitRecord split_node(double impurity) = 0;

        // Returns the impurity of a given split
        virtual double node_impurity() const = 0;

        virtual double node_value() const = 0;

        virtual const size_t get_num_samples() const = 0;

        virtual const arma::uvec& get_node_samples() const = 0;
    };

    class DimensionReductionSplitter : public NodeSplitter {
    public:
        DimensionReductionSplitter(const DataMat &X, const TargetVec &y,
                                   const WeightVec &sample_weight,
                                   const int max_features=-1,
                                   const int min_samples_leaf=2,
                                   const int min_weight_leaf=2,
                                   const int num_slices=10,
                                   const uint seed=123);

        ~DimensionReductionSplitter() {};

        double reset_node(size_t start, size_t end);

        SplitRecord split_node(double impurity);

        double node_impurity() const { return criterion_->node_impurity(); }

        double node_value() const { return criterion_->node_value(); }

        SplitRecord find_best_split(arma::mat &Z, arma::uvec &node_samples,
                                    double impurity);

        const size_t get_num_samples() const { return num_samples_; }

        // helper methods mostly for testing
        const arma::uvec& get_node_samples() const { return samples_; }

    private:
        // references to the full dataset
        const DataMat &X_;
        const TargetVec &y_;
        const WeightVec &sample_weight_;

        // hyper-parameters for splitting algorithm
        int max_features_;
        int min_samples_leaf_;
        int min_weight_leaf_;

        // The samples vector `samples_` is mainted by the Splitter such that
        // the samples contained in the node are sorted by the target y.
        // With this in mind, `split_node` reorganizes the node samples
        // `samples[start:end]` into two subsets `samples[start:pos]`
        // and `samples[pos:end]`.
        arma::uvec samples_;
        size_t num_samples_;          // number of unweighted samples in the
                                      // node
        double weighted_n_samples_;   // number of weighted samples in the node
        size_t start_;                // start position for the current node
        size_t end_;                  // end position for the current node
        size_t pos_;                  // location of the split point

        // Splitting crition used to determine the best split
        std::unique_ptr<Criterion> criterion_;

        // Projection method used to determine split directions
        DimensionReductionProjector projector_;

        // feature sampler
        FeatureSampler sampler_;

        // feature screening
        FeatureScreener screener_;
    };

} // namespace drforest
