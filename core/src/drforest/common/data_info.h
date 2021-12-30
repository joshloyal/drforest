#pragma once


namespace drforest {

    struct FeatureTypes {
        arma::uvec numeric_features;
        arma::uvec categorical_features;
    };

    class FeatureInfo {
    public:
        FeatureInfo(arma::uvec numeric_features, arma::uvec categorical_features);

        FeatureTypes compute_types(arma::uvec feature_ids);

        FeatureTypes get_types() { return feat_types_; }

    private:
        FeatureTypes feat_types_;
    };
}
