#include "drforest.h"


namespace drforest {

    FeatureInfo::FeatureInfo(arma::uvec numeric_features, arma::uvec categorical_features) :
        feat_types_{numeric_features, categorical_features} {}

    FeatureTypes FeatureInfo::compute_types(arma::uvec feature_ids) {
        std::vector<arma::uword> numeric_features;
        std::vector<arma::uword> categorical_features;
        for (int i = 0; i < feature_ids.n_elem; i++) {
            if (arma::any(feat_types_.numeric_features == feature_ids(i))) {
                numeric_features.push_back(i);
            } else {
                categorical_features.push_back(i);
            }
        }

        return {arma::uvec(numeric_features), arma::uvec(categorical_features)};
    }
}
