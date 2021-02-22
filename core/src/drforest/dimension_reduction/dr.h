#pragma once

#include <utility>

namespace drforest {

std::pair<arma::vec, arma::mat> fit_directional_regression(
                                    const DataMat &X,
                                    const TargetStats &y_stats,
                                    const WeightVec &sample_weight,
                                    const int num_slices);

}  // namespace drforest
