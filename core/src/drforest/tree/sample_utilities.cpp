#include "drforest.h"

namespace drforest {

    void init_sample_index(const WeightVec &sample_weight,
                           arma::uvec &samples,
                           double &weighted_n_samples) {
        size_t num_samples = sample_weight.n_rows;

        weighted_n_samples = 0.0;
        std::vector<uint> samples_vec;
        samples_vec.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            // only work with positively weighted samples
            if (sample_weight(i) != 0.0) {
                samples_vec.push_back(i);
                weighted_n_samples += sample_weight(i);
            }
        }
        samples = arma::conv_to<arma::uvec>::from(samples_vec);
    }

} // namespace drforest
