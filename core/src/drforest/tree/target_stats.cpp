#include <vector>
#include <utility>

#include "drforest.h"

namespace drforest {

    TargetSplitter::TargetSplitter(const TargetVec &y,
                                   const WeightVec &sample_weight) :
            y_(y),
            sample_weight_(sample_weight) {
        auto [y_stats, indices] = determine_target_stats(y, sample_weight);
        y_stats_ = std::move(y_stats);
        indices_ = std::move(indices);
    }

    // TODO: Clean me up!
    TargetStats TargetSplitter::split_y(size_t start, size_t end) {
        // find first unique y index with a non-zero sample
        int cursor = start;
        int first_index = -1;
        while(first_index < 0) {
            first_index = indices_(cursor);
            cursor++;
        }

        // find last unique y index with a non-zero sample
        cursor = end - 1;
        int last_index = -1;
        while(last_index < 0) {
            last_index = indices_(cursor);
            cursor--;
        }

        // adjust the counts
        arma::uvec counts = y_stats_.counts.rows(first_index, last_index);

        // figure out how many y_values need to be removed from the start of
        // count
        cursor = start - 1;
        int sample_index;
        uint remove_counts = 0;
        if (cursor > 0) {
            sample_index = indices_(cursor);
            while( (sample_index == first_index) || (sample_index < 0) ) {
                remove_counts += sample_weight_(cursor);
                cursor--;
                if(cursor < 0) {
                    break;
                }
                sample_index = indices_(cursor);
            }
            counts(0) -= remove_counts;
        }

        // do the same for the end of the array
        cursor = end;
        remove_counts = 0;
        if (cursor < indices_.n_rows) {
            sample_index = indices_(cursor);
            while( (sample_index == last_index) || (sample_index < 0) ) {
                remove_counts += sample_weight_(cursor);
                cursor++;
                if (cursor >= indices_.n_rows) {
                    break;
                }
                sample_index = indices_(cursor);
            }
            counts(counts.n_rows - 1) -= remove_counts;
        }

        return {
                (uint) arma::sum(counts),
                y_stats_.unique_y.rows(first_index, last_index),
                counts,
        };
    }

    std::pair<TargetStats, arma::ivec>
    determine_target_stats(const TargetVec &y,
                           const WeightVec &sample_weight) {
        std::vector<double> unique_y;
        std::vector<uint> counts;
        std::vector<int> indices(y.n_elem, -1);

        bool found_first_nonzero = false;
        double previous_y;
        uint unique_y_index = 0;
        for(int i = 0; i < sample_weight.n_elem; ++i) {
            if (sample_weight(i) != 0) {
                if(!found_first_nonzero) {
                    unique_y.push_back(y(i));
                    counts.push_back(sample_weight(i));
                    found_first_nonzero = true;
                } else if (y(i) != previous_y) {  // encounter new y value
                    unique_y_index += 1;
                    unique_y.push_back(y(i));
                    counts.push_back(sample_weight(i));
                } else {  // same y value. just update counter
                    counts[unique_y_index] += sample_weight(i);
                }
                indices.at(i) = unique_y_index;
                previous_y = y(i);
            }
        }

        TargetStats y_stats = {
            (uint)y.n_elem,
            arma::conv_to<arma::vec>::from(unique_y),
            arma::conv_to<arma::uvec>::from(counts)
        };
        return { y_stats, arma::conv_to<arma::ivec>::from(indices) };
    }

}  // namespace drforest
