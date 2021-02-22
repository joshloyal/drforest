#include <algorithm>
#include <vector>
#include <utility>

#include "drforest.h"

namespace drforest {

// Groups each unique y value into a seperate slice
// The y values in each slice are homogeneous
std::pair<arma::uvec, arma::uvec>
make_homogeneous_slices_unsorted(const TargetVec &y,
                                 const arma::vec &unique_y) {

    arma::uvec slice_borders = arma::zeros<arma::uvec>(unique_y.n_elem + 1);
    arma::uvec slice_indices(y.n_elem);
    for (int i = 0; i < unique_y.n_elem; ++i) {
        arma::uvec indices = arma::find(y == unique_y[i]);
        slice_indices.elem(indices).fill(i);
        slice_borders(i+1) = slice_borders(i) + indices.n_elem;
    }

    return { slice_indices, slice_borders };
}

arma::uvec rebalance_slices(arma::uvec &slice_borders) {
    // post process borders to merge slices with less than two counts
    // start to mid-point merges to the left, while mid-point and
    // beyond merges to the right
    //
    // TODO: This may have an effect on leaf nodes. So test how
    // this effects the trees
    uint slice_count = 0;
    uint border_index = 0;
    std::vector<uint> merged_borders{0};

    // determine midpoint
    uint mid_point;
    if(slice_borders.n_elem % 2) {
        mid_point = (slice_borders.n_elem - 1) / 2;
    } else {
        mid_point = slice_borders.n_elem / 2 - 1;
    }

    // merge slices
    for (int i = 0; i < slice_borders.n_elem - 1; ++i) {
        slice_count += slice_borders(i+1) - slice_borders(i);
        if (slice_count < 2 && (i > 0) && (i <= mid_point)) { // merge left
            merged_borders.at(border_index) += slice_count;
            slice_count = 0;
        } else if (slice_count > 1) { // merge right
            merged_borders.push_back(
                slice_count + merged_borders.at(border_index));
            border_index += 1;
            slice_count = 0;
        } else if (i == slice_borders.n_elem - 2) {
            // last index may also merge left if it only has one count
            merged_borders.at(border_index) += slice_count;
        }
    }

    return arma::conv_to<arma::uvec>::from(merged_borders);
}

// Groups each unique y value into a seperate slice
// The y values in each slice are homogeneous
SliceStats make_homogeneous_slices_weighted(const TargetStats &y_stats,
                                            const WeightVec &sample_weight) {

    // count borders of the slices
    arma::uvec slice_borders = arma::join_cols(arma::uvec{0},
                                               arma::cumsum(y_stats.counts));
    slice_borders = rebalance_slices(slice_borders);

    // Determines counts in each slice by taking the diff of slice_borders
    arma::uvec slice_counts(slice_borders.n_elem - 1);
    for (int i = 0; i < slice_borders.n_elem - 1; ++i) {
        slice_counts(i) = slice_borders(i+1) - slice_borders(i);
    }

    // determine index -> slice map while taking into account sample weights.
    uint slice_count = 0;
    uint border_index = 0;
    uint slice_id = 0;
    arma::uvec slice_indices(sample_weight.n_elem);
    for (int i = 0; i < sample_weight.n_elem; ++i) {
        // check if we've moved onto the next slice as long as we have not
        // reached the last slice
        if (slice_count >= slice_borders(border_index + 1) &&
                slice_id != slice_borders.n_elem - 2) {
            slice_id += 1;
            border_index += 1;
        }
        slice_indices(i) = slice_id;
        slice_count += sample_weight(i);
    }

    return { slice_indices, slice_borders, slice_counts };
}


// Determines borders in order(y) seperating the slices
// Example: num_slices = 2
//          y = [1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 1.0]
//          slice_borders = [0, 4, 7]
// TODO: The last slice is biased to always contain less samples.
//       Some times substantially less...
arma::uvec get_slice_borders(const uint num_samples,
                             const arma::uvec &counts,
                             const uint num_slices) {
    // Attempts to put this many observations in each slice.
    // This is not always possible since we need to group common y values
    // together.
    // NOTE: This should be ceil, but this package is attempting to
    //       replicate the slices used by R's DR package which uses floor.
    uint num_obs = std::floor(num_samples / num_slices);


    int n_samples_seen = 0;
    std::vector<uint> slice_borders{0};
    arma::uvec cumsum_y = arma::cumsum(counts);
    while (n_samples_seen < num_samples - 2) {
        arma::uvec start_idx = arma::find(
            cumsum_y >= n_samples_seen + num_obs, 1, "first");
        uint slice_start;
        if (start_idx.n_elem == 0) {
            slice_start = cumsum_y.n_elem - 1;
        } else {
            slice_start = start_idx(0);
        }
        n_samples_seen = cumsum_y(slice_start);
        slice_borders.push_back(n_samples_seen);
    }

    // dump any left-over samples into the last slice
    slice_borders[slice_borders.size() - 1] = num_samples;

    return arma::conv_to<arma::uvec>::from(slice_borders);
}


// Groups data into slices ordered by the target vector y.
//
// In this case slices may contain different y values. However the same y value
// will not be split across two slices.
SliceStats make_heterogeneous_slices_unsorted(const TargetVec &y,
                                              const arma::uvec &counts,
                                              const uint num_slices) {
    arma::uvec slice_borders = get_slice_borders(y.n_elem, counts, num_slices);

    // create the slice indicator from order(y)
    // Example: indices of slice i are
    //  order(y)[slice_borders[i]:slice_borders[i+1]-1]
    arma::uvec y_order = arma::stable_sort_index(y);
    arma::uvec slice_indices(y.n_elem);
    arma::uvec slice_counts(slice_borders.n_elem - 1);
    for (int i = 0; i < slice_borders.n_elem - 1; ++i) {
        arma::uvec indices = y_order.subvec(
            slice_borders(i), slice_borders(i+1) - 1);
        slice_indices.elem(indices).fill(i);

        // Determines counts in each slice by taking the diff of slice_borders
        slice_counts(i) = slice_borders(i+1) - slice_borders(i);
    }

    return { slice_indices, slice_borders, slice_counts };
}

// Groups data into slices ordered by the target vector y.
//
// In this case slices may contain different y values. However the same y value
// will not be split across two slices.
SliceStats make_heterogeneous_slices_weighted(const TargetStats &y_stats,
                                              const WeightVec &sample_weight,
                                              const uint num_slices) {
    uint num_samples = arma::sum(sample_weight);
    arma::uvec slice_borders = get_slice_borders(num_samples,
                                                 y_stats.counts,
                                                 num_slices);

    // Determines counts in each slice by taking the diff of slice_borders
    arma::uvec slice_counts(slice_borders.n_elem - 1);
    for (int i = 0; i < slice_borders.n_elem - 1; ++i) {
        slice_counts(i) = slice_borders(i+1) - slice_borders(i);
    }

    // Assign slice indices to each sample taking into the account the
    // sample weights
    uint sample_counter = 0;
    uint border_index = 0;
    uint slice_id = 0;
    arma::uvec slice_indices(sample_weight.n_elem);
    for (int i = 0; i < sample_weight.n_elem; ++i) {
        // check if we've moved onto the next slice
        if (sample_counter >= slice_borders(border_index + 1) &&
                slice_id < slice_counts.n_elem - 1) {
            slice_id += 1;
            border_index += 1;
        }
        slice_indices(i) = slice_id;
        sample_counter += sample_weight(i);
    }

    return { slice_indices, slice_borders, slice_counts };
}


SliceStats slice_y(const TargetVec &y,
                   const uint num_slices) {
    auto [unique_y, counts] = drforest::math::unique_counts(y);
    uint num_y_values = unique_y.n_elem;

    if (num_y_values == 1) {
        return { arma::zeros<arma::uvec>(y.n_elem), arma::uvec{0, y.n_elem},
                 counts };
    } else if (num_slices >= num_y_values) {
        auto [slices, borders] = make_homogeneous_slices_unsorted(y, unique_y);
        return { slices, borders, counts };
    }
    else {
        return make_heterogeneous_slices_unsorted(y, counts, num_slices);
    }
}


// sample_weight for each y value
SliceStats slice_y_weighted(const TargetStats &y_stats,
                            const WeightVec &sample_weight,
                            const uint num_slices) {
    uint num_y_values = y_stats.unique_y.n_elem;

    if (num_y_values == 1) {
        return { arma::zeros<arma::uvec>(y_stats.num_samples),
                 arma::uvec{0, y_stats.num_samples},
                 y_stats.counts };
    } else if (num_slices >= num_y_values) {
        return make_homogeneous_slices_weighted(y_stats, sample_weight);
    }
    else {
        return make_heterogeneous_slices_weighted(y_stats,
                                                  sample_weight,
                                                  num_slices);
    }
}

} // namespace drforest
