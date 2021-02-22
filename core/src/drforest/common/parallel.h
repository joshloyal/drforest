#pragma once

#include "pthread.h"

namespace drforest {

    inline int get_num_threads(int num_threads) {
        int available_cores = omp_get_max_threads();
        if (num_threads < 1 || num_threads > available_cores) {
            num_threads = available_cores;
        }

        return num_threads;
    }

} // namespace drforest
