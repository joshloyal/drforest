#pragma once

#define ARMA_DONT_PRINT_ERRORS

// Do not mix RcppArmadillo headers with base armadillo headers.
// This will cause segmentations faults in the R wrapper.
#ifdef USE_RCPP_ARMADILLO
    #include <RcppArmadillo.h>
#else
    #include <armadillo>
#endif

// typedefs and macros
#include "drforest/common/backports.h"
#include "drforest/common/enums.h"
#include "drforest/common/macros.h"
#include "drforest/common/parallel.h"
#include "drforest/common/typedefs.h"

// samplers
#include "drforest/samplers/bootstrap.h"
#include "drforest/samplers/categorical.h"
#include "drforest/samplers/feature_sampler.h"
#include "drforest/samplers/permute.h"
#include "drforest/samplers/random_states.h"

// trees
#include "drforest/tree/target_stats.h"
#include "drforest/tree/tree.h"
#include "drforest/tree/criterion.h"
#include "drforest/tree/projector.h"
#include "drforest/tree/sample_utilities.h"
#include "drforest/tree/screener.h"
#include "drforest/tree/splitter.h"
#include "drforest/tree/builder.h"

// forest
#include "drforest/forest/forest.h"
#include "drforest/forest/trainer.h"

// sufficient dimension reduction
#include "drforest/dimension_reduction/determine_slices.h"
#include "drforest/dimension_reduction/sir.h"
#include "drforest/dimension_reduction/save.h"

// linear algebra and vector processing
#include "drforest/math/vector_utilities.h"
#include "drforest/math/scaling.h"

// probability distributions
#include "drforest/math/distributions.h"
