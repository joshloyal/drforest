#pragma once

namespace drforest {

// Note: Could use an enum class, but a plain enum makes interfacing
// with python / R easier
enum DimensionReductionAlgorithm { SIR, SAVE };

} // namespace drforest
