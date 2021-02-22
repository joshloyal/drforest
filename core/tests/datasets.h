#pragma once

#include <utility>

#include "drforest.h"

namespace drforest {
namespace testing {

// Loads the athletes demo dataset fromt the DR package
std::pair<arma::mat, arma::vec> load_athletes();

// Loads the directions learned on the athletes dataset by the DR package
arma::mat load_athletes_directions();

// Loads the quadratic demo dataset
std::pair<arma::mat, arma::vec> load_quadratic();

// Loads the directions learned on the quadratic dataset by the DR package
arma::mat load_quadratic_directions();

// Creates a dataset with y = 1 for X * beta > threshold, y = -1 otherwise.
std::pair<arma::mat, arma::vec> make_simple_split();

std::pair<arma::mat, arma::vec> make_checkerboard();


}  // namespace testing
}  // namespace drforest
