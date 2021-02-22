#pragma once

#include "drforest.h"
#include "catch.hpp"

namespace drforest {
namespace testing {


// Equates two floating point numbers as long as they are within `epsilon`.
inline void almost_equal(const double x, const double y,
                         const double epsilon=1.0e-10) {
    REQUIRE(std::abs(x - y) < epsilon);
}


// Checks that each element of v1 and v2 are equal
template<typename T>
void assert_vector_equal(const arma::Col<T> &vec1, const arma::Col<T> &vec2) {
    REQUIRE(vec1.n_elem == vec2.n_elem);

    for(int i = 0; i < vec1.n_elem; ++i) {
        REQUIRE(vec1(i) == vec2(i));
    }
}


// Checks that each element of v1 and v2 are within `epsilon`.
template<typename T>
void assert_allclose(const arma::Col<T> &vec1, const arma::Col<T> &vec2,
                     const double rtol=1.0e-5, const double atol=1.0e-8) {
    REQUIRE(vec1.n_elem == vec1.n_elem);

    REQUIRE( arma::approx_equal(vec1, vec2, "both", atol, rtol) );
}


// Checks that two matrices have the same dimensions
template<typename T>
void assert_dim_equal(const arma::Mat<T> &mat1, const arma::Mat<T> &mat2) {
    REQUIRE(mat1.n_cols == mat2.n_cols);
    REQUIRE(mat1.n_rows == mat2.n_rows);
}


// Checks that each element of mat1 and mat2 are within `epsilon`.
template<typename T>
void assert_matrix_allclose(const arma::Mat<T> &mat1, const arma::Mat<T> &mat2,
                            const double rtol=1.0e-5,
                            const double atol=1.0e-8) {
    assert_dim_equal(mat1, mat2);

    REQUIRE( arma::approx_equal(mat1, mat2, "both", atol, rtol) );
}

// Checks that each element of mat1 and mat2 are equal
template<typename T>
void assert_matrix_equal(const arma::Mat<T> &mat1, const arma::Mat<T> &mat2) {
    assert_dim_equal(mat1, mat2);
    for (int i = 0; i < mat1.n_cols; ++i) {
        arma::Col<T> vec1 = mat1.col(i);
        arma::Col<T> vec2 = mat2.col(i);
        assert_vector_equal(vec1, vec2);
    }
}

}  // namespace testing
}  // namespace drforest
