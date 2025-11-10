#ifndef GTEST_COMPARE_VEC_H
#define GTEST_COMPARE_VEC_H

#include <gsl/gsl_vector.h>
#include <gtest/gtest.h>

#include <cmath>

inline ::testing::AssertionResult assert_gsl_vectors_close(
    const gsl_vector* expected, const gsl_vector* actual, double rel_tol = 1e-6,
    double abs_tol = 1e-6) {
  if (expected->size != actual->size) {
    return ::testing::AssertionFailure()
           << "Vector sizes differ: expected size " << expected->size
           << " vs actual size " << actual->size;
  }

  for (size_t i = 0; i < expected->size; ++i) {
    const double a = gsl_vector_get(expected, i);
    const double b = gsl_vector_get(actual, i);
    const double diff = std::abs(a - b);
    const double max_ab = std::max(std::abs(a), std::abs(b));
    const double allowed = std::max(rel_tol * max_ab, abs_tol);
    if (diff > allowed) {
      return ::testing::AssertionFailure()
             << "Vectors differ at index " << i << ": expected " << a
             << ", actual " << b << ", diff " << diff << " > tolerance "
             << allowed;
    }
  }

  return ::testing::AssertionSuccess();
}

#endif  // GTEST_COMPARE_VEC_H