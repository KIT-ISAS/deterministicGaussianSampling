#ifndef GSL_UTILS_COMPARE_H
#define GSL_UTILS_COMPARE_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <cmath>

/******************************************************************************/
/*********************************** double ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(double) for equality within a tolerance.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the vectors are equal within the tolerance, false otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_vector* a, const gsl_vector* b,
                                 double tol = 1e-6) {
  if (a->size != b->size) return false;
  for (size_t i = 0; i < a->size; ++i) {
    if (std::fabs(gsl_vector_get(a, i) - gsl_vector_get(b, i)) > tol)
      return false;
  }
  return true;
}

/**
 * @brief Compare two GSL matricies(double) for equality within a tolerance.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the matricies are equal within the tolerance, false
 * otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_matrix* a, const gsl_matrix* b,
                                 double tol = 1e-6) {
  if (a->size1 != b->size1 || a->size2 != b->size2) return false;
  for (size_t i = 0; i < a->size1; ++i) {
    for (size_t j = 0; j < a->size2; ++j) {
      if (std::fabs(gsl_matrix_get(a, i, j) - gsl_matrix_get(b, i, j)) > tol)
        return false;
    }
  }
  return true;
}

/******************************************************************************/
/************************************ float ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(float) for equality within a tolerance.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the vectors are equal within the tolerance, false otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_vector_float* a,
                                 const gsl_vector_float* b, double tol = 1e-6) {
  if (a->size != b->size) return false;
  for (size_t i = 0; i < a->size; ++i) {
    if (std::fabs(gsl_vector_float_get(a, i) - gsl_vector_float_get(b, i)) >
        (float)tol)
      return false;
  }
  return true;
}

/**
 * @brief Compare two GSL matricies(float) for equality within a tolerance.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the matricies are equal within the tolerance, false
 * otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_matrix_float* a,
                                 const gsl_matrix_float* b, double tol = 1e-6) {
  if (a->size1 != b->size1 || a->size2 != b->size2) return false;
  for (size_t i = 0; i < a->size1; ++i) {
    for (size_t j = 0; j < a->size2; ++j) {
      if (std::fabs(gsl_matrix_float_get(a, i, j) -
                    gsl_matrix_float_get(b, i, j)) > (float)tol)
        return false;
    }
  }
  return true;
}

/******************************************************************************/
/********************************* long double ********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(long double) for equality within a tolerance.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the vectors are equal within the tolerance, false otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_vector_long_double* a,
                                 const gsl_vector_long_double* b,
                                 double tol = 1e-6) {
  if (a->size != b->size) return false;
  for (size_t i = 0; i < a->size; ++i) {
    if (std::fabs(gsl_vector_long_double_get(a, i) -
                  gsl_vector_long_double_get(b, i)) > tol)
      return false;
  }
  return true;
}

/**
 * @brief Compare two GSL matricies(long double) for equality within a
 * tolerance.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @param tol Tolerance for comparison.
 * @return true if the matricies are equal within the tolerance, false
 * otherwise.
 */
[[maybe_unused]]
static bool gsl_utils_compare_eq(const gsl_matrix_long_double* a,
                                 const gsl_matrix_long_double* b,
                                 double tol = 1e-6) {
  if (a->size1 != b->size1 || a->size2 != b->size2) return false;
  for (size_t i = 0; i < a->size1; ++i) {
    for (size_t j = 0; j < a->size2; ++j) {
      if (std::fabs(gsl_matrix_long_double_get(a, i, j) -
                    gsl_matrix_long_double_get(b, i, j)) > tol)
        return false;
    }
  }
  return true;
}

/******************************************************************************/
/************************************ char ************************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(char) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_char* a,
                                        const gsl_vector_char* b) {
  return gsl_vector_char_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(char) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_char* a,
                                        const gsl_matrix_char* b) {
  return gsl_matrix_char_equal(a, b);
}

/******************************************************************************/
/************************************ int *************************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(int) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_int* a,
                                        const gsl_vector_int* b) {
  return gsl_vector_int_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(int) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_int* a,
                                        const gsl_matrix_int* b) {
  return gsl_matrix_int_equal(a, b);
}

/******************************************************************************/
/************************************ long ************************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(long) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_long* a,
                                        const gsl_vector_long* b) {
  return gsl_vector_long_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(long) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_long* a,
                                        const gsl_matrix_long* b) {
  return gsl_matrix_long_equal(a, b);
}

/******************************************************************************/
/************************************ short ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(short) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
inline static bool gsl_utils_compare_eq(const gsl_vector_short* a,
                                        const gsl_vector_short* b) {
  return gsl_vector_short_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(short) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_short* a,
                                        const gsl_matrix_short* b) {
  return gsl_matrix_short_equal(a, b);
}

/******************************************************************************/
/************************************ uchar ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(uchar) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_uchar* a,
                                        const gsl_vector_uchar* b) {
  return gsl_vector_uchar_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(uchar) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_uchar* a,
                                        const gsl_matrix_uchar* b) {
  return gsl_matrix_uchar_equal(a, b);
}

/******************************************************************************/
/************************************ uint ************************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(uint) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_uint* a,
                                        const gsl_vector_uint* b) {
  return gsl_vector_uint_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(uint) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_uint* a,
                                        const gsl_matrix_uint* b) {
  return gsl_matrix_uint_equal(a, b);
}

/******************************************************************************/
/************************************ ulong ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(ulong) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_ulong* a,
                                        const gsl_vector_ulong* b) {
  return gsl_vector_ulong_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(ulong) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_ulong* a,
                                        const gsl_matrix_ulong* b) {
  return gsl_matrix_ulong_equal(a, b);
}

/******************************************************************************/
/*********************************** ushort ***********************************/
/******************************************************************************/
/**
 * @brief Compare two GSL vectors(ushort) for equality.
 *
 * @param a First GSL vector.
 * @param b Second GSL vector.
 * @return true if the vectors are equal, false otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_vector_ushort* a,
                                        const gsl_vector_ushort* b) {
  return gsl_vector_ushort_equal(a, b);
}

/**
 * @brief Compare two GSL matricies(ushort) for equality.
 *
 * @param a First GSL matrix.
 * @param b Second GSL vector.
 * @return true if the matricies are equal, false
 * otherwise.
 */
[[maybe_unused]]
inline static bool gsl_utils_compare_eq(const gsl_matrix_ushort* a,
                                        const gsl_matrix_ushort* b) {
  return gsl_matrix_ushort_equal(a, b);
}

#endif  // GSL_UTILS_COMPARE_H
