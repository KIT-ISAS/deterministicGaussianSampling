#ifndef GSL_UTILS_ALLOCATION_H
#define GSL_UTILS_ALLOCATION_H

#include <gsl/gsl_vector.h>

#include <vector>

/******************************************************************************/
/*********************************** double ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(double) from a std::vector<double>.
 *
 * @param v The std::vector<double> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector* create_gsl_vector(const std::vector<double>& v) {
  gsl_vector* g = gsl_vector_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ float ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(float) from a std::vector<float>.
 *
 * @param v The std::vector<float> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_float* create_gsl_vector(const std::vector<float>& v) {
  gsl_vector_float* g = gsl_vector_float_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_float_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/********************************* long double ********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(long double) from a std::vector<long double>.
 *
 * @param v The std::vector<long double> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_long_double* create_gsl_vector(
    const std::vector<long double>& v) {
  gsl_vector_long_double* g = gsl_vector_long_double_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_long_double_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ char ************************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(char) from a std::vector<char>.
 *
 * @param v The std::vector<char> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_char* create_gsl_vector(const std::vector<char>& v) {
  gsl_vector_char* g = gsl_vector_char_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_char_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ int *************************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(int) from a std::vector<int>.
 *
 * @param v The std::vector<int> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_int* create_gsl_vector(const std::vector<int>& v) {
  gsl_vector_int* g = gsl_vector_int_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_int_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ long ************************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(long) from a std::vector<long>.
 *
 * @param v The std::vector<long> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_long* create_gsl_vector(const std::vector<long>& v) {
  gsl_vector_long* g = gsl_vector_long_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_long_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ short ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(short) from a std::vector<short>.
 *
 * @param v The std::vector<short> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_short* create_gsl_vector(const std::vector<short>& v) {
  gsl_vector_short* g = gsl_vector_short_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_short_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ uchar ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(uchar) from a std::vector<unsigned char>.
 *
 * @param v The std::vector<unsigned char> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_uchar* create_gsl_vector(
    const std::vector<unsigned char>& v) {
  gsl_vector_uchar* g = gsl_vector_uchar_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_uchar_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ uint ************************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(uint) from a std::vector<unsigned int>.
 *
 * @param v The std::vector<unsigned int> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_uint* create_gsl_vector(const std::vector<unsigned int>& v) {
  gsl_vector_uint* g = gsl_vector_uint_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_uint_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/************************************ ulong ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(ulong) from a std::vector<unsigned long>.
 *
 * @param v The std::vector<unsigned long> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_ulong* create_gsl_vector(
    const std::vector<unsigned long>& v) {
  gsl_vector_ulong* g = gsl_vector_ulong_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_ulong_set(g, i, v[i]);
  return g;
}

/******************************************************************************/
/*********************************** ushort ***********************************/
/******************************************************************************/
/**
 * @brief Create a GSL vector(ushort) from a std::vector<unsigned
 * short>.
 *
 * @param v The std::vector<unsigned short> to convert.
 * @return A pointer to the newly created GSL vector, or nullptr on failure.
 */
[[maybe_unused]]
static gsl_vector_ushort* create_gsl_vector(
    const std::vector<unsigned short>& v) {
  gsl_vector_ushort* g = gsl_vector_ushort_alloc(v.size());
  if (g == nullptr) return nullptr;
  for (size_t i = 0; i < v.size(); ++i) gsl_vector_ushort_set(g, i, v[i]);
  return g;
}

#endif  // GSL_UTILS_ALLOCATION_H
