#include "gm_to_dirac_short_standard_normal_deviation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "capture_time.h"
#include "gm_to_dirac_short.h"
#include "gsl_minimizer.h"
#include "gsl_utils_view_helper.h"
#include "math_util_defs.h"

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, T* x, const T* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, T(1));
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(covDiag.data(), L, N, bMax, x, wX,
                                       result, options);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                       size_t bMax, T* x, const T* wX) {
  std::vector<T> covDiag(N, T(1));
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq(covDiag.data(), distance, L,
                                                   N, bMax, x, wX);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq_derivative(T* gradient, size_t L,
                                                  size_t N, size_t bMax, T* x,
                                                  const T* wX) {
  std::vector<T> covDiag(N, T(1));
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq_derivative(
      covDiag.data(), gradient, L, N, bMax, x, wX);
}

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, GSLVectorType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(covDiagView, L, N, bMax, x, wX, result,
                                       options);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                       size_t bMax, GSLVectorType* x,
                                       const GSLVectorType* wX) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq(covDiagView, distance, L, N,
                                                   bMax, x, wX);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq_derivative(GSLVectorType* gradient,
                                                  size_t L, size_t N,
                                                  size_t bMax, GSLVectorType* x,
                                                  const GSLVectorType* wX) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq_derivative(
      covDiagView, gradient, L, N, bMax, x, wX);
}

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, GSLMatrixType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(covDiagView, L, N, bMax, x, wX, result,
                                       options);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                       size_t bMax, GSLMatrixType* x,
                                       const GSLVectorType* wX) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq(covDiagView, distance, L, N,
                                                   bMax, x, wX);
}

template <typename T>
void gm_to_dirac_short_standard_normal_deviation<
    T>::modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                  size_t L, size_t N,
                                                  size_t bMax, GSLMatrixType* x,
                                                  const GSLVectorType* wX) {
  std::vector<T> covDiag(N, T(1));
  GSLVectorView<T> covDiagView(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  gmToDiracInstance.modified_van_mises_distance_sq_derivative(
      covDiagView, gradient, L, N, bMax, x, wX);
}

template class gm_to_dirac_short_standard_normal_deviation<double>;
template class gm_to_dirac_short_standard_normal_deviation<float>;