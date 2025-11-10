#include "gm_to_dirac_short_standard_normal_deviation.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_psi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>

#include "capture_time.h"
#include "gm_to_dirac_short.h"
#include "gsl_minimizer.h"
#include "math_util_defs.h"

// #define USE_CACHE_MANAGER

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, T* x, const T* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, 1.0);
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(covDiag.data(), L, N, bMax, x, wX,
                                       result, options);
}

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, GSLVectorType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, 1.0);
  GSLVectorViewType covDiagView =
      GSLTemplateTypeAlias<T>::vector_view_from_array(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(&(covDiagView.vector), L, N, bMax, x, wX,
                                       result, options);
}

template <typename T>
bool gm_to_dirac_short_standard_normal_deviation<T>::approximate(
    size_t L, size_t N, size_t bMax, GSLMatrixType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  std::vector<T> covDiag(N, 1.0);
  GSLVectorViewType covDiagView =
      GSLTemplateTypeAlias<T>::vector_view_from_array(covDiag.data(), N);
  gm_to_dirac_short<T> gmToDiracInstance;
  return gmToDiracInstance.approximate(&(covDiagView.vector), L, N, bMax, x, wX,
                                       result, options);
}

template class gm_to_dirac_short_standard_normal_deviation<double>;
template class gm_to_dirac_short_standard_normal_deviation<float>;