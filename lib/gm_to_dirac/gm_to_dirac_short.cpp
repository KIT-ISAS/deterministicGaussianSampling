#include "gm_to_dirac_short.h"

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
#include "gsl_minimizer.h"
#include "gsl_utils_view_helper.h"
#include "gsl_utils_weight_helper.h"
#include "math_util_defs.h"

template <typename T>
bool gm_to_dirac_short<T>::approximate(const T* covDiag, size_t L, size_t N,
                                       size_t bMax, T* x, const T* wX,
                                       GslminimizerResult* result,
                                       const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(covDiag != nullptr);

  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLVectorView<T> vectorViewCovDiag(covDiag, N);
  return approximate(vectorViewCovDiag, L, N, bMax, vectorViewX, vectorViewWX,
                     result, options);
}

template <typename T>
void gm_to_dirac_short<T>::modified_van_mises_distance_sq(const T* covDiag,
                                                          T* distance, size_t L,
                                                          size_t N, size_t bMax,
                                                          T* x, const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLVectorView<T> vectorViewCovDiag(covDiag, N);
  return modified_van_mises_distance_sq(vectorViewCovDiag, distance, L, N, bMax,
                                        vectorViewX, vectorViewWX);
}

template <typename T>
void gm_to_dirac_short<T>::modified_van_mises_distance_sq_derivative(
    const T* covDiag, T* gradient, size_t L, size_t N, size_t bMax, T* x,
    const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLVectorView<T> vectorViewCovDiag(covDiag, N);
  GSLVectorView<T> vectorViewGradient(gradient, L * N);
  return modified_van_mises_distance_sq_derivative(
      vectorViewCovDiag, vectorViewGradient, L, N, bMax, vectorViewX,
      vectorViewWX);
}

template <typename T>
bool gm_to_dirac_short<T>::approximate(const GSLVectorType* covDiag, size_t L,
                                       size_t N, size_t bMax, GSLMatrixType* x,
                                       const GSLVectorType* wX,
                                       GslminimizerResult* result,
                                       const ApproximateOptions& options) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  return approximate(covDiag, L, N, bMax, vectorViewX, wX, result, options);
}

template <typename T>
void gm_to_dirac_short<T>::modified_van_mises_distance_sq(
    const GSLVectorType* covDiag, T* distance, size_t L, size_t N, size_t bMax,
    GSLMatrixType* x, const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  return modified_van_mises_distance_sq(covDiag, distance, L, N, bMax,
                                        vectorViewX, wX);
}

template <typename T>
void gm_to_dirac_short<T>::modified_van_mises_distance_sq_derivative(
    const GSLVectorType* covDiag, GSLMatrixType* gradient, size_t L, size_t N,
    size_t bMax, GSLMatrixType* x, const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewGradient(gradient);
  return modified_van_mises_distance_sq_derivative(covDiag, vectorViewGradient,
                                                   L, N, bMax, vectorViewX, wX);
}

template <typename T>
double gm_to_dirac_short<T>::modified_van_mises_distance_sq(const gsl_vector* x,
                                                            void* params) {
  double d = 0.00;
  combined_distance_metric(x, params, &d, nullptr);
  return d;
}

template <typename T>
void gm_to_dirac_short<T>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* x, void* params, gsl_vector* grad) {
  combined_distance_metric(x, params, nullptr, grad);
}

template <typename T>
void gm_to_dirac_short<T>::combined_distance_metric(const gsl_vector* x,
                                                    void* params, double* f,
                                                    gsl_vector* grad) {
  GMToDiracConstWeightOptimizationParams* optiParams =
      static_cast<GMToDiracConstWeightOptimizationParams*>(params);

  if (f) *f = 0.00;
  if (grad) gsl_vector_set_zero(grad);

  calculateD2(x, optiParams, f, grad);
  calculateD3(x, optiParams, f, grad);
}

template <typename T>
double gm_to_dirac_short<T>::c_b(size_t bMax) {
  const double bMaxSqrd = static_cast<double>(bMax * bMax);
  return std::log(4.00 * bMaxSqrd);
}

template <>
bool gm_to_dirac_short<float>::approximate(const gsl_vector_float* covDiag,
                                           size_t L, size_t N, size_t bMax,
                                           gsl_vector_float* x,
                                           const gsl_vector_float* wX,
                                           GslminimizerResult* result,
                                           const ApproximateOptions& options) {
  assert(x->size == L * N);
  gsl_vector* xDouble = gsl_vector_alloc(x->size);
  gsl_vector* covDiagDouble = gsl_vector_alloc(covDiag->size);
  gsl_vector* wXDouble = nullptr;

  if (wX) {
    gsl_vector* wXDouble = gsl_vector_alloc(L);
    for (size_t i = 0; i < wX->size; ++i) {
      wXDouble->data[i] = static_cast<double>(wX->data[i]);
    }
  }

  if (options.initialX) {
    for (size_t i = 0; i < x->size; ++i) {
      xDouble->data[i] = static_cast<double>(x->data[i]);
    }
  }

  for (size_t i = 0; i < covDiag->size; ++i) {
    covDiagDouble->data[i] = static_cast<double>(covDiag->data[i]);
  }

  gm_to_dirac_short<double> doubleApprox;
  bool success = doubleApprox.approximate(covDiagDouble, L, N, bMax, xDouble,
                                          wXDouble, result, options);

  for (size_t i = 0; i < x->size; ++i) {
    x->data[i] = static_cast<float>(xDouble->data[i]);
  }

  gsl_vector_free(xDouble);
  if (wXDouble) gsl_vector_free(const_cast<gsl_vector*>(wXDouble));

  return success;
}

template <>
bool gm_to_dirac_short<double>::approximate(const gsl_vector* covDiag, size_t L,
                                            size_t N, size_t bMax,
                                            gsl_vector* x, const gsl_vector* wX,
                                            GslminimizerResult* result,
                                            const ApproximateOptions& options) {
  assert(x->size % N == 0);
  assert(x->size == L * N);
  assert(covDiag->size == N);

  if (!options.initialX) {
    gsl_rng_env_setup();
    gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
    for (size_t i = 0; i < L; i++) {
      for (size_t k = 0; k < N; k++) {
        x->data[i * N + k] = gsl_ran_gaussian(r, covDiag->data[k]);
      }
    }
    gsl_rng_free(r);
  }

  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracConstWeightOptimizationParams params(covDiag, wXHelper, N, L, bMax,
                                                c_b(bMax));

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);

  const int status = gslMinimizer.minimize(x, result, options.verbose);

  correctMean(x, params.wX, L, N);

  return status == GSL_SUCCESS;
}

template <>
void gm_to_dirac_short<float>::modified_van_mises_distance_sq(
    const gsl_vector_float* covDiag, float* distance, size_t L, size_t N,
    size_t bMax, gsl_vector_float* x, const gsl_vector_float* wX) {
  double distanceDouble = 0.00;
  GSLVectorView<double> vectorViewCovDiag(covDiag, L * N);
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);
  gm_to_dirac_short<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq(vectorViewCovDiag,
                                              &distanceDouble, L, N, bMax,
                                              vectorViewX, vectorViewWX);
  *distance = static_cast<float>(distanceDouble);
}

template <>
void gm_to_dirac_short<double>::modified_van_mises_distance_sq(
    const gsl_vector* covDiag, double* distance, size_t L, size_t N,
    size_t bMax, gsl_vector* x, const gsl_vector* wX) {
  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracConstWeightOptimizationParams optiParams =
      GMToDiracConstWeightOptimizationParams(covDiag, wXHelper, N, L, bMax,
                                             c_b(bMax));
  *distance = modified_van_mises_distance_sq(x, &optiParams);
}

template <>
void gm_to_dirac_short<float>::modified_van_mises_distance_sq_derivative(
    const gsl_vector_float* covDiag, gsl_vector_float* gradient, size_t L,
    size_t N, size_t bMax, gsl_vector_float* x, const gsl_vector_float* wX) {
  gsl_vector* gradientDouble = gsl_vector_alloc(gradient->size);

  GSLVectorView<double> vectorViewCovDiag(covDiag, N);
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);
  gm_to_dirac_short<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq_derivative(
      vectorViewCovDiag, gradientDouble, L, N, bMax, vectorViewX, vectorViewWX);

  for (size_t i = 0; i < gradient->size; i++)
    gradient->data[i] = static_cast<float>(gradientDouble->data[i]);

  gsl_vector_free(gradientDouble);
}

template <>
void gm_to_dirac_short<double>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* covDiag, gsl_vector* gradient, size_t L, size_t N,
    size_t bMax, gsl_vector* x, const gsl_vector* wX) {
  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracConstWeightOptimizationParams optiParams =
      GMToDiracConstWeightOptimizationParams(covDiag, wXHelper, N, L, bMax,
                                             c_b(bMax));
  modified_van_mises_distance_sq_derivative(x, &optiParams, gradient);
}

template class gm_to_dirac_short<double>;
template class gm_to_dirac_short<float>;