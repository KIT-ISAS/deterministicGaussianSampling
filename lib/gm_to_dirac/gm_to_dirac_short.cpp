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
#include "math_util_defs.h"

template <typename T>
bool gm_to_dirac_short<T>::approximate(const T* covDiag, size_t L, size_t N,
                                       size_t bMax, T* x, const T* wX,
                                       GslminimizerResult* result,
                                       const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(covDiag != nullptr);

  GSLVectorViewType xFlat =
      GSLTemplateTypeAlias<T>::vector_view_from_array(x, L * N);
  GSLVectorViewType covDiagView =
      GSLTemplateTypeAlias<T>::vector_view_from_array(covDiag, N);

  GSLVectorType* wXVector = nullptr;
  GSLVectorViewType wXVectorView;
  if (wX) {
    wXVectorView = GSLTemplateTypeAlias<T>::vector_view_from_array(wX, L);
    wXVector = &(wXVectorView.vector);
  }
  approximate(&(covDiagView.vector), L, N, bMax, &(xFlat.vector), wXVector,
              result, options);
  return true;
}

template <typename T>
bool gm_to_dirac_short<T>::approximate(const GSLVectorType* covDiag, size_t L,
                                       size_t N, size_t bMax, GSLMatrixType* x,
                                       const GSLVectorType* wX,
                                       GslminimizerResult* result,
                                       const ApproximateOptions& options) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorViewType xFlat =
      GSLTemplateTypeAlias<T>::flatten_matrix_to_vector(x);
  return approximate(covDiag, L, N, bMax, &(xFlat.vector), wX, result, options);
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

  const gsl_vector* localWX;
  const bool freeWx = wX == nullptr;
  if (freeWx) {
    gsl_vector* tmpWx = gsl_vector_alloc(L);
    gsl_vector_set_all(tmpWx, 1.00 / static_cast<double>(L));
    localWX = tmpWx;
  } else {
    localWX = wX;
  }

  GMToDiracConstWeightOptimizationParams params(covDiag, localWX, N, L, bMax,
                                                c_b(bMax));

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);

  const int status = gslMinimizer.minimize(x, result, options.verbose);

  correctMean(x, params.wX, L, N);

  if (freeWx) gsl_vector_free(const_cast<gsl_vector*>(localWX));

  return status == GSL_SUCCESS;
}

template class gm_to_dirac_short<double>;
template class gm_to_dirac_short<float>;