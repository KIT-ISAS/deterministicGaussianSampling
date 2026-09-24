#include "gm_to_dirac_even_binf.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#include "gsl_minimizer.h"
#include "gsl_utils_view_helper.h"
#include "gsl_utils_weight_helper.h"

namespace {
  constexpr double kBinfFailureValue = std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief objective and gradient of the bMax-free distance
 *
 * The order matters: project t onto the zero-mean manifold and refresh the
 * c_i / T_ij caches first, then accumulate the gradient with respect to s.
 * With chainRuleToT set, the assembled gradient is finally mapped back to the
 * minimizer's variable t; without it, grad_s is returned as is.
 *
 * A sample projected exactly onto the origin is outside the closed form's
 * domain (see GMToDiracEvenBinfOptimizationParams::hasDegenerateSample); it
 * yields NaN here rather than the abort lcd_ei(0) would trigger.
 *
 * @param t optimizer variable (or, for the public entry points, the sample locations), L * N
 * @param params optimization parameters
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator, may be nullptr
 * @param chainRuleToT true for grad_t, false for grad_s
 */
template <typename T>
void gm_to_dirac_even_binf<T>::evaluate(
    const gsl_vector* t, GMToDiracEvenBinfOptimizationParams* params, double* f,
    gsl_vector* grad, bool chainRuleToT) {
  if (f) *f = 0.00;
  if (grad) gsl_vector_set_zero(grad);

  params->update(t);  // s = t - 1 * (w^T t), then the caches

  if (params->hasDegenerateSample()) {
    if (f) *f = kBinfFailureValue;
    if (grad)
      for (size_t i = 0; i < grad->size; ++i) grad->data[i] = kBinfFailureValue;
    return;
  }

  calculateAttraction(params, f, grad);
  calculateRepulsion(params, f, grad);

  if (grad && chainRuleToT) projectGradient(params, grad);
}

template <typename T>
double gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq(
    const gsl_vector* t, void* params) {
  double d = 0.00;
  evaluate(t, static_cast<GMToDiracEvenBinfOptimizationParams*>(params), &d,
           nullptr, true);
  return d;
}

template <typename T>
void gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* t, void* params, gsl_vector* grad) {
  evaluate(t, static_cast<GMToDiracEvenBinfOptimizationParams*>(params),
           nullptr, grad, true);
}

template <typename T>
void gm_to_dirac_even_binf<T>::combined_distance_metric(const gsl_vector* t,
                                                        void* params, double* f,
                                                        gsl_vector* grad) {
  evaluate(t, static_cast<GMToDiracEvenBinfOptimizationParams*>(params), f,
           grad, true);
}

// raw pointer

template <typename T>
bool gm_to_dirac_even_binf<T>::approximate(size_t L, size_t N, T* x,
                                           const T* wX,
                                           GslminimizerResult* result,
                                           const ApproximateOptions& options) {
  assert(x != nullptr);

  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  return approximate(L, N, vectorViewX.get(), vectorViewWX.get(), result,
                     options);
}

template <typename T>
void gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq(T* distance,
                                                              size_t L,
                                                              size_t N, T* x,
                                                              const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  modified_van_mises_distance_sq(distance, L, N, vectorViewX.get(),
                                 vectorViewWX.get());
}

template <typename T>
void gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq_derivative(
    T* gradient, size_t L, size_t N, T* x, const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLVectorView<T> vectorViewGradient(gradient, L * N);
  modified_van_mises_distance_sq_derivative(vectorViewGradient.get(), L, N,
                                            vectorViewX.get(),
                                            vectorViewWX.get());
}

// gsl matrix

template <typename T>
bool gm_to_dirac_even_binf<T>::approximate(size_t L, size_t N, GSLMatrixType* x,
                                           const GSLVectorType* wX,
                                           GslminimizerResult* result,
                                           const ApproximateOptions& options) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  return approximate(L, N, vectorViewX.get(), wX, result, options);
}

template <typename T>
void gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, GSLMatrixType* x,
    const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  modified_van_mises_distance_sq(distance, L, N, vectorViewX.get(), wX);
}

template <typename T>
void gm_to_dirac_even_binf<T>::modified_van_mises_distance_sq_derivative(
    GSLMatrixType* gradient, size_t L, size_t N, GSLMatrixType* x,
    const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewGradient(gradient);
  modified_van_mises_distance_sq_derivative(vectorViewGradient.get(), L, N,
                                            vectorViewX.get(), wX);
}

// float

template <>
bool gm_to_dirac_even_binf<float>::approximate(
    size_t L, size_t N, gsl_vector_float* x, const gsl_vector_float* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(x->size == L * N);

  gsl_vector* xDouble = gsl_vector_alloc(x->size);
  gsl_vector* wXDouble = nullptr;

  if (wX) {
    wXDouble = gsl_vector_alloc(wX->size);
    for (size_t i = 0; i < wX->size; ++i)
      wXDouble->data[i] = static_cast<double>(wX->data[i]);
  }

  if (options.initialX) {
    for (size_t i = 0; i < x->size; ++i)
      xDouble->data[i] = static_cast<double>(x->data[i]);
  }

  gm_to_dirac_even_binf<double> doubleApprox;
  const bool success =
      doubleApprox.approximate(L, N, xDouble, wXDouble, result, options);

  for (size_t i = 0; i < x->size; ++i)
    x->data[i] = static_cast<float>(xDouble->data[i]);

  gsl_vector_free(xDouble);
  if (wXDouble) gsl_vector_free(wXDouble);

  return success;
}

template <>
void gm_to_dirac_even_binf<float>::modified_van_mises_distance_sq(
    float* distance, size_t L, size_t N, gsl_vector_float* x,
    const gsl_vector_float* wX) {
  double distanceDouble = 0.00;
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);

  gm_to_dirac_even_binf<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq(
      &distanceDouble, L, N, vectorViewX.get(), vectorViewWX.get());
  *distance = static_cast<float>(distanceDouble);
}

template <>
void gm_to_dirac_even_binf<float>::modified_van_mises_distance_sq_derivative(
    gsl_vector_float* gradient, size_t L, size_t N, gsl_vector_float* x,
    const gsl_vector_float* wX) {
  gsl_vector* gradientDouble = gsl_vector_alloc(gradient->size);

  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);

  gm_to_dirac_even_binf<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq_derivative(
      gradientDouble, L, N, vectorViewX.get(), vectorViewWX.get());

  for (size_t i = 0; i < gradient->size; ++i)
    gradient->data[i] = static_cast<float>(gradientDouble->data[i]);

  gsl_vector_free(gradientDouble);
}

// double

template <>
bool gm_to_dirac_even_binf<double>::approximate(
    size_t L, size_t N, gsl_vector* x, const gsl_vector* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(x->size == L * N);
  if (!preconditionsHold(L, N)) return false;

  if (!options.initialX) {
    gsl_rng_env_setup();
    gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
    for (size_t i = 0; i < L; ++i) {
      for (size_t d = 0; d < N; ++d)
        x->data[i * N + d] = gsl_ran_gaussian(r, 1.00);  // standard normal
    }
    gsl_rng_free(r);
  }

  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracEvenBinfOptimizationParams params(wXHelper.get(), N, L);

  // refuse a start that projects a sample onto the origin before the
  // minimizer ever sees it; L == 1 always does, having only the point 0 on
  // the zero-mean manifold
  params.update(x);
  if (params.hasDegenerateSample()) return false;

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);

  const int status = gslMinimizer.minimize(x, result, options.verbose);

  // x holds the optimizer variable t; this is the very projection every
  // evaluation applied internally, so it returns the sample set s that the
  // reported objective belongs to
  correctMean(x, params.wX, L, N);

  return status == GSL_SUCCESS;
}

template <>
void gm_to_dirac_even_binf<double>::modified_van_mises_distance_sq(
    double* distance, size_t L, size_t N, gsl_vector* x, const gsl_vector* wX) {
  assert(distance != nullptr);
  if (!preconditionsHold(L, N)) {
    *distance = kBinfFailureValue;
    return;
  }

  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracEvenBinfOptimizationParams optiParams(wXHelper.get(), N, L);

  evaluate(x, &optiParams, distance, nullptr, false);
}

template <>
void gm_to_dirac_even_binf<double>::modified_van_mises_distance_sq_derivative(
    gsl_vector* gradient, size_t L, size_t N, gsl_vector* x,
    const gsl_vector* wX) {
  assert(gradient != nullptr);
  if (!preconditionsHold(L, N)) {
    for (size_t i = 0; i < gradient->size; ++i)
      gradient->data[i] = kBinfFailureValue;
    return;
  }

  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracEvenBinfOptimizationParams optiParams(wXHelper.get(), N, L);

  // grad_s, the gradient w.r.t. the sample locations; the minimizer's grad_t
  // additionally carries the projection chain rule
  evaluate(x, &optiParams, nullptr, gradient, false);
}

template class gm_to_dirac_even_binf<double>;
template class gm_to_dirac_even_binf<float>;
