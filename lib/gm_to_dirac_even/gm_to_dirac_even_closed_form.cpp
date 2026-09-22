#include "gm_to_dirac_even_closed_form.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <cassert>
#include <cmath>
#include <vector>

#include "gsl_minimizer.h"
#include "gsl_utils_view_helper.h"
#include "gsl_utils_weight_helper.h"

template <typename T>
double gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    const gsl_vector* x, void* params) {
  double d = 0.00;
  combined_distance_metric(x, params, &d, nullptr);
  return d;
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* x, void* params, gsl_vector* grad) {
  combined_distance_metric(x, params, nullptr, grad);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::combined_distance_metric(
    const gsl_vector* x, void* params, double* f, gsl_vector* grad) {
  GMToDiracEvenOptimizationParams* optiParams =
      static_cast<GMToDiracEvenOptimizationParams*>(params);

  if (f) *f = 0.00;
  if (grad) gsl_vector_set_zero(grad);

  optiParams->update(x);

  calculateD2(x, optiParams, f, grad);
  calculateD3(x, optiParams, f, grad);

  if (f) *f += optiParams->objectiveOffset;
}


template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate(
    size_t L, size_t N, double bMax, T* x, const T* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x != nullptr);

  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  return approximate(L, N, bMax, vectorViewX.get(), vectorViewWX.get(), result,
                     options);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, double bMax, T* x, const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  modified_van_mises_distance_sq(distance, L, N, bMax, vectorViewX.get(),
                                 vectorViewWX.get());
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    T* gradient, size_t L, size_t N, double bMax, T* x, const T* wX) {
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLVectorView<T> vectorViewGradient(gradient, L * N);
  modified_van_mises_distance_sq_derivative(vectorViewGradient.get(), L, N,
                                            bMax, vectorViewX.get(),
                                            vectorViewWX.get());
}


template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate(
    size_t L, size_t N, double bMax, GSLMatrixType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  return approximate(L, N, bMax, vectorViewX.get(), wX, result, options);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, double bMax, GSLMatrixType* x,
    const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  modified_van_mises_distance_sq(distance, L, N, bMax, vectorViewX.get(), wX);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    GSLMatrixType* gradient, size_t L, size_t N, double bMax, GSLMatrixType* x,
    const GSLVectorType* wX) {
  assert(x->size1 == L);
  assert(x->size2 == N);
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewGradient(gradient);
  modified_van_mises_distance_sq_derivative(vectorViewGradient.get(), L, N,
                                            bMax, vectorViewX.get(), wX);
}


template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate_isotropic(
    size_t L, size_t N, double sigma, double bMax, T* x, const T* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(sigma > 0.00);
  if (!(sigma > 0.00)) return false;

  if (options.initialX) {
    for (size_t i = 0; i < L * N; ++i)
      x[i] = static_cast<T>(static_cast<double>(x[i]) / sigma);
  }

  const bool success =
      approximate(L, N, bMax / sigma, x, wX, result, options);

  for (size_t i = 0; i < L * N; ++i)
    x[i] = static_cast<T>(static_cast<double>(x[i]) * sigma);

  return success;
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_isotropic(
    T* distance, size_t L, size_t N, double sigma, double bMax, T* x,
    const T* wX) {
  assert(sigma > 0.00);

  std::vector<T> scaled(L * N);
  for (size_t i = 0; i < L * N; ++i)
    scaled[i] = static_cast<T>(static_cast<double>(x[i]) / sigma);

  T standardDistance = static_cast<T>(0);
  modified_van_mises_distance_sq(&standardDistance, L, N, bMax / sigma,
                                 scaled.data(), wX);

  *distance = static_cast<T>(sigma * sigma *
                             static_cast<double>(standardDistance));
}


template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate(
    size_t L, size_t N, T* x, const T* wX, GslminimizerResult* result,
    const ApproximateOptions& options) {
  return approximate(L, N, static_cast<double>(options.bMax), x, wX, result,
                     options);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, size_t bMax, T* x, const T* wX) {
  modified_van_mises_distance_sq(distance, L, N, static_cast<double>(bMax), x,
                                 wX);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    T* gradient, size_t L, size_t N, size_t bMax, T* x, const T* wX) {
  modified_van_mises_distance_sq_derivative(gradient, L, N,
                                            static_cast<double>(bMax), x, wX);
}

template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate(
    size_t L, size_t N, GSLVectorType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  return approximate(L, N, static_cast<double>(options.bMax), x, wX, result,
                     options);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, size_t bMax, GSLVectorType* x,
    const GSLVectorType* wX) {
  modified_van_mises_distance_sq(distance, L, N, static_cast<double>(bMax), x,
                                 wX);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    GSLVectorType* gradient, size_t L, size_t N, size_t bMax, GSLVectorType* x,
    const GSLVectorType* wX) {
  modified_van_mises_distance_sq_derivative(gradient, L, N,
                                            static_cast<double>(bMax), x, wX);
}

template <typename T>
bool gm_to_dirac_even_closed_form<T>::approximate(
    size_t L, size_t N, GSLMatrixType* x, const GSLVectorType* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  return approximate(L, N, static_cast<double>(options.bMax), x, wX, result,
                     options);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq(
    T* distance, size_t L, size_t N, size_t bMax, GSLMatrixType* x,
    const GSLVectorType* wX) {
  modified_van_mises_distance_sq(distance, L, N, static_cast<double>(bMax), x,
                                 wX);
}

template <typename T>
void gm_to_dirac_even_closed_form<T>::modified_van_mises_distance_sq_derivative(
    GSLMatrixType* gradient, size_t L, size_t N, size_t bMax, GSLMatrixType* x,
    const GSLVectorType* wX) {
  modified_van_mises_distance_sq_derivative(gradient, L, N,
                                            static_cast<double>(bMax), x, wX);
}

// float

template <>
bool gm_to_dirac_even_closed_form<float>::approximate(
    size_t L, size_t N, double bMax, gsl_vector_float* x,
    const gsl_vector_float* wX, GslminimizerResult* result,
    const ApproximateOptions& options) {
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

  gm_to_dirac_even_closed_form<double> doubleApprox;
  doubleApprox.includeD1InObjective = includeD1InObjective;
  const bool success =
      doubleApprox.approximate(L, N, bMax, xDouble, wXDouble, result, options);

  for (size_t i = 0; i < x->size; ++i)
    x->data[i] = static_cast<float>(xDouble->data[i]);

  gsl_vector_free(xDouble);
  if (wXDouble) gsl_vector_free(wXDouble);

  return success;
}

template <>
void gm_to_dirac_even_closed_form<float>::modified_van_mises_distance_sq(
    float* distance, size_t L, size_t N, double bMax, gsl_vector_float* x,
    const gsl_vector_float* wX) {
  double distanceDouble = 0.00;
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);

  gm_to_dirac_even_closed_form<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq(&distanceDouble, L, N, bMax,
                                              vectorViewX.get(),
                                              vectorViewWX.get());
  *distance = static_cast<float>(distanceDouble);
}

template <>
void gm_to_dirac_even_closed_form<
    float>::modified_van_mises_distance_sq_derivative(
    gsl_vector_float* gradient, size_t L, size_t N, double bMax,
    gsl_vector_float* x, const gsl_vector_float* wX) {
  gsl_vector* gradientDouble = gsl_vector_alloc(gradient->size);

  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWX(wX, L);

  gm_to_dirac_even_closed_form<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq_derivative(
      gradientDouble, L, N, bMax, vectorViewX.get(), vectorViewWX.get());

  for (size_t i = 0; i < gradient->size; ++i)
    gradient->data[i] = static_cast<float>(gradientDouble->data[i]);

  gsl_vector_free(gradientDouble);
}

// double

template <>
bool gm_to_dirac_even_closed_form<double>::approximate(
    size_t L, size_t N, double bMax, gsl_vector* x, const gsl_vector* wX,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(x->size == L * N);
  if (!preconditionsHold(L, N, bMax)) return false;

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
  GMToDiracEvenOptimizationParams params(wXHelper.get(), N, L, bMax,
                                         includeD1InObjective);

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);

  const int status = gslMinimizer.minimize(x, result, options.verbose);

  correctMean(x, params.wX, L, N);

  return status == GSL_SUCCESS;
}

template <>
void gm_to_dirac_even_closed_form<double>::modified_van_mises_distance_sq(
    double* distance, size_t L, size_t N, double bMax, gsl_vector* x,
    const gsl_vector* wX) {
  assert(distance != nullptr);
  if (!preconditionsHold(L, N, bMax)) return;

  GSLWeightHelper<double> wXHelper(wX, L);
  // always report the TRUE distance, offset included, regardless of what the
  // minimizer's objective was configured to carry
  GMToDiracEvenOptimizationParams optiParams(wXHelper.get(), N, L, bMax, true);

  *distance = modified_van_mises_distance_sq(x, &optiParams);
}

template <>
void gm_to_dirac_even_closed_form<
    double>::modified_van_mises_distance_sq_derivative(gsl_vector* gradient,
                                                       size_t L, size_t N,
                                                       double bMax,
                                                       gsl_vector* x,
                                                       const gsl_vector* wX) {
  assert(gradient != nullptr);
  if (!preconditionsHold(L, N, bMax)) return;

  GSLWeightHelper<double> wXHelper(wX, L);
  GMToDiracEvenOptimizationParams optiParams(wXHelper.get(), N, L, bMax, true);

  modified_van_mises_distance_sq_derivative(x, &optiParams, gradient);
}

template class gm_to_dirac_even_closed_form<double>;
template class gm_to_dirac_even_closed_form<float>;
