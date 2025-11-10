#include "dirac_to_dirac_approx_short_thread.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_psi.h>
#include <omp.h>

#include <cassert>
#include <iostream>

#include "capture_time.h"
#include "dirac_to_dirac_optimization_params.h"
#include "gsl_minimizer.h"
#include "math_util_defs.h"

#define eps 0.000001
#define diagamma_1 -0.5772156649015328606065120900824

template <typename T>
bool dirac_to_dirac_approx_short_thread<T>::approximate(
    const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x, const T* wX,
    const T* wY, GslminimizerResult* result,
    const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(y != nullptr);
  GSLVectorViewType xFlat =
      GSLTemplateTypeAlias<T>::vector_view_from_array(x, L * N);
  GSLVectorViewType yFlat =
      GSLTemplateTypeAlias<T>::vector_view_from_array(y, M * N);

  GSLVectorType* wXVector = nullptr;
  GSLVectorViewType wXVectorView;
  if (wX) {
    wXVectorView = GSLTemplateTypeAlias<T>::vector_view_from_array(wX, L);
    wXVector = &(wXVectorView.vector);
  }
  GSLVectorType* wYVector = nullptr;
  GSLVectorViewType wYVectorView;
  if (wY) {
    wYVectorView = GSLTemplateTypeAlias<T>::vector_view_from_array(wY, M);
    wYVector = &(wYVectorView.vector);
  }
  return approximate(&(yFlat.vector), L, N, bMax, &(xFlat.vector), wXVector,
                     wYVector, result, options);
}

template <typename T>
bool dirac_to_dirac_approx_short_thread<T>::approximate(
    GSLMatrixType* y, size_t L, size_t bMax, GSLMatrixType* x,
    const GSLVectorType* wX, const GSLVectorType* wY,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorViewType yFlat =
      GSLTemplateTypeAlias<T>::flatten_matrix_to_vector(y);
  GSLVectorViewType xFlat =
      GSLTemplateTypeAlias<T>::flatten_matrix_to_vector(x);
  return approximate(&(yFlat.vector), L, N, bMax, &(xFlat.vector), wX, wY,
                     result, options);
}

template <typename T>
inline double dirac_to_dirac_approx_short_thread<T>::c_b(size_t bMax) {
  return 100.00;
  // return std::log(4.00 * bMax * bMax) + diagamma_1;
  return std::pow(std::log(4.00 * static_cast<double>(bMax)), 2) + diagamma_1;
}

template <typename T>
inline double
dirac_to_dirac_approx_short_thread<T>::modified_van_mises_distance_sq(
    const gsl_vector* x, void* params) {
  double distance = 0.00;
  dirac_to_dirac_approx_short_thread<T>::combined_distance_metric(
      x, params, &distance, nullptr);
  return distance;
}

template <typename T>
inline void dirac_to_dirac_approx_short_thread<
    T>::modified_van_mises_distance_sq_derivative(const gsl_vector* x,
                                                  void* params,
                                                  gsl_vector* grad) {
  dirac_to_dirac_approx_short_thread<T>::combined_distance_metric(
      x, params, nullptr, grad);
}

template <typename T>
inline void dirac_to_dirac_approx_short_thread<T>::combined_distance_metric(
    const gsl_vector* x, void* params, double* f, gsl_vector* grad) {
  DiracToDiracConstWeightOptimizationParams* optiParams =
      static_cast<DiracToDiracConstWeightOptimizationParams*>(params);
  const size_t N = optiParams->N;
  const size_t M = optiParams->M;
  const size_t L = optiParams->L;
  const gsl_vector* wX = optiParams->wX;
  const gsl_vector* wY = optiParams->wY;
  const gsl_vector* y = optiParams->y;
  const double cB = optiParams->cB;
  const double Dy = optiParams->Dy;
  const double piPrefactor = optiParams->piPrefactor;
  const gsl_vector* meanY = optiParams->meanY;
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL =
      optiParams->squaredEuclideanDistanceUtilLL;
  SquaredEuclideanDistanceUtilsLM* squaredEuclideanDistanceUtilLM =
      optiParams->squaredEuclideanDistanceUtilLM;

  if (grad) gsl_vector_set_zero(grad);

  gsl_matrix xMatrix = gsl_matrix_view_array(x->data, L, N).matrix;
  squaredEuclideanDistanceUtilLL->calculateDistance(&xMatrix, &xMatrix);
  gsl_matrix yMatrix = gsl_matrix_view_array(y->data, M, N).matrix;
  squaredEuclideanDistanceUtilLM->calculateDistance(&xMatrix, &yMatrix);

  double Dxy = 0.00;
  double Dx = 0.00;
  double De = 0.00;

#pragma omp parallel for reduction(+ : Dx, Dxy)
  for (size_t i = 0; i < L; i++) {
    const double wXi = wX->data[i];
    // + Dx
    for (size_t j = 0; j < L; j++) {
      const double localDistSq =
          squaredEuclideanDistanceUtilLL->getDistance(i, j);

      if (localDistSq <= 0.0) continue;
      const double logLocalDistSq = std::log(localDistSq);
      if (f) FMA_ACC(wXi, wX->data[j] * localDistSq * logLocalDistSq, Dx);
      if (grad) {
        const double constFactor =
            piPrefactor * 4.00 * wXi * wX->data[j] * (1.00 + logLocalDistSq);
        for (size_t k = 0; k < N; ++k) {
          const double diff = x->data[i * N + k] - x->data[j * N + k];
          FMA_ACC(constFactor, diff, grad->data[i * N + k]);
        }
      }
    }

    // - 2*Dxy
    for (size_t j = 0; j < M; j++) {
      const double localDistSq =
          squaredEuclideanDistanceUtilLM->getDistance(i, j);

      if (localDistSq <= 0.0) continue;
      const double logLocalDistSq = std::log(localDistSq);
      if (f) FMA_ACC(wXi, wY->data[j] * localDistSq * logLocalDistSq, Dxy);
      if (grad) {
        const double constFactor =
            piPrefactor * 2.00 * wXi * wY->data[j] * (1.00 + logLocalDistSq);
        for (size_t k = 0; k < N; ++k) {
          const double diff = x->data[i * N + k] - y->data[j * N + k];
          FMA_ACC(-constFactor, 2.00 * diff, grad->data[i * N + k]);
        }
      }
    }
  }

  // + 2*De
  gsl_vector* meanX = optiParams->vecN;
  gsl_vector_set_zero(meanX);
  for (size_t i = 0; i < L; i++) {
    const double wxi = wX->data[i];
    for (size_t k = 0; k < N; k++) {
      FMA_ACC(wxi, x->data[i * N + k], meanX->data[k]);
    }
  }

  for (size_t k = 0; k < N; k++) {
    const double meanDiff = meanX->data[k] - meanY->data[k];
    if (grad) {
      for (size_t i = 0; i < L; i++) {
        FMA_ACC(piPrefactor * wX->data[i], 4.00 * cB * meanDiff,
                grad->data[i * N + k]);
      }
    }
    if (f) FMA_ACC(meanDiff, meanDiff, De);
  }

  if (f) *f = piPrefactor * ((Dy - 2.00 * Dxy + Dx) + 2.00 * cB * De);
}

template <typename T>
inline void dirac_to_dirac_approx_short_thread<T>::correctMean(
    const gsl_vector* meanY, gsl_vector* x, const gsl_vector* wX, size_t L,
    size_t N) {
  std::vector<double> mean(N, 0.0);
  for (size_t i = 0; i < L; i++) {
    for (size_t k = 0; k < N; k++) {
      mean[k] += wX->data[i] * x->data[i * N + k];
    }
  }
  for (size_t k = 0; k < N; k++) {
    mean[k] -= meanY->data[k];
  }
  for (size_t i = 0; i < L; i++) {
    for (size_t k = 0; k < N; k++) {
      x->data[i * N + k] -= mean[k];
    }
  }
}

template <>
bool dirac_to_dirac_approx_short_thread<float>::approximate(
    const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const gsl_vector_float* wX, const gsl_vector_float* wY,
    GslminimizerResult* result, const ApproximateOptions& options) {
  gsl_vector* yDouble = gsl_vector_alloc(y->size);
  gsl_vector* xDouble = gsl_vector_alloc(x->size);
  gsl_vector* wXDouble = nullptr;
  gsl_vector* wYDouble = nullptr;

  for (size_t i = 0; i < y->size; ++i) {
    yDouble->data[i] = static_cast<double>(y->data[i]);
  }
  if (wX) {
    wXDouble = gsl_vector_alloc(L);
    for (size_t i = 0; i < L; i++) {
      wXDouble->data[i] = static_cast<double>(wX->data[i]);
    }
  }
  if (wY) {
    const size_t M = y->size / N;
    wYDouble = gsl_vector_alloc(M);
    for (size_t i = 0; i < M; i++) {
      wYDouble->data[i] = static_cast<double>(wY->data[i]);
    }
  }

  dirac_to_dirac_approx_short_thread<double> doubleApprox;
  bool success = doubleApprox.approximate(yDouble, L, N, bMax, xDouble,
                                          wXDouble, wYDouble, result, options);

  for (size_t i = 0; i < x->size; ++i) {
    x->data[i] = static_cast<float>(xDouble->data[i]);
  }

  gsl_vector_free(yDouble);
  gsl_vector_free(xDouble);
  if (wXDouble) gsl_vector_free(wXDouble);
  if (wYDouble) gsl_vector_free(wYDouble);

  return success;
}

template <>
bool dirac_to_dirac_approx_short_thread<double>::approximate(
    const gsl_vector* y, size_t L, size_t N, size_t bMax, gsl_vector* x,
    const gsl_vector* wX, const gsl_vector* wY, GslminimizerResult* result,
    const ApproximateOptions& options) {
  assert(x->size == L * N);
  assert(y->size % N == 0);
  assert(x->size % N == 0);

  size_t M = y->size / N;
  if (!options.initialX) {
    for (size_t iX = 0; iX < x->size; iX += N) {
      for (size_t iD = 0; iD < N; iD++) {
        x->data[iX + iD] = y->data[(iX * (y->size / x->size)) + iD];
      }
    }
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
  const gsl_vector* localWY;
  const bool freeWy = wY == nullptr;
  if (freeWy) {
    gsl_vector* tmpWy = gsl_vector_alloc(M);
    gsl_vector_set_all(tmpWy, 1.00 / static_cast<double>(M));
    localWY = tmpWy;
  } else {
    localWY = wY;
  }

  // Set up optimization parameters
  DiracToDiracConstWeightOptimizationParams params =
      DiracToDiracConstWeightOptimizationParams(localWX, localWY, y, N, M, L,
                                                bMax, c_b(bMax));

  gsl_minimizer gslMinimizer = gsl_minimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);
  const int status = gslMinimizer.minimize(x, result, options.verbose);

  correctMean(params.meanY, x, params.wX, L, N);

  return status == GSL_SUCCESS;
}

template class dirac_to_dirac_approx_short_thread<double>;
template class dirac_to_dirac_approx_short_thread<float>;