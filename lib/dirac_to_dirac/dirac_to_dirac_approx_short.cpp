#include "dirac_to_dirac_approx_short.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_psi.h>

#include <cassert>
#include <iostream>

#include "capture_time.h"
#include "dirac_to_dirac_optimization_params.h"
#include "gsl_minimizer.h"
#include "gsl_utils_view_helper.h"
#include "gsl_utils_weight_helper.h"
#include "math_util_defs.h"
#include "squared_euclidean_distance_utils.h"

#define diagamma_1 -0.5772156649015328606065120900824

template <typename T>
bool dirac_to_dirac_approx_short<T>::approximate(
    const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x, const T* wX,
    const T* wY, GslminimizerResult* result,
    const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWY(wY, M);
  GSLVectorView<T> vectorViewWX(wX, L);
  return approximate(vectorViewY, L, N, bMax, vectorViewX, vectorViewWX,
                     vectorViewWY, result, options);
}

template <typename T>
void dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq(
    T* distance, const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x,
    const T* wX, const T* wY) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWY(wY, M);
  GSLVectorView<T> vectorViewWX(wX, L);
  modified_van_mises_distance_sq(distance, vectorViewY, L, N, bMax, vectorViewX,
                                 vectorViewWX, vectorViewWY);
}

template <typename T>
void dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq_derivative(
    T* gradient, const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x,
    const T* wX, const T* wY) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLVectorView<T> vectorViewWY(wY, M);
  GSLVectorView<T> vectorViewWX(wX, L);
  GSLMatrixView<T> matrixViewGradient(gradient, L, N);
  modified_van_mises_distance_sq_derivative(matrixViewGradient, vectorViewY, L,
                                            N, bMax, vectorViewX, vectorViewWX,
                                            vectorViewWY);
}

template <typename T>
bool dirac_to_dirac_approx_short<T>::approximate(
    GSLMatrixType* y, size_t L, size_t bMax, GSLMatrixType* x,
    const GSLVectorType* wX, const GSLVectorType* wY,
    GslminimizerResult* result, const ApproximateOptions& options) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewY(y);
  return approximate(vectorViewY, L, N, bMax, vectorViewX, wX, wY, result,
                     options);
}

template <typename T>
void dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq(
    T* distance, GSLMatrixType* y, size_t L, size_t bMax, GSLMatrixType* x,
    const GSLVectorType* wX, const GSLVectorType* wY) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewY(y);
  modified_van_mises_distance_sq(distance, vectorViewY, L, N, bMax, vectorViewX,
                                 wX, wY);
}

template <typename T>
void dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq_derivative(
    GSLMatrixType* gradient, GSLMatrixType* y, size_t L, size_t bMax,
    GSLMatrixType* x, const GSLVectorType* wX, const GSLVectorType* wY) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewX(x);
  GSLVectorView<T> vectorViewY(y);
  modified_van_mises_distance_sq_derivative(gradient, vectorViewY, L, N, bMax,
                                            vectorViewX, wX, wY);
}

template <typename T>
inline double dirac_to_dirac_approx_short<T>::c_b(size_t bMax) {
  return 100.00;
  // return std::log(4.00 * bMax * bMax) + diagamma_1;
  return std::pow(std::log(4.00 * static_cast<double>(bMax)), 2) + diagamma_1;
}

template <typename T>
inline double dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq(
    const gsl_vector* x, void* params) {
  double distance = 0.00;
  dirac_to_dirac_approx_short<T>::combined_distance_metric(x, params, &distance,
                                                           nullptr);
  return distance;
}

template <typename T>
inline void
dirac_to_dirac_approx_short<T>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* x, void* params, gsl_vector* grad) {
  dirac_to_dirac_approx_short<T>::combined_distance_metric(x, params, nullptr,
                                                           grad);
}

template <typename T>
inline void dirac_to_dirac_approx_short<T>::combined_distance_metric(
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

  if (f) *f = 0.00;
  if (grad) gsl_vector_set_zero(grad);

  gsl_matrix xMatrix = gsl_matrix_view_array(x->data, L, N).matrix;
  squaredEuclideanDistanceUtilLL->calculateDistance(&xMatrix, &xMatrix);
  gsl_matrix yMatrix = gsl_matrix_view_array(y->data, M, N).matrix;
  squaredEuclideanDistanceUtilLM->calculateDistance(&xMatrix, &yMatrix);

  double Dxy = 0.00;
  double Dx = 0.00;
  double De = 0.00;

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
inline void dirac_to_dirac_approx_short<T>::correctMean(
    const GSLVectorType* meanY, GSLVectorType* x, const GSLVectorType* wX,
    size_t L, size_t N) {
  std::vector<T> mean(N, T(0));
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
bool dirac_to_dirac_approx_short<float>::approximate(
    const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const gsl_vector_float* wX, const gsl_vector_float* wY,
    GslminimizerResult* result, const ApproximateOptions& options) {
  const size_t M = y->size / N;

  GSLVectorView<double> vectorViewY(y, M * N);
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWY(wY, M);
  GSLVectorView<double> vectorViewWX(wX, L);
  dirac_to_dirac_approx_short<double> doubleApprox;
  bool success =
      doubleApprox.approximate(vectorViewY, L, N, bMax, vectorViewX,
                               vectorViewWX, vectorViewWY, result, options);

  const size_t xSize = x->size;
  for (size_t i = 0; i < xSize; ++i) {
    x->data[i] = static_cast<float>(vectorViewX.get()->data[i]);
  }

  return success;
}

template <>
bool dirac_to_dirac_approx_short<double>::approximate(
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

  GSLWeightHelper<double> wXHelper(wX, L);
  GSLWeightHelper<double> wYHelper(wY, M);

  DiracToDiracConstWeightOptimizationParams params =
      DiracToDiracConstWeightOptimizationParams(wXHelper, wYHelper, y, N, M, L,
                                                bMax, c_b(bMax));

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);
  const int status = gslMinimizer.minimize(x, result, options.verbose);

  correctMean(params.meanY, x, params.wX, L, N);

  return status == GSL_SUCCESS;
}

template <>
void dirac_to_dirac_approx_short<float>::modified_van_mises_distance_sq(
    float* distance, const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const gsl_vector_float* wX,
    const gsl_vector_float* wY) {
  double distanceDouble = 0.00;
  const size_t M = y->size / N;
  GSLVectorView<double> vectorViewY(y, M * N);
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWY(wY, M);
  GSLVectorView<double> vectorViewWX(wX, L);
  dirac_to_dirac_approx_short<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq(&distanceDouble, vectorViewY, L,
                                              N, bMax, vectorViewX,
                                              vectorViewWX, vectorViewWY);
  *distance = static_cast<float>(distanceDouble);
}

template <>
void dirac_to_dirac_approx_short<double>::modified_van_mises_distance_sq(
    double* distance, const gsl_vector* y, size_t L, size_t N, size_t bMax,
    gsl_vector* x, const gsl_vector* wX, const gsl_vector* wY) {
  const size_t M = y->size / N;
  GSLWeightHelper<double> wXHelper(wX, L);
  GSLWeightHelper<double> wYHelper(wY, M);
  DiracToDiracConstWeightOptimizationParams optiParams =
      DiracToDiracConstWeightOptimizationParams(wXHelper, wYHelper, y, N, M, L, bMax,
                                                c_b(bMax));
  *distance = modified_van_mises_distance_sq(x, &optiParams);
}

template <>
void dirac_to_dirac_approx_short<float>::
    modified_van_mises_distance_sq_derivative(gsl_matrix_float* gradient,
                                              const gsl_vector_float* y,
                                              size_t L, size_t N, size_t bMax,
                                              gsl_vector_float* x,
                                              const gsl_vector_float* wX,
                                              const gsl_vector_float* wY) {
  gsl_matrix* gradientDouble =
      gsl_matrix_alloc(gradient->size1, gradient->size2);

  const size_t M = y->size / N;
  GSLVectorView<double> vectorViewY(y, M * N);
  GSLVectorView<double> vectorViewX(x, L * N);
  GSLVectorView<double> vectorViewWY(wY, M);
  GSLVectorView<double> vectorViewWX(wX, L);
  dirac_to_dirac_approx_short<double> doubleApprox;
  doubleApprox.modified_van_mises_distance_sq_derivative(
      gradientDouble, vectorViewY, L, N, bMax, vectorViewX, vectorViewWX,
      vectorViewWY);

  const size_t gradientSize = gradient->size1 * gradient->size2;
  for (size_t i = 0; i < gradientSize; i++)
    gradient->data[i] = static_cast<float>(gradientDouble->data[i]);

  gsl_matrix_free(gradientDouble);
}

template <>
void dirac_to_dirac_approx_short<
    double>::modified_van_mises_distance_sq_derivative(gsl_matrix* gradient,
                                                       const gsl_vector* y,
                                                       size_t L, size_t N,
                                                       size_t bMax,
                                                       gsl_vector* x,
                                                       const gsl_vector* wX,
                                                       const gsl_vector* wY) {
  const size_t M = y->size / N;
  GSLWeightHelper<double> wXHelper(wX, L);
  GSLWeightHelper<double> wYHelper(wY, M);
  DiracToDiracConstWeightOptimizationParams optiParams =
      DiracToDiracConstWeightOptimizationParams(wXHelper, wYHelper, y, N, M, L, bMax,
                                                c_b(bMax));
  gsl_vector_view flatGradient =
      gsl_vector_view_array(gradient->data, gradient->size1 * gradient->size2);
  modified_van_mises_distance_sq_derivative(x, &optiParams,
                                            &(flatGradient.vector));
}

template class dirac_to_dirac_approx_short<double>;
template class dirac_to_dirac_approx_short<float>;
