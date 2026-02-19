#include "dirac_to_dirac_approx_short_function.h"

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
#include "math_util_defs.h"

template <typename T>
bool dirac_to_dirac_approx_short_function<T>::approximate(
    const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x, wXf wXcallback,
    wXd wXDcallback, GslminimizerResult* result,
    const ApproximateOptions& options) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  return approximate(vectorViewY, L, N, bMax, vectorViewX, wXcallback,
                     wXDcallback, result, options);
}

template <typename T>
void dirac_to_dirac_approx_short_function<T>::modified_van_mises_distance_sq(
    T* distance, const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x,
    wXf wXcallback, wXd wXDcallback) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  modified_van_mises_distance_sq(distance, vectorViewY, L, N, bMax, vectorViewX,
                                 wXcallback, wXDcallback);
}

template <typename T>
void dirac_to_dirac_approx_short_function<
    T>::modified_van_mises_distance_sq_derivative(T* gradient, const T* y,
                                                  size_t M, size_t L, size_t N,
                                                  size_t bMax, T* x,
                                                  wXf wXcallback,
                                                  wXd wXDcallback) {
  assert(x != nullptr);
  assert(y != nullptr);

  GSLVectorView<T> vectorViewY(y, M * N);
  GSLVectorView<T> vectorViewX(x, L * N);
  GSLMatrixView<T> matrixViewGradient(gradient, L, N);
  modified_van_mises_distance_sq_derivative(matrixViewGradient, vectorViewY, L,
                                            N, bMax, vectorViewX, wXcallback,
                                            wXDcallback);
}

template <typename T>
bool dirac_to_dirac_approx_short_function<T>::approximate(
    GSLMatrixType* y, size_t L, size_t bMax, GSLMatrixType* x, wXf wXcallback,
    wXd wXDcallback, GslminimizerResult* result,
    const ApproximateOptions& options) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewY(y);
  GSLVectorView<T> vectorViewX(x);
  return approximate(vectorViewY, L, N, bMax, vectorViewX, wXcallback,
                     wXDcallback, result, options);
}

template <typename T>
void dirac_to_dirac_approx_short_function<T>::modified_van_mises_distance_sq(
    T* distance, GSLMatrixType* y, size_t L, size_t bMax, GSLMatrixType* x,
    wXf wXcallback, wXd wXDcallback) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewY(y);
  GSLVectorView<T> vectorViewX(x);
  modified_van_mises_distance_sq(distance, vectorViewY, L, N, bMax, vectorViewX,
                                 wXcallback, wXDcallback);
}

template <typename T>
void dirac_to_dirac_approx_short_function<
    T>::modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                  GSLMatrixType* y, size_t L,
                                                  size_t bMax, GSLMatrixType* x,
                                                  wXf wXcallback,
                                                  wXd wXDcallback) {
  assert(x->size2 == y->size2);
  assert(x->size1 == L);

  size_t N = y->size2;
  GSLVectorView<T> vectorViewY(y);
  GSLVectorView<T> vectorViewX(x);
  modified_van_mises_distance_sq_derivative(
      gradient, vectorViewY, L, N, bMax, vectorViewX, wXcallback, wXDcallback);
}

template <typename T>
inline double dirac_to_dirac_approx_short_function<T>::c_b(size_t bMax) {
  return 100.00;
  return std::pow(std::log(4.00 * static_cast<double>(bMax)), 2) + diagamma_1;
}

template <typename T>
inline double
dirac_to_dirac_approx_short_function<T>::modified_van_mises_distance_sq(
    const gsl_vector* x, void* params) {
  double distance = 0.00;
  dirac_to_dirac_approx_short_function<T>::combined_distance_metric(
      x, params, &distance, nullptr);
  return distance;
}

template <typename T>
inline void dirac_to_dirac_approx_short_function<
    T>::modified_van_mises_distance_sq_derivative(const gsl_vector* x,
                                                  void* params,
                                                  gsl_vector* grad) {
  dirac_to_dirac_approx_short_function<T>::combined_distance_metric(
      x, params, nullptr, grad);
}

template <typename T>
inline void dirac_to_dirac_approx_short_function<T>::combined_distance_metric(
    const gsl_vector* x, void* params, double* f, gsl_vector* grad) {
  DiracToDiracVariableWeightOptimizationParams* optiParams =
      static_cast<DiracToDiracVariableWeightOptimizationParams*>(params);
  gsl_vector* wX = optiParams->wX;
  gsl_matrix* wXderiv = optiParams->wXD;
  const size_t N = optiParams->N;
  const size_t M = optiParams->M;
  const size_t L = optiParams->L;
  const wXf wXcallback = optiParams->wXcallback;
  const wXd wXDcallback = optiParams->wXDcallback;
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

  wXcallback(x->data, wX->data, L, N);
  wXDcallback(x->data, wXderiv->data, L, N);

  gsl_matrix xMatrix = gsl_matrix_view_array(x->data, L, N).matrix;
  squaredEuclideanDistanceUtilLL->calculateDistance(&xMatrix, &xMatrix);
  gsl_matrix yMatrix = gsl_matrix_view_array(y->data, M, N).matrix;
  squaredEuclideanDistanceUtilLM->calculateDistance(&xMatrix, &yMatrix);

  double S = 0.00;
  for (size_t i = 0; i < L; i++) {
    S += wX->data[i];
  }
  const double invS = 1.00 / S;

  gsl_vector* gradTmp = gsl_vector_alloc(L * N);

  // + Dx
  gsl_vector_set_zero(gradTmp);
  double Dx = 0.00;
#pragma omp parallel for reduction(+ : Dx)
  for (size_t i = 0; i < L; i++) {
    const double wXi = wX->data[i];
    for (size_t j = 0; j < L; j++) {
      const double wXj = wX->data[j];
      const double localDistSq =
          squaredEuclideanDistanceUtilLL->getDistance(i, j);

      if (localDistSq <= 0.0) continue;
      const double logLocalDistSq = std::log(localDistSq);
      const double xlogLocalDistSq = localDistSq * logLocalDistSq;
      FMA_ACC(wXi, wXj * xlogLocalDistSq, Dx);
      if (grad) {
        for (size_t k = 0; k < N; ++k) {
          double gradIK = wXderiv->data[i * N + k] * xlogLocalDistSq;
          const double diff = x->data[i * N + k] - x->data[j * N + k];
          FMA_ACC(2.00 * wXi * diff, (logLocalDistSq + 1.00), gradIK);
          FMA_ACC(2.00 * wXj, gradIK, gradTmp->data[i * N + k]);
        }
      }
    }
  }
  if (f) Dx *= invS * invS;
  if (grad) {
    for (size_t xi = 0; xi < L; xi++) {
      for (size_t eta = 0; eta < N; eta++) {
        grad->data[xi * N + eta] +=
            piPrefactor *
            (gradTmp->data[xi * N + eta] / (S * S) -
             2.00 * Dx * wXderiv->data[xi * N + eta] / (S * S * S));
      }
    }
  }

  // - 2*Dxy
  gsl_vector_set_zero(gradTmp);
  double Dxy = 0.00;
#pragma omp parallel for reduction(+ : Dxy)
  for (size_t i = 0; i < L; i++) {
    const double wXi = wX->data[i];
    for (size_t j = 0; j < M; j++) {
      const double wYj = wY->data[j];

      const double localDistSq =
          squaredEuclideanDistanceUtilLM->getDistance(i, j);

      if (localDistSq <= 0.0) continue;

      const double logLocalDistSq = std::log(localDistSq);
      const double xlogLocalDistSq = localDistSq * logLocalDistSq;
      FMA_ACC(xlogLocalDistSq, wXi * wYj, Dxy);

      if (grad) {
        for (size_t k = 0; k < N; ++k) {
          const double diff = x->data[i * N + k] - y->data[j * N + k];
          double gradIK = wXderiv->data[i * N + k] * xlogLocalDistSq;
          gradIK += 2.0 * wXi * diff * (logLocalDistSq + 1.0);
          gradTmp->data[i * N + k] += wYj * gradIK;
        }
      }
    }
  }
  if (f) Dxy *= invS;
  if (grad) {
    for (size_t xi = 0; xi < L; xi++) {
      for (size_t eta = 0; eta < N; eta++) {
        grad->data[xi * N + eta] +=
            -2.00 * piPrefactor *
            (gradTmp->data[xi * N + eta] / S -
             (Dxy / (S * S)) * wXderiv->data[xi * N + eta]);
      }
    }
  }

  // + 2*De
  std::vector<double> meanX(N, 0.0);
  gsl_vector* meanK = optiParams->vecN;
  gsl_vector_set_zero(meanK);
  gsl_vector_set_zero(gradTmp);
  for (size_t i = 0; i < L; i++) {
    for (size_t k = 0; k < N; k++) {
      meanX[k] += wX->data[i] * x->data[i * N + k];
    }
  }
  double De = 0.00;
  for (size_t k = 0; k < N; k++) {
    meanX[k] /= S;
    meanK->data[k] = meanX[k] - meanY->data[k];
    De += meanK->data[k] * meanK->data[k];
  }
  if (grad) {
    const double gradPrefactor = piPrefactor * 4.00 * cB / S;
    for (size_t xi = 0; xi < L; xi++) {
      for (size_t eta = 0; eta < N; eta++) {
        for (size_t k = 0; k < N; k++) {
          grad->data[xi * N + eta] +=
              gradPrefactor * meanK->data[k] *
              (wXderiv->data[xi * N + eta] * (x->data[xi * N + k] - meanX[k]) +
               (k == eta ? wX->data[xi] : 0.00));
        }
      }
    }
  }

  if (f) *f = piPrefactor * ((Dy - 2.00 * Dxy + Dx) + 2.00 * cB * De);

  gsl_vector_free(gradTmp);
}

template <>
bool dirac_to_dirac_approx_short_function<double>::approximate(
    const gsl_vector* y, size_t L, size_t N, size_t bMax, gsl_vector* x,
    wXf wXcallback, wXd wXDcallback, GslminimizerResult* result,
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

  // Set up optimization parameters
  DiracToDiracVariableWeightOptimizationParams params =
      DiracToDiracVariableWeightOptimizationParams(wXcallback, wXDcallback, y,
                                                   N, M, L, bMax, c_b(bMax));

  gsl_minimizer gslMinimizer(
      options.maxIterations, options.xtolAbs, options.xtolRel, options.ftolAbs,
      options.ftolRel, options.gtol, &params, modified_van_mises_distance_sq,
      modified_van_mises_distance_sq_derivative, combined_distance_metric);
  const int status = gslMinimizer.minimize(x, result, options.verbose);
  return status == GSL_SUCCESS;
}

template <>
void dirac_to_dirac_approx_short_function<
    double>::modified_van_mises_distance_sq(double* distance,
                                            const gsl_vector* y, size_t L,
                                            size_t N, size_t bMax,
                                            gsl_vector* x, wXf wXcallback,
                                            wXd wXDcallback) {
  const size_t M = y->size / N;
  DiracToDiracVariableWeightOptimizationParams optiParams =
      DiracToDiracVariableWeightOptimizationParams(wXcallback, wXDcallback, y,
                                                   N, M, L, bMax, c_b(bMax));
  *distance = modified_van_mises_distance_sq(x, &optiParams);
}

template <>
void dirac_to_dirac_approx_short_function<
    double>::modified_van_mises_distance_sq_derivative(gsl_matrix* gradient,
                                                       const gsl_vector* y,
                                                       size_t L, size_t N,
                                                       size_t bMax,
                                                       gsl_vector* x,
                                                       wXf wXcallback,
                                                       wXd wXDcallback) {
  const size_t M = y->size / N;
  DiracToDiracVariableWeightOptimizationParams optiParams =
      DiracToDiracVariableWeightOptimizationParams(wXcallback, wXDcallback, y,
                                                   N, M, L, bMax, c_b(bMax));
  gsl_vector_view flatGradient =
      gsl_vector_view_array(gradient->data, gradient->size1 * gradient->size2);
  modified_van_mises_distance_sq_derivative(x, &optiParams,
                                            &(flatGradient.vector));
}

template class dirac_to_dirac_approx_short_function<double>;