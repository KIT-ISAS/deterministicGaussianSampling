#ifndef DIRAC_TO_DIRAC_OPTIMIZATION_PARAMS_H
#define DIRAC_TO_DIRAC_OPTIMIZATION_PARAMS_H

#include <gsl/gsl_vector.h>

#include <cassert>

#include "dirac_to_dirac_approx_function_i.h"
#include "gsl_minimizer.h"
#include "math_util_defs.h"
#include "squared_euclidean_distance_utils.h"

/**
 * @brief base struct for the Dirac To Dirac optimization parameters
 *
 * contains the common parameters for the optimization
 *
 */
struct DiracToDiracBaseOptimizationParams
    : public GslMinimizerOptimizationParams {
 public:
  /**
   * @brief Construct a new Dirac To Dirac Base Optimization Params object
   *
   * @param wY weight vector for the y data points
   * @param y input data points
   * @param N dimension of the data
   * @param M number of input data points
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   * @param Dy Dy
   */
  DiracToDiracBaseOptimizationParams(const gsl_vector* wY, const gsl_vector* y,
                                     size_t N, size_t M, size_t L, size_t bMax,
                                     double cB)
      : GslMinimizerOptimizationParams(L, N),
        wY(wY),
        y(y),
        M(M),
        bMax(bMax),
        cB(cB),
        Dy(calculateDy(wY, y, M, N)),
        piPrefactor(calculatePiPrefactor(N)),
        meanY(createMeanY(y, wY, M, N)) {
    assert(wY != nullptr);
    assert(y != nullptr);
    assert(N > 0);
    assert(M > 0);
    assert(L > 0);
    assert(bMax > 0);
    vecN = gsl_vector_alloc(N);
    squaredEuclideanDistanceUtilLL =
        new SquaredEuclideanDistance_LL_vectorized(L, N);
    squaredEuclideanDistanceUtilLM =
        new SquaredEuclideanDistance_LM_matrix(L, M, N);
  }

  /**
   * @brief Destroy the Dirac To Dirac Base Optimization Params object
   *
   */
  ~DiracToDiracBaseOptimizationParams() {
    if (vecN) gsl_vector_free(vecN);
    if (squaredEuclideanDistanceUtilLL) delete squaredEuclideanDistanceUtilLL;
    if (squaredEuclideanDistanceUtilLM) delete squaredEuclideanDistanceUtilLM;
    if (meanY) gsl_vector_free(const_cast<gsl_vector*>(meanY));
  }

  const gsl_vector* wY;
  const gsl_vector* y;
  const size_t M;
  const size_t bMax;
  const double cB;
  const double Dy;
  const double piPrefactor;
  const gsl_vector* meanY;
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL;
  SquaredEuclideanDistanceUtilsLM* squaredEuclideanDistanceUtilLM;
  gsl_vector* vecN;

 private:
  static gsl_vector* createMeanY(const gsl_vector* y, const gsl_vector* wY,
                                 size_t M, size_t N) {
    gsl_vector* meanY = gsl_vector_calloc(N);
    for (size_t k = 0; k < N; k++) {
      for (size_t j = 0; j < M; j++) {
        FMA_ACC(wY->data[j], y->data[j * N + k], meanY->data[k]);
      }
    }
    return meanY;
  }

  static double calculateDy(const gsl_vector* wY, const gsl_vector* y, size_t M,
                            size_t N) {
    SquaredEuclideanDistance_LL_vectorized squaredEuclideanDistanceUtilLL(M, N);
    gsl_matrix yMatrix = gsl_matrix_view_array(y->data, M, N).matrix;
    squaredEuclideanDistanceUtilLL.calculateDistance(&yMatrix, &yMatrix);

    const int numThreads = omp_get_max_threads();
    std::vector<double> threadDy((size_t)numThreads, 0.0);

#pragma omp parallel for
    for (size_t i = 1; i < M; i++) {
      const double wYi = wY->data[i];
      for (size_t j = 0; j < i; j++) {
        const double localDistSq =
            squaredEuclideanDistanceUtilLL.getDistance(i, j);
        if (localDistSq <= 0.0) continue;

        const double logLocalDistSq = std::log(localDistSq);
        threadDy[(size_t)omp_get_thread_num()] +=
            wYi * wY->data[j] * localDistSq * logLocalDistSq;
      }
    }

    double Dy = 0.00;
    for (size_t i = 0; i < (size_t)numThreads; ++i) {
      Dy += threadDy[i];
    }
    return 2.00 * Dy;
  }

  static double calculatePiPrefactor(size_t N) {
    return std::pow(M_PI, static_cast<double>(N) / 2.00) * 0.125;
  }
};

/**
 * @brief optimization parameters for the Dirac To Dirac approximation with
 * constant weights
 *
 */
struct DiracToDiracConstWeightOptimizationParams
    : public DiracToDiracBaseOptimizationParams {
 public:
  /**
   * @brief Construct a new Dirac To Dirac Const Weight Optimization Params
   * object
   *
   * uses custom weights wX
   *
   * @param wX weight vector for the x data points
   * @param wY weight vector for the y data points
   * @param y input data points
   * @param N dimension of the data
   * @param M number of input data points
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   */
  DiracToDiracConstWeightOptimizationParams(const gsl_vector* wX,
                                            const gsl_vector* wY,
                                            const gsl_vector* y, size_t N,
                                            size_t M, size_t L, size_t bMax,
                                            double cB)
      : DiracToDiracBaseOptimizationParams(wY, y, N, M, L, bMax, cB),
        wX(wX),
        freeWeights(false) {}

  /**
   * @brief Construct a new Dirac To Dirac Const Weight Optimization Params
   * object
   *
   * automatically creates weights wX = 1.00 / L, wY = 1.00 / M
   *
   * @param y input data points
   * @param N dimension of the data
   * @param M number of input data points
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   */
  DiracToDiracConstWeightOptimizationParams(const gsl_vector* y, size_t N,
                                            size_t M, size_t L, size_t bMax,
                                            double cB)
      : DiracToDiracBaseOptimizationParams(getConstWeight(M), y, N, M, L, bMax,
                                           cB),
        wX(getConstWeight(L)),
        freeWeights(true) {}

  /**
   * @brief Destroy the Dirac To Dirac Const Weight Optimization Params object
   *
   */
  ~DiracToDiracConstWeightOptimizationParams() {
    if (freeWeights) {
      gsl_vector_free(const_cast<gsl_vector*>(wX));
      gsl_vector_free(const_cast<gsl_vector*>(wY));
    }
  }

  const gsl_vector* wX;

 private:
  const bool freeWeights;

  static gsl_vector* getConstWeight(size_t size) {
    gsl_vector* constWeight = gsl_vector_alloc(size);
    gsl_vector_set_all(constWeight, (1.00 / static_cast<double>(size)));
    return constWeight;
  }
};

/**
 * @brief optimization parameters for the GMToDirac approximation with variable
 * weights
 *
 */
struct DiracToDiracVariableWeightOptimizationParams
    : public DiracToDiracBaseOptimizationParams {
 public:
  DiracToDiracVariableWeightOptimizationParams(
      dirac_to_dirac_approx_function_i<double>::wXf wXcallback,
      dirac_to_dirac_approx_function_i<double>::wXd wXDcallback,
      const gsl_vector* y, size_t N, size_t M, size_t L, size_t bMax, double cB)
      : DiracToDiracBaseOptimizationParams(getConstWeight(M), y, N, M, L, bMax,
                                           cB),
        wXcallback(wXcallback),
        wXDcallback(wXDcallback),
        freeWeight(true) {
    setup();
  }

  DiracToDiracVariableWeightOptimizationParams(
      const gsl_vector* wY,
      dirac_to_dirac_approx_function_i<double>::wXf wXcallback,
      dirac_to_dirac_approx_function_i<double>::wXd wXDcallback,
      const gsl_vector* y, size_t N, size_t M, size_t L, size_t bMax, double cB)
      : DiracToDiracBaseOptimizationParams(wY, y, N, M, L, bMax, cB),
        wXcallback(wXcallback),
        wXDcallback(wXDcallback),
        freeWeight(false) {
    setup();
  }

  ~DiracToDiracVariableWeightOptimizationParams() {
    if (wX) gsl_vector_free(wX);
    if (wXD) gsl_matrix_free(wXD);
    if (wXN) gsl_vector_free(wXN);
    if (wXNDXiEta) gsl_vector_free(wXNDXiEta);
    if (freeWeight) gsl_vector_free(const_cast<gsl_vector*>(wY));
    if (squaredEuclideanDistanceUtilLL) delete squaredEuclideanDistanceUtilLL;
    if (squaredEuclideanDistanceUtilLM) delete squaredEuclideanDistanceUtilLM;
  }
  gsl_vector* wX;
  gsl_matrix* wXD;
  gsl_vector* wXN;
  gsl_vector* wXNDXiEta;
  dirac_to_dirac_approx_function_i<double>::wXf wXcallback;
  dirac_to_dirac_approx_function_i<double>::wXd wXDcallback;
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL;
  SquaredEuclideanDistanceUtilsLM* squaredEuclideanDistanceUtilLM;

 private:
  const bool freeWeight;

  void setup() {
    this->wX = gsl_vector_alloc(L);
    this->wXD = gsl_matrix_alloc(L, N);
    this->wXN = gsl_vector_alloc(L);
    this->wXNDXiEta = gsl_vector_alloc(L);
    squaredEuclideanDistanceUtilLL =
        new SquaredEuclideanDistance_LL_vectorized(L, N);
    squaredEuclideanDistanceUtilLM =
        new SquaredEuclideanDistance_LM_matrix(L, M, N);
  }

  static gsl_vector* getConstWeight(size_t size) {
    gsl_vector* constWeight = gsl_vector_alloc(size);
    gsl_vector_set_all(constWeight, (1.00 / static_cast<double>(size)));
    return constWeight;
  }
};

#endif  // DIRAC_TO_DIRAC_OPTIMIZATION_PARAMS_H