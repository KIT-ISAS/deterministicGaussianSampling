#ifndef GM_TO_DIRAC_OPTIMIZATION_PARAMS_H
#define GM_TO_DIRAC_OPTIMIZATION_PARAMS_H

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_vector.h>

#include <cassert>

#include "floating_bucket_cache.h"
#include "gsl_minimizer.h"
#include "gsl_quadrature_adaptive_gauss_kronrod.h"
#include "math_util_defs.h"
#include "squared_euclidean_distance_utils.h"

/**
 * @brief additional parameters for the GMToDiracIntegration function
 *
 */
struct GMToDiracIntegrationParams {
  GMToDiracIntegrationParams(size_t L, size_t bSizeEstimate) {
    cacheManagerPrefactor = new floatingBucketCacheManager(bSizeEstimate);
    cacheManagerInnerSum =
        new floatingBucketCacheManagerIntKey(L * bSizeEstimate);
    int numThreads = omp_get_max_threads();
    xi = new size_t[numThreads];
    eta = new size_t[numThreads];
  }

  ~GMToDiracIntegrationParams() {
    if (cacheManagerPrefactor) delete cacheManagerPrefactor;
    if (cacheManagerInnerSum) delete cacheManagerInnerSum;
    if (xi) delete[] xi;
    if (eta) delete[] eta;
  }

  /**
   * @brief resets cacheManager and x to the new x
   *
   * should be called whenever x changes
   *
   * @param x
   */
  inline void reset(const gsl_vector* x) {
    cacheManagerInnerSum->clear();
    this->x = x;
  }

  size_t* xi;
  size_t* eta;
  const gsl_vector* x;
  floatingBucketCacheManager* cacheManagerPrefactor;
  floatingBucketCacheManagerIntKey* cacheManagerInnerSum;
};

/**
 * @brief base struct for the GMToDirac optimization parameters
 *
 * contains the common parameters for the optimization
 *
 */
struct GMToDiracBaseOptimizationParams : public GslMinimizerOptimizationParams {
 public:
  /**
   * @brief Construct a new GMToDiracBaseOptimizationParams object
   *
   * @param covDiag covariance matrix diagonal
   * @param N dimension of the data
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   */
  GMToDiracBaseOptimizationParams(const gsl_vector* covDiag, size_t N, size_t L,
                                  size_t bMax, double cB)
      : GslMinimizerOptimizationParams(L, N),
        covDiag(covDiag),
        covDiagSqrd(createCovDiagSqrd(covDiag)),
        bMax(bMax),
        cB(cB),
        // twoPiNHalf(std::pow(2.00 * M_PI, static_cast<double>(N) / 2.00)),
        twoPiNHalf(std::pow(2.00, static_cast<double>(N) / 2.00)),
        D1(calculateD1(bMax, covDiag, N)),
        gaussKronrod(new GslQuadratureAdaptiveGaussKronrod()) {
    assert(covDiag != nullptr);
    assert(covDiag->size == N);
    assert(bMax > 0);
    squaredEuclideanDistanceUtilLL =
        new SquaredEuclideanDistance_LL_vectorized(L, N);
    vecN = gsl_vector_alloc(N);
    integrationParams = new GMToDiracIntegrationParams(L, 200);
  }

  /**
   * @brief Destroy the GMToDiracBaseOptimizationParams object
   *
   */
  ~GMToDiracBaseOptimizationParams() {
    if (vecN) gsl_vector_free(vecN);
    if (squaredEuclideanDistanceUtilLL) delete squaredEuclideanDistanceUtilLL;
    if (integrationParams) delete integrationParams;
  }

  const gsl_vector* covDiag;
  const gsl_vector* covDiagSqrd;
  const size_t bMax;
  const double cB;
  const double twoPiNHalf;
  const double D1;
  const GslQuadratureAdaptiveGaussKronrod* gaussKronrod;
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL;
  gsl_vector* vecN;
  GMToDiracIntegrationParams* integrationParams;

 private:
  static gsl_vector* createCovDiagSqrd(const gsl_vector* covDiag) {
    gsl_vector* covDiagSqrd = gsl_vector_alloc(covDiag->size);
    for (size_t i = 0; i < covDiag->size; ++i) {
      covDiagSqrd->data[i] = covDiag->data[i] * covDiag->data[i];
    }
    return covDiagSqrd;
  }
  struct P1IntegrationParams {
    const size_t N;
    const gsl_vector* covDiag;
  };
  static double calculateP1(double b, void* params) {
    P1IntegrationParams* p1Params = static_cast<P1IntegrationParams*>(params);
    const size_t N = p1Params->N;
    const gsl_vector* covDiag = p1Params->covDiag;
    double sum = gsl_sf_pow_int(b, (int)(N + 1));
    for (size_t k = 0; k < N; k++) {
      sum *=
          1.00 / std::sqrt(covDiag->data[k] * covDiag->data[k] + 2.00 * b * b);
    }
    return sum;
  }
  static double calculateD1(size_t bMax, const gsl_vector* covDiag, size_t N) {
    gsl_integration_workspace* workspace =
        gsl_integration_workspace_alloc((size_t)1000);
    double result = 0.00;
    double abserr = 0.00;
    P1IntegrationParams p1Params = P1IntegrationParams{N, covDiag};
    gsl_function F;
    F.function = calculateP1;
    F.params = &p1Params;
    int status =
        gsl_integration_qag(&F, 0.00, (double)bMax, 0.00, 1e-10, 1000,
                            GSL_INTEG_GAUSS31, workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    if (status != GSL_SUCCESS) return 0.00;
    return result;
  }

  friend class benchmark_gm_to_dirac_short;
};

/**
 * @brief optimization parameters for the GMToDirac approximation with constant
 * weights
 *
 */
struct GMToDiracConstWeightOptimizationParams
    : public GMToDiracBaseOptimizationParams {
 public:
  /**
   * @brief Construct a new GMToDiracConstWeightOptimizationParams object
   *
   * uses custom weights wX
   *
   * @param covDiag covariance matrix diagonal
   * @param wX weights for the Dirac approximation
   * @param N dimension of the data
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   */
  GMToDiracConstWeightOptimizationParams(const gsl_vector* covDiag,
                                         const gsl_vector* wX, size_t N,
                                         size_t L, size_t bMax, double cB)
      : GMToDiracBaseOptimizationParams(covDiag, N, L, bMax, cB),
        wX(wX),
        freeWeight(false) {
    assert(covDiag != nullptr);
    assert(covDiag->size == N);
  }

  /**
   * @brief Construct a new GMToDiracConstWeightOptimizationParams object
   *
   * automatically creates weights wX = 1.00 / L
   *
   * @param covDiag covariance matrix diagonal
   * @param N dimension of the data
   * @param L number of data points for apprioximation
   * @param bMax bMax
   * @param cB cB
   */
  GMToDiracConstWeightOptimizationParams(const gsl_vector* covDiag, size_t N,
                                         size_t L, size_t bMax, double cB)
      : GMToDiracBaseOptimizationParams(covDiag, N, L, bMax, cB),
        wX(getConstWeight(L)),
        freeWeight(true) {
    assert(covDiag != nullptr);
    assert(covDiag->size == N);
  }

  /**
   * @brief Destroy the GMToDiracConstWeightOptimizationParams object
   *
   */
  ~GMToDiracConstWeightOptimizationParams() {
    if (freeWeight) gsl_vector_free(const_cast<gsl_vector*>(wX));
  }

  const gsl_vector* wX;

 private:
  const bool freeWeight;

  static gsl_vector* getConstWeight(size_t size) {
    gsl_vector* constWeight = gsl_vector_alloc(size);
    gsl_vector_set_all(constWeight, (1.00 / static_cast<double>(size)));
    return constWeight;
  }
};

#endif  // GM_TO_DIRAC_OPTIMIZATION_PARAMS_H