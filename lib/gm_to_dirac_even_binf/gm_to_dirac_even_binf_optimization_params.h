#ifndef GM_TO_DIRAC_EVEN_BINF_OPTIMIZATION_PARAMS_H
#define GM_TO_DIRAC_EVEN_BINF_OPTIMIZATION_PARAMS_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <cassert>
#include <vector>

#include "gsl_minimizer.h"
#include "lcd_even_binf_closed_form.h"
#include "squared_euclidean_distance_utils.h"

/**
 * @brief optimization parameters for the bMax-free closed-form even-N Gaussian-to-Dirac approximation
 *
 * Mirrors GMToDiracEvenOptimizationParams minus bMax and everything derived from it:
 * there is no reportedOffset and no objectiveOffset, because the bMax-free objective carries
 * its only constant, lcd_binf_constant(k), directly.
 *
 * It holds one thing the finite-bMax struct does not: the PROJECTED sample buffer.
 * The minimizer's variable is t, but the distance is a function of
 *
 *     s_i = t_i - sum_j w_j t_j
 *
 * so c_i and T_ij must be computed from s, never from t. update() performs
 * that projection and then refreshes both caches from the result.
 */
struct GMToDiracEvenBinfOptimizationParams
    : public GslMinimizerOptimizationParams {
 public:
  /**
   * @brief Construct a new GMToDiracEvenBinfOptimizationParams object
   *
   * @param wX weights of the Dirac mixture, must be non-null and of size L
   * @param N dimension of the data, must be even and >= 2
   * @param L number of Dirac components, must be >= 1
   */
  GMToDiracEvenBinfOptimizationParams(const gsl_vector* wX, size_t N, size_t L)
      : GslMinimizerOptimizationParams(L, N),
        wX(wX),
        k(N / 2),
        constant(lcd_binf_constant(N / 2)),
        sProjected(L * N, 0.00),
        cSqrdNorm(L, 0.00) {
    assert(wX != nullptr);
    assert(wX->size == L);
    assert(N % 2 == 0);
    assert(N >= 2);
    assert(L >= 1);

    if (L > 1)
      squaredEuclideanDistanceUtilLL =
          new SquaredEuclideanDistance_LL_vectorized(L, N);
  }

  /**
   * @brief Destroy the GMToDiracEvenBinfOptimizationParams object
   */
  ~GMToDiracEvenBinfOptimizationParams() {
    if (squaredEuclideanDistanceUtilLL) delete squaredEuclideanDistanceUtilLL;
  }

  /**
   * @brief project t onto the zero-mean manifold and refresh the caches
   *
   * Writes s_i = t_i - sum_j w_j t_j into sProjected, then fills cSqrdNorm
   * and the pairwise distances from s. Must be called whenever t changes,
   * before reading sample(), cSqrdNorm or distanceSq().
   *
   * @param t optimizer variable, L * N
   */
  inline void update(const gsl_vector* t) {
    assert(t->size == L * N);

    // s = t - 1 * (w^T t)
    std::vector<double> mean(N, 0.00);
    for (size_t i = 0; i < L; ++i) {
      const double wXi = wX->data[i];
      for (size_t d = 0; d < N; ++d) mean[d] += wXi * t->data[i * N + d];
    }

    for (size_t i = 0; i < L; ++i) {
      double sum = 0.00;
      for (size_t d = 0; d < N; ++d) {
        const double sid = t->data[i * N + d] - mean[d];
        sProjected[i * N + d] = sid;
        sum += sid * sid;
      }
      cSqrdNorm[i] = sum;
    }

    if (!squaredEuclideanDistanceUtilLL) return;

    const gsl_matrix sMatrix =
        gsl_matrix_view_array(sProjected.data(), L, N).matrix;
    squaredEuclideanDistanceUtilLL->calculateDistance(&sMatrix, nullptr);
  }

  /**
   * @brief true if any projected sample sits exactly on the origin
   *
   * c_i == 0 is outside the domain of lcd_binf_phi(): it reaches lcd_ei(0),
   * which aborts through GSL's default error handler.
   *
   * Reads the cache, so update() must have run.
   *
   * @return true if the closed form cannot be evaluated for this sample set
   */
  inline bool hasDegenerateSample() const {
    for (size_t i = 0; i < L; ++i)
      if (!(cSqrdNorm[i] > 0.00)) return true;
    return false;
  }

  /**
   * @brief cached projected sample coordinate s_{i,d}
   *
   * @param i index of the sample
   * @param d index of the dimension
   * @return s_{i,d}
   */
  inline double sample(size_t i, size_t d) const {
    return sProjected[i * N + d];
  }

  /**
   * @brief cached squared distance T_ij between projected samples i and j
   *
   * The projection is a common translation of every sample, so T_ij is the
   * same for t and for s; it is computed from s regardless, to keep one
   * source of truth.
   *
   * @param i index of the first sample
   * @param j index of the second sample
   * @return ||s_i - s_j||^2
   */
  inline double distanceSq(size_t i, size_t j) const {
    if (!squaredEuclideanDistanceUtilLL) return 0.00;  // L == 1
    return squaredEuclideanDistanceUtilLL->getDistance(i, j);
  }

  const gsl_vector* wX;
  const size_t k;         ///< N / 2
  const double constant;  ///< K_k, see lcd_binf_constant()

  std::vector<double> sProjected;  ///< s = t - 1 * (w^T t), size L * N
  std::vector<double> cSqrdNorm;   ///< c_i = ||s_i||^2, size L
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL = nullptr;
};

#endif  // GM_TO_DIRAC_EVEN_BINF_OPTIMIZATION_PARAMS_H
