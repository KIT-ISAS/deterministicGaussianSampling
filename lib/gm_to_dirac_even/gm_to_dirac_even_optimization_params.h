#ifndef GM_TO_DIRAC_EVEN_OPTIMIZATION_PARAMS_H
#define GM_TO_DIRAC_EVEN_OPTIMIZATION_PARAMS_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <cassert>
#include <vector>

#include "gsl_minimizer.h"
#include "lcd_even_closed_form.h"
#include "squared_euclidean_distance_utils.h"

/**
 * @brief optimization parameters for the closed-form even-N Gaussian-to-Dirac
 * approximation
 */
struct GMToDiracEvenOptimizationParams : public GslMinimizerOptimizationParams {
 public:
  /**
   * @brief Construct a new GMToDiracEvenOptimizationParams object
   *
   * @param wX weights of the Dirac mixture, must be non-null and of size L
   * @param N dimension of the data, must be even and >= 2
   * @param L number of Dirac components, must be >= 1
   * @param bMax upper bound of the b-integral, must be > 0
   * @param includeD1 true to carry the sample-independent constant in the
   * objective (see gm_to_dirac_even_closed_form::includeD1InObjective)
   */
  GMToDiracEvenOptimizationParams(const gsl_vector* wX, size_t N, size_t L,
                                  double bMax, bool includeD1)
      : GslMinimizerOptimizationParams(L, N),
        wX(wX),
        bMax(bMax),
        k(N / 2),
        reportedOffset(
            lcd_reported_offset(N, bMax, weightSumOf(wX, L))),
        objectiveOffset(includeD1 ? reportedOffset : 0.00),
        cSqrdNorm(L, 0.00) {
    assert(wX != nullptr);
    assert(wX->size == L);
    assert(N % 2 == 0);
    assert(N >= 2);
    assert(bMax > 0.00);
    assert(L >= 1);

    if (L > 1)
      squaredEuclideanDistanceUtilLL =
          new SquaredEuclideanDistance_LL_vectorized(L, N);
  }

  /**
   * @brief Destroy the GMToDiracEvenOptimizationParams object
   */
  ~GMToDiracEvenOptimizationParams() {
    if (squaredEuclideanDistanceUtilLL) delete squaredEuclideanDistanceUtilLL;
  }

  /**
   * @brief refresh the per-evaluation caches for a new x
   *
   * Must be called whenever x changes, before reading cSqrdNorm or
   * distanceSq().
   *
   * @param x sample locations, L * N
   */
  inline void update(const gsl_vector* x) {
    assert(x->size == L * N);

    for (size_t i = 0; i < L; ++i) {
      double sum = 0.00;
      for (size_t d = 0; d < N; ++d) {
        const double xid = x->data[i * N + d];
        sum += xid * xid;
      }
      cSqrdNorm[i] = sum;
    }

    if (!squaredEuclideanDistanceUtilLL) return;

    const gsl_matrix xMatrix = gsl_matrix_view_array(x->data, L, N).matrix;
    squaredEuclideanDistanceUtilLL->calculateDistance(&xMatrix, nullptr);
  }

  /**
   * @brief cached squared distance T_ij between samples i and j
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
  const double bMax;
  const size_t k;  ///< N / 2

  /// constant turning the reduced objective into the true distance:
  /// distance = f_reduced + reportedOffset. See lcd_reported_offset().
  const double reportedOffset;

  /// what the minimizer's objective actually carries: reportedOffset, or 0
  /// if the caller opted out. Adding a constant costs no x-dependent
  /// precision.
  const double objectiveOffset;

  std::vector<double> cSqrdNorm;   ///< c_i = ||s_i||^2, size L
  SquaredEuclideanDistanceUtilsLL* squaredEuclideanDistanceUtilLL = nullptr;

 private:
  static double weightSumOf(const gsl_vector* wX, size_t L) {
    double sum = 0.00;
    for (size_t i = 0; i < L; ++i) sum += wX->data[i];
    return sum;
  }
};

#endif  // GM_TO_DIRAC_EVEN_OPTIMIZATION_PARAMS_H
