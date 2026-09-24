#ifndef GM_TO_DIRAC_EVEN_BINF_TPP
#define GM_TO_DIRAC_EVEN_BINF_TPP

#include <cassert>
#include <cmath>
#include <vector>

/**
 * @brief check the preconditions of the bMax-free even-N closed-form path
 *
 * There is no bMax to validate; what remains is the parity of N and the
 * component count.
 *
 * @param L number of Dirac components
 * @param N dimension
 * @return true if the closed form applies
 */
template <typename T>
inline bool gm_to_dirac_even_binf<T>::preconditionsHold(size_t L, size_t N) {
  assert(N % 2 == 0);
  assert(N >= 2);
  assert(L >= 1);

  return N >= 2 && N % 2 == 0 && L >= 1;
}

/**
 * @brief attraction term: contributes K_k - 2 * sum_i w_i phi_k(c_i) to f and
 * its gradient to grad
 *
 * f    += K_k - 2 * sum_i w_i * phi_k(c_i)
 *
 * grad += -4 * w_q * phi_k'(c_q) * s_q
 *
 * The factor 4 rather than 2 comes from dc_q / ds_q = 2 s_q.
 *
 * @param params optimization parameters
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator with respect to s, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_binf<T>::calculateAttraction(
    GMToDiracEvenBinfOptimizationParams* params, double* f, gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const size_t k = params->k;
  const gsl_vector* wX = params->wX;

  if (f) {
    double sum = 0.00;
    for (size_t i = 0; i < L; ++i)
      sum += wX->data[i] * lcd_binf_phi(k, params->cSqrdNorm[i]);

    *f += params->constant - 2.00 * sum;
  }

  if (grad) {
    for (size_t q = 0; q < L; ++q) {
      const double factor =
          -4.00 * wX->data[q] * lcd_binf_phi_deriv(k, params->cSqrdNorm[q]);
      for (size_t d = 0; d < N; ++d)
        grad->data[q * N + d] += factor * params->sample(q, d);
    }
  }
}

/**
 * @brief repulsion term: contributes the T ln T double sum to f and its
 * gradient to grad
 *
 * f    += sum_{i,j, T_ij > 0} w_i * w_j * (T_ij / 8) * ln(T_ij / 4)
 *
 * grad += 0.5 * w_q * sum_{j != q, T_qj > 0} w_j * (s_q - s_j)
 *                                          * (ln(T_qj / 4) + 1)
 *
 * The 1/2 rather than 1/4 is because the double sum reaches index q both as
 * i and as j. Coincident samples (T <= 0) are skipped, exactly as
 * gm_to_dirac_even_closed_form::calculateD3 does; the term and its gradient
 * both tend to zero there.
 *
 * @param params optimization parameters, caches must be up to date
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator with respect to s, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_binf<T>::calculateRepulsion(
    GMToDiracEvenBinfOptimizationParams* params, double* f, gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const gsl_vector* wX = params->wX;

  double repulsion = 0.00;

  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];

    for (size_t j = 0; j < L; ++j) {
      const double localDistSq = params->distanceSq(i, j);

      // coincident samples
      if (localDistSq <= 0.00) continue;

      const double wXiwXj = wXi * wX->data[j];
      const double logTerm = std::log(0.25 * localDistSq);

      if (f) repulsion += wXiwXj * 0.125 * localDistSq * logTerm;

      if (!grad) continue;

      const double constFactor = 0.50 * wXiwXj * (logTerm + 1.00);
      for (size_t d = 0; d < N; ++d)
        grad->data[i * N + d] +=
            constFactor * (params->sample(i, d) - params->sample(j, d));
    }
  }

  if (f) *f += repulsion;
}

/**
 * @brief chain-rule the gradient back through the zero-mean projection
 *
 * With s = t - 1 * (w^T t),
 *
 *     grad_{t,q} = grad_{s,q} - w_q * sum_i grad_{s,i}
 *
 * Applied in place to the assembled ds-gradient.
 *
 * @param params optimization parameters
 * @param grad gradient with respect to s on entry, with respect to t on exit
 */
template <typename T>
inline void gm_to_dirac_even_binf<T>::projectGradient(
    const GMToDiracEvenBinfOptimizationParams* params, gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const gsl_vector* wX = params->wX;

  std::vector<double> total(N, 0.00);
  for (size_t i = 0; i < L; ++i)
    for (size_t d = 0; d < N; ++d) total[d] += grad->data[i * N + d];

  for (size_t q = 0; q < L; ++q) {
    const double wXq = wX->data[q];
    for (size_t d = 0; d < N; ++d) grad->data[q * N + d] -= wXq * total[d];
  }
}

/**
 * @brief subtract the weighted mean from every sample, in place
 *
 * This is exactly the projection s = t - 1 * (w^T t) that every evaluation
 * applies internally, so calling it once after minimize() turns the returned
 * optimizer variable t into the sample set s the objective was actually evaluated at.
 *
 * @param x sample locations (L * N), modified in place
 * @param wX weights of the Dirac mixture
 * @param L number of Dirac components
 * @param N dimension
 */
template <typename T>
inline void gm_to_dirac_even_binf<T>::correctMean(gsl_vector* x,
                                                  const gsl_vector* wX,
                                                  size_t L, size_t N) {
  std::vector<double> mean(N, 0.00);
  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];
    for (size_t d = 0; d < N; ++d) mean[d] += wXi * x->data[i * N + d];
  }
  for (size_t i = 0; i < L; ++i) {
    for (size_t d = 0; d < N; ++d) x->data[i * N + d] -= mean[d];
  }
}

#endif  // GM_TO_DIRAC_EVEN_BINF_TPP
