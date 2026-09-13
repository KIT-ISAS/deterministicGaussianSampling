#ifndef GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP
#define GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP

/**
 * @brief check the preconditions of the even-N closed-form path
 *
 * @param L number of Dirac components
 * @param N dimension
 * @param bMax upper bound of the b-integral
 * @return true if the closed form applies
 */
template <typename T>
inline bool gm_to_dirac_even_closed_form<T>::preconditionsHold(size_t L,
                                                               size_t N,
                                                               double bMax) {
  assert(N % 2 == 0);
  assert(N >= 2);
  assert(bMax > 0.00);
  assert(L >= 1);

  return N >= 2 && N % 2 == 0 && bMax > 0.00 && L >= 1;
}

/**
 * @brief attraction term: contributes -2 * D2 to f and its gradient to grad
 *
 * f += -2 * pow(2,k) * sum_i w_i * dBkk(k, bMax, c_i)
 *
 * grad += pow(2,k+1) * w_q * s_q * dBkk1(k, bMax, c_q)
 *
 * @param x sample locations
 * @param params optimization parameters, caches must be up to date
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_closed_form<T>::calculateD2(
    const gsl_vector* x, GMToDiracEvenOptimizationParams* params, double* f,
    gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const size_t k = params->k;
  const double bMax = params->bMax;
  const gsl_vector* wX = params->wX;

  if (f) {
    double sum = 0.00;
    for (size_t i = 0; i < L; ++i)
      sum += wX->data[i] * lcd_delta_bkk_reduced(k, bMax, params->cSqrdNorm[i]);

    *f += -2.00 * std::ldexp(sum, static_cast<int>(k));
  }

  if (grad) {
    for (size_t q = 0; q < L; ++q) {
      const double factor =
          std::ldexp(wX->data[q] * lcd_delta_bkk1(k, bMax, params->cSqrdNorm[q]),
                     static_cast<int>(k) + 1);
      for (size_t d = 0; d < N; ++d)
        grad->data[q * N + d] += factor * x->data[q * N + d];
    }
  }
}

/**
 * @brief repulsion term: contributes D3 to f and its gradient to grad
 *
 * f    += sum_{i,j} w_i * w_j * (C(bMax, T_ij) - bMax^2/2)
 *
 * grad += 0.5 * w_q * sum_{i != q} w_i * (s_q - s_i) * Ei(-T_qi/(4 bMax^2))
 *
 * @param x sample locations
 * @param params optimization parameters
 * @param f objective accumulator, may be nullptr
 * @param grad gradient accumulator, may be nullptr
 */
template <typename T>
inline void gm_to_dirac_even_closed_form<T>::calculateD3(
    const gsl_vector* x, GMToDiracEvenOptimizationParams* params, double* f,
    gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const double bMax = params->bMax;
  const gsl_vector* wX = params->wX;

  const double fourBMaxSqrd = 4.00 * bMax * bMax;
  double d3 = 0.00;

  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];

    for (size_t j = 0; j < L; ++j) {
      const double wXiwXj = wXi * wX->data[j];
      const double localDistSq = params->distanceSq(i, j);

      // coincident samples
      if (localDistSq <= 0.00) continue;

      const double z = -localDistSq / fourBMaxSqrd;
      const double ei = lcd_ei(z);

      if (f)
        d3 += wXiwXj * (0.50 * bMax * bMax * std::expm1(z) +
                        0.125 * localDistSq * ei);

      if (!grad) continue;

      const double constFactor = 0.50 * wXiwXj * ei;
      for (size_t d = 0; d < N; ++d)
        grad->data[i * N + d] +=
            constFactor * (x->data[i * N + d] - x->data[j * N + d]);
    }
  }

  if (f) *f += d3;
}

template <typename T>
inline void gm_to_dirac_even_closed_form<T>::correctMean(gsl_vector* x,
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

#endif  // GM_TO_DIRAC_EVEN_CLOSED_FORM_TPP
