#ifndef LCD_EVEN_BINF_CLOSED_FORM_H
#define LCD_EVEN_BINF_CLOSED_FORM_H

#include <cassert>
#include <cmath>
#include <cstddef>

#include "lcd_even_closed_form.h"  // lcd_ei()

/**
 * @file lcd_even_binf_closed_form.h
 * @brief bMax-free closed forms for the LCD / modified Cramer-von-Mises distance of
 * an even-dimensional standard normal target
 *
 * lcd_even_closed_form.h evaluates the mCvM distance for a finite integration bound bMax.
 * This header takes the limit bMax -> infinity analytically, so bMax disappears from the problem entirely.
 *
 * With N = 2k, samples s_i, weights w_i summing to 1, c_i = ||s_i||^2 and
 * T_ij = ||s_i - s_j||^2 the limit distance is
 *
 *     D = K_k - 2 * sum_i w_i * phi_k(c_i)
 *             + sum_{i,j, T_ij > 0} w_i w_j * (T_ij / 8) * ln(T_ij / 4)
 *
 * @note PRECONDITION: the limit exists only for a ZERO-MEAN sample set, sum_i w_i s_i = 0.
 * For a non-zero mean the true distance diverges like ||mu||^2 / 4 * ln(bMax^2);
 * the formulas here still return a finite number in that case, but that number is NOT the distance.
 * gm_to_dirac_even_binf removes the hazard by optimising on the zero-mean manifold and projecting
 * before every evaluation.
 *
 * @note PRECONDITION: the target must be the STANDARD normal N(0, I) and
 * N = 2k must be EVEN.
 *
 * @warning c == 0 IS A DOMAIN ERROR, not merely inaccurate: it reaches lcd_ei(0),
 * whose GSL_EDOM result GSL's default error handler turns into an
 * abort long before lcd_ei() can map it to 0.0. Every entry point here
 * asserts c > 0, and gm_to_dirac_even_binf refuses a sample set that projects
 * any component onto the origin.
 */

/// Euler-Mascheroni constant
constexpr double lcdBinfEulerGamma = 0.5772156649015328606;

/**
 * @brief B_{0,d}(0, c), the lower-limit base case of the attraction term
 *
 * With e0 = exp(-c/2) and Ei0 = Ei(-c/2),
 *
 *     B_{0,0}(0,c) = e0/4 + (c/8) * Ei0
 *
 *     B_{0,1}(0,c) = -Ei0/4
 *
 *     B_{0,d}(0,c) = e0 * sum_{m=2..d} (d-2)! * 2^(d-m-1)
 *                                      / ( (m-2)! * c^(d-m+1) )   for d >= 2
 *
 * The d >= 2 sum is accumulated downwards from m = d, where the term is
 * 1 / (2c), using term_{m-1} = term_m * 2 (m - 2) / c. That is the same
 * recurrence lcd_delta_b0() uses and it keeps both (d-2)! and c^(d-1) out of
 * the intermediate values.
 *
 * @param d index of the base case
 * @param e0 exp(-c / 2)
 * @param ei0 Ei(-c / 2)
 * @param c squared norm of the sample, must be > 0
 * @return B_{0,d}(0, c)
 */
inline double lcd_binf_b0_zero(size_t d, double e0, double ei0, double c) {
  assert(c > 0.00);

  if (d == 0) return 0.25 * e0 + 0.125 * c * ei0;
  if (d == 1) return -0.25 * ei0;

  double term = 0.50 / c;  // m = d
  double sum = term;
  for (size_t m = d; m > 2; --m) {
    term *= 2.00 * static_cast<double>(m - 2) / c;  // step down to m - 1
    sum += term;
  }

  return e0 * sum;
}

/**
 * @brief d/dc B_{0,d}(0, c)
 *
 * Using d/dc Ei(-c/2) = e0 / c,
 *
 *     dB_{0,0}/dc = Ei0 / 8
 *
 *     dB_{0,1}/dc = -e0 / (4c)
 *
 *     dB_{0,d}/dc = -e0/2 * T_d(c) + e0 * T_d'(c)                 for d >= 2
 *
 * where T_d(c) is the d >= 2 sum of lcd_binf_b0_zero() without the e0 factor
 * and T_d'(c) is its termwise derivative, i.e. term_m scaled by
 * -(d - m + 1) / c. Both sums share the one downward recurrence.
 *
 * @param d index of the base case
 * @param e0 exp(-c / 2)
 * @param ei0 Ei(-c / 2)
 * @param c squared norm of the sample, must be > 0
 * @return d/dc B_{0,d}(0, c)
 */
inline double lcd_binf_b0_zero_deriv(size_t d, double e0, double ei0,
                                     double c) {
  assert(c > 0.00);

  if (d == 0) return 0.125 * ei0;
  if (d == 1) return -0.25 * e0 / c;

  double term = 0.50 / c;      // m = d
  double sum = term;           // T_d(c)
  double sumDeriv = -term / c; // T_d'(c), factor -(d - m + 1) = -1 at m = d
  for (size_t m = d; m > 2; --m) {
    term *= 2.00 * static_cast<double>(m - 2) / c;  // step down to m - 1
    sum += term;
    sumDeriv -= static_cast<double>(d - m + 2) * term / c;
  }

  return e0 * (sumDeriv - 0.50 * sum);
}

/**
 * @brief B_{k,k}(0, c), the lower-limit attraction term
 *
 * B_{k,k}(0,c) = 2^-k * sum_{j=0..k} (-1)^j * binom(k,j) * B_{0,j}(0,c)
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return B_{k,k}(0, c)
 * @note standard-normal target and even N only
 */
inline double lcd_binf_bkk_zero(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  const double e0 = std::exp(-0.50 * c);
  const double ei0 = lcd_ei(-0.50 * c);

  double sum = 0.00;
  double binomial = 1.00;  // binom(k, j)
  for (size_t j = 0; j <= k; ++j) {
    const double term = lcd_binf_b0_zero(j, e0, ei0, c);
    sum += (j % 2 == 0) ? binomial * term : -binomial * term;
    binomial *= static_cast<double>(k - j) / static_cast<double>(j + 1);
  }

  return std::ldexp(sum, -static_cast<int>(k));
}

/**
 * @brief d/dc B_{k,k}(0, c)
 *
 * dB_{k,k}(0,c)/dc = 2^-k * sum_{j=0..k} (-1)^j * binom(k,j)
 *                                              * dB_{0,j}(0,c)/dc
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return d/dc B_{k,k}(0, c)
 * @note standard-normal target and even N only
 */
inline double lcd_binf_bkk_zero_deriv(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  const double e0 = std::exp(-0.50 * c);
  const double ei0 = lcd_ei(-0.50 * c);

  double sum = 0.00;
  double binomial = 1.00;  // binom(k, j)
  for (size_t j = 0; j <= k; ++j) {
    const double term = lcd_binf_b0_zero_deriv(j, e0, ei0, c);
    sum += (j % 2 == 0) ? binomial * term : -binomial * term;
    binomial *= static_cast<double>(k - j) / static_cast<double>(j + 1);
  }

  return std::ldexp(sum, -static_cast<int>(k));
}

/**
 * @brief S_k(c), the pole-cancelling companion of B_{k,k}(0, c)
 *
 * S_k(c) = sum_{j=2..k} (-1)^j * binom(k,j) * (j-2)! * 2^(j-3) / c^(j-1)
 *
 * The sum is empty for k = 1. binom(k,j), (j-2)! * 2^(j-3) and c^(j-1) are
 * each carried forward by their own recurrence, as lcd_delta_bkk_zero() does.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return S_k(c), 0 for k = 1
 */
inline double lcd_binf_s(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  double sum = 0.00;
  double binomial = static_cast<double>(k);  // binom(k, 1), stepped up below
  double coefficient = 0.50;                 // (j-2)! * 2^(j-3), j = 2
  double cPower = c;                         // c^(j-1),          j = 2

  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double term = binomial * coefficient / cPower;
    sum += (j % 2 == 0) ? term : -term;

    coefficient *= 2.00 * static_cast<double>(j - 1);
    cPower *= c;
  }

  return sum;
}

/**
 * @brief S_k'(c)
 *
 * S_k'(c) = sum_{j=2..k} (-1)^j * binom(k,j) * (j-2)! * 2^(j-3)
 *                                            * (1 - j) / c^j
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return S_k'(c), 0 for k = 1
 */
inline double lcd_binf_s_deriv(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  double sum = 0.00;
  double binomial = static_cast<double>(k);  // binom(k, 1), stepped up below
  double coefficient = 0.50;                 // (j-2)! * 2^(j-3), j = 2
  double cPower = c;                         // c^(j-1),          j = 2

  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double term = binomial * coefficient *
                        (1.00 - static_cast<double>(j)) / (cPower * c);
    sum += (j % 2 == 0) ? term : -term;

    coefficient *= 2.00 * static_cast<double>(j - 1);
    cPower *= c;
  }

  return sum;
}

/**
 * @brief phi_k(c), the per-sample attraction potential of the bMax-free
 * distance
 *
 * phi_k(c) = (c + 2k)/8 * ln(c/4) + S_k(c) - 2^k * B_{k,k}(0,c)
 *
 * The distance uses it as D = K_k - 2 * sum_i w_i * phi_k(c_i) + repulsion.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return phi_k(c)
 */
inline double lcd_binf_phi(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  return 0.125 * (c + 2.00 * static_cast<double>(k)) * std::log(0.25 * c) +
         lcd_binf_s(k, c) -
         std::ldexp(lcd_binf_bkk_zero(k, c), static_cast<int>(k));
}

/**
 * @brief phi_k'(c)
 *
 * phi_k'(c) = ln(c/4)/8 + (c + 2k)/(8c) + S_k'(c)
 *                                       - 2^k * dB_{k,k}(0,c)/dc
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param c squared norm of the sample, must be > 0
 * @return phi_k'(c)
 */
inline double lcd_binf_phi_deriv(size_t k, double c) {
  assert(k >= 1);
  assert(c > 0.00);

  return 0.125 * std::log(0.25 * c) +
         (c + 2.00 * static_cast<double>(k)) / (8.00 * c) +
         lcd_binf_s_deriv(k, c) -
         std::ldexp(lcd_binf_bkk_zero_deriv(k, c), static_cast<int>(k));
}

/**
 * @brief K_k, the sample-independent constant of the bMax-free distance
 *
 * K_k = k * sum_{i=2..k} 1/(2i) - 1/2 - k * gamma / 2
 *
 * K_k does not depend on the samples, so it shifts the objective without
 * moving the optimum. It is carried anyway so the reported value is the true
 * distance.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @return K_k
 */
inline double lcd_binf_constant(size_t k) {
  assert(k >= 1);

  double harmonic = 0.00;  // sum_{i=2..k} 1 / (2i)
  for (size_t i = 2; i <= k; ++i) harmonic += 0.50 / static_cast<double>(i);

  return static_cast<double>(k) * harmonic - 0.50 -
         0.50 * static_cast<double>(k) * lcdBinfEulerGamma;
}

#endif  // LCD_EVEN_BINF_CLOSED_FORM_H
