#ifndef LCD_EVEN_CLOSED_FORM_H
#define LCD_EVEN_CLOSED_FORM_H

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_expint.h>

#include <cassert>
#include <cmath>
#include <cstddef>

/**
 * @file lcd_even_closed_form.h
 * @brief quadrature-free closed forms for the LCD / modified Cramer-von-Mises
 * distance of an even-dimensional standard normal target
 *
 * The mCvM distance of a Dirac mixture with samples s_i and weights w_i
 * against N(0, I) decomposes into
 *
 *     D = D1 - 2 * D2 + D3
 *
 * where D1 is sample-independent, D2 is the attraction term (samples towards
 * the mean) and D3 is the repulsion term (samples away from each other).
 *
 * @note Preconditions for the D2 helpers (lcd_delta_bkk, lcd_delta_bkk1):
 * the target must be the STANDARD normal N(0, I) and N = 2k must be EVEN.
 * lcd_a_n and lcd_c_repulsion are valid for all N.
 */

/**
 * @brief exponential integral Ei(x) for negative arguments
 *
 * @param x argument, must be strictly negative
 * @return Ei(x), or exactly 0.0 in case of underflow, domain error or overflow
 */
inline double lcd_ei(double x) {
  assert(x < 0.00);

  gsl_sf_result result;
  const int status = gsl_sf_expint_Ei_e(x, &result);

  if (status == GSL_EUNDRFLW || status == GSL_EDOM || status == GSL_EOVRFLW) return 0.00;
  assert(status == GSL_SUCCESS);

  return result.val;
}

/**
 * @brief D1, the sample-independent term of the mCvM distance
 *
 * Evaluates A_N(b) = integral_0^b x^(N+1) / (1 + x^2)^(N/2) dx via the
 * recursion
 *
 *     A_1(b) = 0.5 * (b * sqrt(1 + b^2) - asinh(b))
 * 
 *     A_2(b) = 0.5 * (b^2 - log(1 + b^2))
 * 
 *     A_n(b) = (n * A_{n-2}(b) - b^n / (1 + b^2)^((n-2)/2)) / (n - 2)
 *
 * applied iteratively. Valid for all N, odd and even.
 *
 * @param N dimension, must be >= 1
 * @param b upper integration bound, must be >= 0
 * @return A_N(b)
 */
inline double lcd_a_n(size_t N, double b) {
  assert(N >= 1);
  assert(b >= 0.00);

  const double bSqrd = b * b;

  size_t n = (N % 2 == 0) ? 2 : 1;
  double aN = (N % 2 == 0)
                  ? 0.50 * (bSqrd - std::log1p(bSqrd))
                  : 0.50 * (b * std::sqrt(1.00 + bSqrd) - std::asinh(b));

  for (n += 2; n <= N; n += 2) {
    const double dn = static_cast<double>(n);
    const double tail =
        bSqrd * std::pow(bSqrd / (1.00 + bSqrd), (dn - 2.00) / 2.00);
    aN = (dn * aN - tail) / (dn - 2.00);
  }

  return aN;
}

/**
 * @brief C(b, c), the per-pair repulsion term of D3
 *
 * C(b, c) = integral_0^b x * exp(-c / (4 x^2)) dx
 * 
 *         = (b^2 / 2) * exp(-c / (4 b^2)) + (c / 8) * Ei(-c / (4 b^2))
 *
 * @param b upper integration bound, must be > 0
 * @param c squared distance between the two samples, must be >= 0
 * @return C(b, c); C(b, 0) = b^2 / 2
 */
inline double lcd_c_repulsion(double b, double c) {
  assert(b > 0.00);
  assert(c >= 0.00);

  const double bSqrd = b * b;
  if (c <= 0.00) return 0.50 * bSqrd;  // coincident samples

  const double z = -c / (4.00 * bSqrd);
  return 0.50 * bSqrd * std::exp(z) + 0.125 * c * lcd_ei(z);
}

/**
 * @brief parameters of the raw b-integrand used by the small-c fallback
 */
struct LcdDeltaBIntegrandParams {
  size_t k;      ///< half the dimension
  size_t q;      ///< exponent of (1 + 2b^2); k for dBkk, k+1 for dBkk1
  double cValue; ///< squared norm of the sample
  bool reduced;  ///< true to use expm1, giving dBkk(c) - dBkk(0) directly
};

/**
 * @brief b^(2k+1) * (1 + 2b^2)^(-q) * exp(-c / (2 (1 + 2b^2)))
 *
 * With reduced set, exp is replaced by expm1, so the integral evaluates
 * dBkk(c) - dBkk(0) directly and never forms the O(bMax^2) intermediate -
 * the same trick gm_to_dirac_short::calculateP2 uses.
 *
 * @param b integration variable
 * @param params LcdDeltaBIntegrandParams
 * @return value of the integrand at b
 */
inline double lcd_delta_b_integrand(double b, void* params) {
  const LcdDeltaBIntegrandParams* p =
      static_cast<const LcdDeltaBIntegrandParams*>(params);

  const double u = 1.00 + 2.00 * b * b;
  double value = b * std::pow(b * b / u, static_cast<double>(p->k));
  for (size_t i = p->k; i < p->q; ++i) value /= u;

  const double exponent = -p->cValue / (2.00 * u);
  return value * (p->reduced ? std::expm1(exponent) : std::exp(exponent));
}

/**
 * @brief dB_{0,d}(c) = B_{0,d}(bMax, c) - B_{0,d}(0, c)
 *
 * @param d index of the base case
 * @param u 1 + 2 * bMax^2
 * @param eULeading exp(-c / (2u)), or expm1(-c / (2u)) for the reduced form
 * @param eU exp(-c / (2u))
 * @param e0 exp(-c / 2)
 * @param dEi Ei(-c / (2u)) - Ei(-c / 2)
 * @param c squared norm of the sample, must be > 0
 * @return dB_{0,d}(c), less u/4 for d == 0 if eULeading is the expm1 form
 */
inline double lcd_delta_b0(size_t d, double u, double eULeading, double eU,
                           double e0, double dEi, double c) {
  assert(c > 0.00);

  if (d == 0) return 0.25 * u * eULeading - 0.25 * e0 + 0.125 * c * dEi;
  if (d == 1) return -0.25 * dEi;

  double coefficient = 0.50;
  double cPower = c;      // c^(d-j+1), j = d
  double uPower = 1.00;   // u^(j-2),   j = d ... built downwards below
  for (size_t j = d; j > 2; --j) uPower *= u;

  double sum = 0.00;
  for (size_t j = d;; --j) {
    sum += coefficient * (eU / (cPower * uPower) - e0 / cPower);
    if (j == 2) break;
    coefficient *= static_cast<double>(j - 2) * 2.00;
    cPower *= c;
    uPower /= u;
  }

  return sum;
}

/**
 * @brief dBkk(k, c) = B_{k,k}(bMax, c) - B_{k,k}(0, c), the D2 term
 *
 * D2 = pow(2, k) * sum_i w_i * lcd_delta_bkk(k, bMax, c_i), with
 * c_i = ||s_i||^2.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk(k, c)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk(size_t k, double bMax, double c);

/**
 * @brief dBkk1(k, c) = B_{k,k+1}(bMax, c) - B_{k,k+1}(0, c), the D2 gradient
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk1(k, c)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk1(size_t k, double bMax, double c);

/**
 * @brief closed form of dBkk / dBkk1
 *
 * @param k half the dimension, N = 2k
 * @param bMax upper integration bound
 * @param c squared norm of the sample, must be > 0
 * @param shift 0 for dBkk (uses dB_{0,j}), 1 for dBkk1 (uses dB_{0,j+1})
 * @return pow(2,-k) * sum_{j=0..k} (-1)^j * binom(k,j) * dB_{0,j+shift}(c)
 */
inline double lcd_delta_bkk_closed(size_t k, double bMax, double c,
                                   size_t shift, bool reduced = false) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c > 0.00);
  assert(!reduced || shift == 0);  // no reduction is defined for dBkk1

  const double u = 1.00 + 2.00 * bMax * bMax;
  const double exponent = -c / (2.00 * u);
  const double eU = std::exp(exponent);
  const double eULeading = reduced ? std::expm1(exponent) : eU;
  const double e0 = std::exp(-c / 2.00);
  const double dEi = lcd_ei(exponent) - lcd_ei(-c / 2.00);

  double sum = 0.00;
  double binomial = 1.00;  // binom(k, j)
  for (size_t j = 0; j <= k; ++j) {
    const double term =
        lcd_delta_b0(j + shift, u, eULeading, eU, e0, dEi, c);
    sum += (j % 2 == 0) ? binomial * term : -binomial * term;
    binomial *= static_cast<double>(k - j) / static_cast<double>(j + 1);
  }

  return std::ldexp(sum, -static_cast<int>(k));
}

/**
 * @brief dBkk(k, 0) = integral{0 to bMax} b^(2k+1) / (1 + 2b^2)^k db
 *
 * The c = 0 value of the attraction term, in elementary closed form. With
 * u = 1 + 2b^2 the substitution gives
 *
 *   Z = 2^-(k+2) * integral{1 to U} (1 - 1/u)^k du
 * 
 *     = 2^-(k+2) * [ (U-1) - k*ln(U)
 *                    + sum_{j=2..k} (-1)^j C(k,j) (U^(1-j) - 1)/(1-j) ]
 *
 * This is the constant lcd_delta_bkk_reduced() removes. It is O(bMax^2) by
 * construction, so it belongs in the reported
 * offset, never inside the optimizer's objective.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @return dBkk(k, 0)
 */
inline double lcd_delta_bkk_zero(size_t k, double bMax) {
  assert(k >= 1);
  assert(bMax > 0.00);

  const double u = 1.00 + 2.00 * bMax * bMax;

  double sum = (u - 1.00) - static_cast<double>(k) * std::log(u);

  double binomial = static_cast<double>(k);  // binom(k, 1), stepped up below
  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double dj = static_cast<double>(j);
    const double term =
        binomial * (std::pow(u, 1.00 - dj) - 1.00) / (1.00 - dj);
    sum += (j % 2 == 0) ? term : -term;
  }

  return std::ldexp(sum, -static_cast<int>(k) - 2);
}

/**
 * @brief lcd_delta_bkk_zero(k, bMax) - pow(2, -k) * U / 4
 *
 * The same expression with (U - 1) replaced by -1, i.e. with the O(bMax^2)
 * leading term removed. The result is O(ln U), so subtracting it from the
 * expm1 form of the closed dBkk introduces no cancellation.
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @return dBkk(k, 0) - pow(2, -k) * (1 + 2 bMax^2) / 4
 */
inline double lcd_delta_bkk_zero_minus_leading(size_t k, double bMax) {
  assert(k >= 1);
  assert(bMax > 0.00);

  const double u = 1.00 + 2.00 * bMax * bMax;

  double sum = -1.00 - static_cast<double>(k) * std::log(u);

  double binomial = static_cast<double>(k);
  for (size_t j = 2; j <= k; ++j) {
    binomial *= static_cast<double>(k - j + 1) / static_cast<double>(j);
    const double dj = static_cast<double>(j);
    const double term =
        binomial * (std::pow(u, 1.00 - dj) - 1.00) / (1.00 - dj);
    sum += (j % 2 == 0) ? term : -term;
  }

  return std::ldexp(sum, -static_cast<int>(k) - 2);
}

/**
 * @brief dBkk(k, c) - dBkk(k, 0), the cancellation-free attraction term
 *
 * Equal to integral{0 to bMax} b^(2k+1)/(1+2b^2)^k * expm1(-c/(2(1+2b^2))) db.
 *
 * The constant removed per sample is lcd_delta_bkk_zero(k, bMax); D2 as a
 * whole loses pow(2, k) * (sum_i w_i) * lcd_delta_bkk_zero(k, bMax).
 *
 * @param k half the dimension, N = 2k; must be >= 1
 * @param bMax upper integration bound, must be > 0
 * @param c squared norm of the sample, must be >= 0
 * @return dBkk(k, c) - dBkk(k, 0)
 * @note standard-normal target and even N only
 */
inline double lcd_delta_bkk_reduced(size_t k, double bMax, double c) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c >= 0.00);

  if (c <= 0.00) return 0.00;  // dBkk(0) - dBkk(0)

  // the expm1 form has already dropped pow(2,-k) * U/4; take off what is left
  // of dBkk(k, 0), which is only O(ln U)
  return lcd_delta_bkk_closed(k, bMax, c, 0, true) -
         lcd_delta_bkk_zero_minus_leading(k, bMax);
}

/**
 * @brief shared body of lcd_delta_bkk / lcd_delta_bkk1
 *
 * @param k half the dimension, N = 2k
 * @param bMax upper integration bound
 * @param c squared norm of the sample
 * @param shift 0 for dBkk (uses dB_{0,j}), 1 for dBkk1 (uses dB_{0,j+1})
 * @return dBkk (shift 0) or dBkk1 (shift 1)
 */
inline double lcd_delta_bkk_impl(size_t k, double bMax, double c,
                                 size_t shift) {
  assert(k >= 1);
  assert(bMax > 0.00);
  assert(c >= 0.00);

  return lcd_delta_bkk_closed(k, bMax, c, shift);
}

inline double lcd_delta_bkk(size_t k, double bMax, double c) {
  return lcd_delta_bkk_impl(k, bMax, c, 0);
}

inline double lcd_delta_bkk1(size_t k, double bMax, double c) {
  return lcd_delta_bkk_impl(k, bMax, c, 1);
}


/**
 * @brief the constant that turns the reduced objective back into the true
 * distance
 *
 * The optimizer minimises a reduced objective in which every O(bMax^2) piece
 * has been removed analytically:
 *
 *   f_reduced = -2 * pow(2,k) * sum_i w_i * lcd_delta_bkk_reduced(k, bMax, c_i)
 *             + sum_ij w_i w_j * C(bMax, T_ij)
 *
 * and the true distance is D = f_reduced + K, with
 *
 *   K = A_N(bMax)
 *     - pow(2, k+1) * W * lcd_delta_bkk_zero(k, bMax)
 *     + W^2 * bMax^2 / 2,          W = sum_i w_i
 *
 * collecting the D1 term, the per-sample attraction constant and the
 * per-pair repulsion constant respectively.
 *
 * @param N dimension, must be even and >= 2
 * @param bMax upper integration bound, must be > 0
 * @param weightSum sum of the Dirac weights, normally 1
 * @return K
 */
inline double lcd_reported_offset(size_t N, double bMax, double weightSum) {
  assert(N >= 2);
  assert(N % 2 == 0);
  assert(bMax > 0.00);

  const size_t k = N / 2;
  const double bMaxSqrd = bMax * bMax;

  if (N == 2) {
    // K = W^2 bMax^2/2 + bMax^2/2 - W bMax^2 - ln(1+bMax^2)/2
    //     + W ln(1+2bMax^2)/2
    //
    // Two rearrangements:
    //  - the three bMax^2 terms collapse to (1-W)^2 bMax^2 / 2;
    //  - W ln(1+2b^2)/2 - ln(1+b^2)/2 is split as
    //    W/2 * ln((1+2b^2)/(1+b^2)) + (W-1)/2 * ln(1+b^2), whose first factor
    //    is log1p(b^2/(1+b^2)).
    const double weightDefect = 1.00 - weightSum;
    return 0.50 * bMaxSqrd * weightDefect * weightDefect +
           0.50 * weightSum * std::log1p(bMaxSqrd / (1.00 + bMaxSqrd)) -
           0.50 * weightDefect * std::log1p(bMaxSqrd);
  }

  return lcd_a_n(N, bMax) -
         std::ldexp(weightSum * lcd_delta_bkk_zero(k, bMax),
                    static_cast<int>(k) + 1) +
         0.50 * weightSum * weightSum * bMaxSqrd;
}

#endif  // LCD_EVEN_CLOSED_FORM_H
