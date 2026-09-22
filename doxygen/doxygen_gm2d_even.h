/**
 * \page page_gm_even gm_to_dirac_even_closed_form
 *
 * Gaussian-to-Dirac approximation, quadrature-free (even dimension).
 *
 * \section gm_even_overview Overview
 *
 * gm_to_dirac_even_closed_form<T> approximates a standard normal
 * distribution by a Dirac mixture with L components in N dimensions,
 * exactly as gm_to_dirac_short_standard_normal_deviation does, but with
 * every integral over the kernel-width variable b evaluated in closed
 * form.
 *
 * The quadrature path issues L × N adaptive Gauss-Kronrod integrations
 * per gradient evaluation. This path issues none.
 *
 * \section gm_even_preconditions Preconditions
 *
 * - N must be EVEN and >= 2. Odd N fails loudly (assert in Debug,
 *   approximate() returns false in Release); there is no silent fallback.
 * - The target must be the standard normal N(0, I), or an isotropic
 *   N(0, sigma² I) through the approximate_isotropic() overloads.
 *   General diagonal covariance with unequal sigma_k is NOT supported:
 *   the scaling reduction that makes the isotropic case work does not
 *   exist there.
 * - bMax may be ANY value > 0. Unlike the quadrature path, whose D3 uses
 *   a large-bMax logarithmic approximation, D3 here is the exact Ei form,
 *   so no "for large bMax" assumption is made anywhere.
 *
 *
 * \section gm_even_differences Differences from the quadrature path
 *
 * Two behavioural differences are worth knowing when comparing results
 * against gm_to_dirac_short_standard_normal_deviation:
 *
 * - The reported distance is the TRUE modified Cramér-von-Mises distance:
 *   it includes the sample-independent D1 term, which the quadrature path
 *   omits.
 * - D3 is exact rather than logarithmically approximated.
 *
 *
 * \section gm_even_interface Interface
 *
 * Inherits:
 *
 * - gm_to_dirac_approx_standard_normal_distribution_i<T>
 *
 * Template parameter:
 *
 * - T ∈ {float, double}; computation is always performed in double and
 *   cast at the interface boundary.
 *
 * Provides overloads of:
 *
 * - approximate(...)
 * - modified_van_mises_distance_sq(...)
 * - modified_van_mises_distance_sq_derivative(...)
 * - approximate_isotropic(...)
 * - modified_van_mises_distance_sq_isotropic(...)
 *
 *
 * \section gm_even_parameters Parameters
 *
 * Common parameters:
 *
 * - L     -> number of Dirac components
 * - N     -> dimension, must be even
 * - bMax  -> integration bound, any value > 0
 * - x     -> initial guess and output locations (L × N)
 * - wX    -> weights of the Dirac mixture (optional)
 *
 * If wX is nullptr:
 *
 * - Uniform weights are assumed
 *
 * \section gm_even_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout:
 *
 * - x represents L × N Dirac locations, row major
 *
 *
 * \section example_gm_even_raw Example (Raw Pointer)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * bool ok = approx.approximate(
 *     L,
 *     N,        // must be even
 *     100.0,    // bMax, any value > 0
 *     x,        // initial guess / output (L × N)
 *     wX,       // weights (optional)
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_gm_even_distance Example (true distance)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * double distance = 0.0;
 * approx.modified_van_mises_distance_sq(&distance, L, N, 100.0, x, wX);
 * // distance includes D1; the quadrature path's value would be short by
 * // lcd_a_n(N, 100.0)
 * \endcode
 *
 *
 * \section example_gm_even_isotropic Example (isotropic sigma² I)
 *
 * \code
 * gm_to_dirac_even_closed_form<double> approx;
 *
 * const double sigma = 2.0;
 * const double bMax  = 100.0;
 *
 * // solves the standard-normal problem at bMax / sigma and scales by sigma
 * bool ok = approx.approximate_isotropic(
 *     L, N, sigma, bMax, x, wX, &result, options);
 * \endcode
 *
 *
 * \section gm_even_math Closed forms
 *
 * The distance decomposes as D = D1 - 2·D2 + D3, all expressed in units
 * of pi^(N/2) — the same normalisation the quadrature path uses, so the
 * two are directly comparable. The building blocks live in
 * lcd_even_closed_form.h:
 *
 * - lcd_a_n()          -> D1, valid for all N (odd and even)
 * - lcd_delta_bkk()    -> the per-sample D2 term
 * - lcd_delta_bkk1()   -> the per-sample D2 gradient term
 * - lcd_c_repulsion()  -> the exact per-pair D3 term
 * - lcd_ei()      -> Ei(x)
 *
 * \section gm_even_notes Notes
 *
 * - Single-threaded implementation
 * - Uses analytical gradients with no quadrature
 * - The objective is assembled from reduced, O(1) terms and additionally
 *   carries the constant K by default
 *   (gm_to_dirac_even_closed_form::includeD1InObjective), so that f is the
 *   true distance and ApproximateOptions::ftolRel stays meaningful
 * - Interface-compatible with
 *   gm_to_dirac_short_standard_normal_deviation
 *
 */
