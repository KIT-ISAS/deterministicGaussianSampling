/**
 * \page page_gm_short gm_to_dirac_short
 *
 * Gaussian-to-Dirac approximation (diagonal covariance).
 *
 * \section gm_short_overview Overview
 *
 * gm_to_dirac_short<T> approximates a multivariate Gaussian
 * distribution with diagonal covariance by a Dirac mixture
 * with L components in N dimensions.
 *
 * The covariance is provided as a diagonal vector (covDiag).
 *
 * This implementation performs gradient-based optimization
 * using a GSL minimizer backend.
 *
 * It is a single-threaded implementation.
 *
 *
 * \section gm_short_interface Interface
 *
 * Inherits:
 *
 * - gm_to_dirac_approx_i<T>
 *
 * Template parameter:
 *
 * - T ∈ {float, double}
 *
 * Provides overloads of:
 *
 * - approximate(...)
 * - modified_van_mises_distance_sq(...)
 * - modified_van_mises_distance_sq_derivative(...)
 *
 *
 * \section gm_short_parameters Parameters
 *
 * Common parameters:
 *
 * - covDiag → diagonal of Gaussian covariance (size N)
 * - L       → number of Dirac components
 * - N       → dimension
 * - bMax    → integration bound
 * - x       → initial guess and output locations (L × N)
 * - wX      → weights of the Dirac mixture (optional)
 *
 * If wX is nullptr:
 *
 * - Uniform weights are assumed
 *
 *
 * \section gm_short_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout:
 *
 * - covDiag represents the covariance diagonal (size N)
 * - x represents L × N Dirac locations
 *
 *
 * \section example_gm_short_raw Example (Raw Pointer)
 *
 * \code
 * gm_to_dirac_short<double> approx;
 *
 * bool ok = approx.approximate(
 *     covDiag,  // covariance diagonal (size N)
 *     L,
 *     N,
 *     bMax,
 *     x,        // initial guess / output (L × N)
 *     wX,       // weights (optional)
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_gm_short_gsl_vector Example (GSL Vector)
 *
 * \code
 * gm_to_dirac_short<double> approx;
 *
 * bool ok = approx.approximate(
 *     covDiagVector,
 *     L,
 *     N,
 *     bMax,
 *     xVector,
 *     wXVector,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_gm_short_gsl_matrix Example (GSL Matrix)
 *
 * \code
 * gm_to_dirac_short<double> approx;
 *
 * bool ok = approx.approximate(
 *     covDiagVector,
 *     L,
 *     N,
 *     bMax,
 *     xMatrix,
 *     wXVector,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section gm_short_notes Notes
 *
 * - Single-threaded implementation
 * - Uses analytical gradient evaluation
 * - Requires diagonal covariance input
 * - Interface-compatible with other Gaussian-to-Dirac variants
 *
 */
