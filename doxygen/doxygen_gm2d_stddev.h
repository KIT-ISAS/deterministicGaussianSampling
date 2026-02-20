/**
 * \page page_gm_stddev gm_to_dirac_short_standard_normal_deviation
 *
 * Gaussian-to-Dirac approximation (standard normal deviation variant).
 *
 * \section gm_stddev_overview Overview
 *
 * gm_to_dirac_short_standard_normal_deviation<T> approximates a
 * multivariate Gaussian distribution by a Dirac mixture with L
 * components in N dimensions.
 *
 * This implementation assumes a standardized Gaussian structure
 * (standard normal deviation scaling) and performs gradient-based
 * optimization using a GSL minimizer backend.
 *
 * It is a single-threaded implementation.
 *
 *
 * \section gm_stddev_interface Interface
 *
 * Inherits:
 *
 * - gm_to_dirac_approx_standard_normal_distribution_i<T>
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
 * \section gm_stddev_parameters Parameters
 *
 * Common parameters:
 *
 * - L     → number of Dirac components
 * - N     → dimension
 * - bMax  → integration bound
 * - x     → initial guess and output locations (L × N)
 * - wX    → weights of the Dirac mixture (optional)
 *
 * If wX is nullptr:
 *
 * - Uniform weights are assumed
 *
 *
 * \section gm_stddev_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout:
 *
 * - x represents L × N Dirac locations
 *
 *
 * \section example_gm_stddev_raw Example (Raw Pointer)
 *
 * \code
 * gm_to_dirac_short_standard_normal_deviation<double> approx;
 *
 * bool ok = approx.approximate(
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
 * \section example_gm_stddev_gsl_vector Example (GSL Vector)
 *
 * \code
 * gm_to_dirac_short_standard_normal_deviation<double> approx;
 *
 * bool ok = approx.approximate(
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
 * \section example_gm_stddev_gsl_matrix Example (GSL Matrix)
 *
 * \code
 * gm_to_dirac_short_standard_normal_deviation<double> approx;
 *
 * bool ok = approx.approximate(
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
 * \section gm_stddev_notes Notes
 *
 * - Single-threaded implementation
 * - Uses analytical gradient evaluation
 * - Specialized for standardized Gaussian structures
 * - Interface-compatible with other Gaussian-to-Dirac variants
 *
 */
