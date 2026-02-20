/**
 * \page page_d2d_short dirac_to_dirac_approx_short
 *
 * Single-threaded Dirac-to-Dirac reduction.
 *
 * \section d2d_short_overview Overview
 *
 * dirac_to_dirac_approx_short<T> reduces a Dirac mixture with M
 * components to a compact mixture with L components in N dimensions.
 *
 * This is the base (non-threaded) implementation.
 *
 * It performs gradient-based optimization of the internal distance
 * metric using a GSL minimizer backend.
 *
 * Use this implementation for:
 * - Small to medium problem sizes
 * - Deterministic execution
 * - Environments where threading is not desired
 *
 * For a multi-threaded drop-in replacement, see:
 * - \ref page_d2d_thread
 *
 *
 * \section d2d_short_interface Interface
 *
 * Inherits:
 *
 * - dirac_to_dirac_approx_i<T>
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
 * \section d2d_short_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout and parameter semantics are identical across
 * overloads.
 *
 *
 * \section d2d_short_weights Weights
 *
 * - wY: weights of the input Dirac mixture
 * - wX: weights of the reduced Dirac mixture
 *
 * If nullptr:
 *
 * - Uniform weights are assumed
 *
 *
 * \section example_d2d_short_raw Example (Raw Pointer)
 *
 * \code
 * dirac_to_dirac_approx_short<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     y,        // input points (M × N)
 *     M,
 *     L,
 *     N,
 *     bMax,
 *     x,        // initial guess / output
 *     wX,       // reduced weights (optional)
 *     wY,       // input weights (optional)
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_d2d_short_gsl_vector Example (GSL Vector)
 *
 * \code
 * dirac_to_dirac_approx_short<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     yVector,
 *     L,
 *     N,
 *     bMax,
 *     xVector,
 *     wXVector,
 *     wYVector,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_d2d_short_gsl_matrix Example (GSL Matrix)
 *
 * \code
 * dirac_to_dirac_approx_short<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     yMatrix,
 *     L,
 *     bMax,
 *     xMatrix,
 *     wXVector,
 *     wYVector,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section d2d_short_notes Notes
 *
 * - Deterministic single-threaded execution
 * - Uses analytical gradient evaluation
 * - Interface-compatible with dirac_to_dirac_approx_short_thread
 *
 */
