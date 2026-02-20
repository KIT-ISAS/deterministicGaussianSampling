/**
 * \page page_d2d_thread dirac_to_dirac_approx_short_thread
 *
 * Threaded Dirac-to-Dirac reduction.
 *
 * \section d2d_thread_overview Overview
 *
 * dirac_to_dirac_approx_short_thread<T> is the multi-threaded variant
 * of \ref page_d2d_short.
 *
 * It provides the same functionality, interface, and behavior as the
 * single-threaded implementation, but evaluates the distance metric
 * and its gradient in parallel.
 *
 * The optimization result is equivalent to the non-threaded version.
 *
 * Use this implementation for:
 * - Large M (input components)
 * - Large L (reduced components)
 * - High-dimensional problems
 * - Performance-critical applications
 *
 *
 * \section d2d_thread_interface Interface
 *
 * Inherits:
 *
 * - dirac_to_dirac_approx_i<T>
 *
 * Template parameter:
 *
 * - T ∈ {float, double}
 *
 * All overloads of:
 *
 * - approximate(...)
 * - modified_van_mises_distance_sq(...)
 * - modified_van_mises_distance_sq_derivative(...)
 *
 * are identical to the non-threaded implementation.
 *
 *
 * \section d2d_thread_input Supported Input Formats
 *
 * The same overload families as in \ref page_d2d_short are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector / gsl_vector_float)
 * - GSL matrix interface (gsl_matrix / gsl_matrix_float)
 *
 * Memory layout and parameter semantics are unchanged.
 *
 *
 * \section d2d_thread_weights Weights
 *
 * - wY: weights of input Dirac mixture
 * - wX: weights of reduced mixture
 *
 * If nullptr:
 *
 * - Uniform weights are assumed
 *
 *
 * \section d2d_thread_example_raw Example (Raw Pointer)
 *
 * \code
 * dirac_to_dirac_approx_short_thread<double> reducer;
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
 * \section d2d_thread_example_gsl_vector Example (GSL Vector)
 *
 * \code
 * dirac_to_dirac_approx_short_thread<double> reducer;
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
 * \section d2d_thread_example_gsl_matrix Example (GSL Matrix)
 *
 * \code
 * dirac_to_dirac_approx_short_thread<double> reducer;
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
 * \section d2d_thread_notes Notes
 *
 * - Produces the same results as dirac_to_dirac_approx_short
 * - Parallelizes internal distance and gradient evaluation
 * - Interface-compatible replacement for the non-threaded version
 *
 * If threading is not required, use \ref page_d2d_short.
 *
 */
