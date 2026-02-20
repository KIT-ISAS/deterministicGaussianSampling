/**
 * \page page_d2d_function dirac_to_dirac_approx_short_function
 *
 * Dirac-to-Dirac reduction with user-defined weight function.
 *
 * \section d2d_function_overview Overview
 *
 * dirac_to_dirac_approx_short_function<T> reduces a Dirac mixture
 * with M components to L components in N dimensions, similar to:
 *
 * - \ref page_d2d_short
 *
 * The key difference:
 *
 * Instead of providing static weights wX, this implementation
 * computes the reduced mixture weights via user-defined callbacks.
 *
 * This allows:
 *
 * - State-dependent weighting
 * - Structure-adaptive reduction
 * - Custom weighting strategies
 *
 *
 * \section d2d_function_interface Interface
 *
 * Inherits:
 *
 * - dirac_to_dirac_approx_function_i<T>
 *
 * Template parameter:
 *
 * - T ∈ {float, double}
 *
 * The user must provide:
 *
 * - wXf  → weight callback
 * - wXd  → weight gradient callback
 *
 *
 * \section d2d_function_callbacks Callback Types
 *
 * The following function types are required:
 *
 * \code
 * using wXf = void (*)(const T* x,
 *                      T* weights,
 *                      size_t L,
 *                      size_t N);
 *
 * using wXd = void (*)(const T* x,
 *                      T* gradient,
 *                      size_t L,
 *                      size_t N);
 * \endcode
 *
 * - wXf computes the weights for the reduced components
 * - wXd computes the derivative of the weights w.r.t. x
 *
 * Both callbacks operate on flattened L × N data.
 *
 *
 * \section d2d_function_input Supported Input Formats
 *
 * Three overload families are available:
 *
 * - Raw pointer interface (T*)
 * - GSL vector interface (gsl_vector)
 * - GSL matrix interface (gsl_matrix)
 *
 * Memory layout and parameter semantics match the
 * non-function-based implementation.
 *
 *
 * \section example_d2d_function_callbacks Example Callbacks
 *
 * \code
 * static void wXcallback(const double* x,
 *                        double* res,
 *                        size_t L,
 *                        size_t N)
 * {
 *     for (size_t i = 0; i < L; ++i) {
 *         double sum = 0.0;
 *         for (size_t k = 0; k < N; ++k) {
 *             double v = x[i * N + k];
 *             sum += v * v;
 *         }
 *         res[i] = std::exp(0.5 * sum);
 *     }
 * }
 *
 * static void wXDcallback(const double* x,
 *                         double* grad,
 *                         size_t L,
 *                         size_t N)
 * {
 *     for (size_t i = 0; i < L; ++i) {
 *         double sum = 0.0;
 *         for (size_t k = 0; k < N; ++k) {
 *             double v = x[i * N + k];
 *             sum += v * v;
 *         }
 *         double factor = std::exp(0.5 * sum);
 *
 *         for (size_t k = 0; k < N; ++k) {
 *             grad[i * N + k] = x[i * N + k] * factor;
 *         }
 *     }
 * }
 * \endcode
 *
 *
 * \section example_d2d_function_raw Example (Raw Pointer)
 *
 * \code
 * dirac_to_dirac_approx_short_function<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     y,            // input points (M × N)
 *     M,
 *     L,
 *     N,
 *     bMax,
 *     x,            // initial guess / output
 *     wXcallback,   // weight function
 *     wXDcallback,  // weight derivative
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_d2d_function_gsl_vector Example (GSL Vector)
 *
 * \code
 * dirac_to_dirac_approx_short_function<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     yVector,
 *     L,
 *     N,
 *     bMax,
 *     xVector,
 *     wXcallback,
 *     wXDcallback,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section example_d2d_function_gsl_matrix Example (GSL Matrix)
 *
 * \code
 * dirac_to_dirac_approx_short_function<double> reducer;
 *
 * bool ok = reducer.approximate(
 *     yMatrix,
 *     L,
 *     bMax,
 *     xMatrix,
 *     wXcallback,
 *     wXDcallback,
 *     &result,
 *     options
 * );
 * \endcode
 *
 *
 * \section d2d_function_notes Notes
 *
 * - Weights are fully controlled by user callbacks
 * - Analytical gradient includes weight derivatives
 * - Multi-threaded implementation
 * - Interface-compatible with other Dirac-to-Dirac variants
 *
 */
