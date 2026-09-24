/**
 * \page page_gm_even_binf gm_to_dirac_even_binf
 *
 * Gaussian-to-Dirac approximation, quadrature-free and bMax-free (even dimensions).
 *
 * \section gm_even_binf_overview Overview
 *
 * gm_to_dirac_even_binf<T> approximates a standard normal distribution by a
 * Dirac mixture with L components in N dimensions, exactly as
 * gm_to_dirac_even_closed_form does, but with the limit bMax -> infinity
 * taken analytically.
 *
 * The finite-bMax path evaluates every b-integral in closed form but still carries the integration
 * bound as a parameter, and the answer it reports depends on it. Here the bound is gone:
 * there is no bMax argument on any method, and ApproximateOptions::bMax is ignored.
 *
 * With N = 2k, \f$ c_i = \parallel s_i \parallel^2 \f$ and \f$ T_ij = \parallel s_i - s_j \parallel ^2 \f$, the limit distance
 * is
 * 
 * <center>\f$ D = K_k - 2 \sum_i w_i \phi_k(c_i) + \sum_{i,j, T_{ij} > 0} w_i w_j \left( \frac{T_{ij}}{8} \right) \ln\left( \frac{T_{ij}}{4} \right) \f$</center>
 *
 * in units of \f$ \pi^{N/2} \f$, the same normalisation the other paths use, so all
 * three are directly comparable.
 *
 *
 * \section gm_even_binf_gallery Sample gallery
 *
 * Every point below was produced by gm_to_dirac_even_binf alone, no
 * quadrature path, no finite-bMax path, no bMax anywhere. The shaded field is
 * the target standard normal density and the red points are the optimized
 * Dirac locations. D is the reported bMax-free mCvM distance at the optimum,
 * K_k included.
 *
 * \image html samples_even_binf_standard_normal.png "bMax-free closed-form LCD (N = 2): standard normal at L = 100, 200, 400." width=80%
 *
 * Reproduce with:
 *
 * \code
 * cmake --build build --target generate_samples_even_binf
 * python plots/plot_samples_even_binf.py \
 *     --generator build/plots/generate_samples_even_binf \
 *     --out-dir doxygen/images
 * \endcode
 *
 *
 * \section gm_even_binf_preconditions Preconditions
 *
 * - N must be even and >= 2.
 * - The target must be the standard normal N(0, I). There is no isotropic variant here.
 * - The sample set must have zero mean, \f$ \sum_i w_i s_i = 0 \f$.
 *
 *
 * \section gm_even_binf_zeromean The zero-mean precondition
 *
 * The bMax -> infinity limit exists only on the zero-mean manifold. Off it,
 * the true distance diverges,
 *
 * <center>\f$ D(bMax) \sim \frac{||\mu||^2}{4} * ln(bMax^2) + const,  \   \mu = \sum_i w_i s_i \f$</center>
 *
 * and grows without bound as the integration bound is relaxed.
 *
 * \warning The closed form above nevertheless evaluates to a perfectly ordinary,
 * finite number when it is handed a sample set with a non-zero mean. However, this result would
 * not be the true distance measure, because it does not include the terms which disappeared
 * when using the fact that the sample mean is zero. Thus, this result cannot be applied
 * to computing the distance measure for cases without zero mean of the deterministic sample set.
 *
 * The class removes the hazard structurally rather than checking for it.
 * The minimizer varies an unconstrained t, and every objective and gradient evaluation begins by projecting
 *
 * <center>\f$ s = t - 1 * (w^T t) \f$</center>
 *
 * with the assembled gradient chain-ruled back by
 *
 * <center>\f$ grad_{t,q} = grad_{s,q} - w_q * \sum_i grad_{s,i} \f$</center>
 *
 * so no iterate can leave the manifold and the finite non-distance is unreachable.
 * The same projection runs inside the standalone distance and derivative entry points,
 * which therefore report the value for the mean-corrected input rather than for the raw input.
 *
 * correctMean() after minimize() is not a substitute for this and must not be relied on as one.
 * It is called at the end because it is exactly the same projection and so turns the returned
 * optimizer variable t into the sample set s that the reported objective belongs to.
 *
 * \section gm_even_binf_interface Interface
 *
 * Template parameter:
 *
 * - T in {float, double}; computation is always performed in double and cast
 *   at the interface boundary.
 *
 * Provides overloads of:
 *
 * - approximate(...)
 * - modified_van_mises_distance_sq(...)
 * - modified_van_mises_distance_sq_derivative(...)
 *
 * for raw pointers (T*), GSL vectors (gsl_vector / gsl_vector_float) and GSL
 * matrices (gsl_matrix / gsl_matrix_float). x is L x N, row major.
 *
 * If wX is nullptr, uniform weights are assumed.
 *
 * \note modified_van_mises_distance_sq_derivative() returns \f$ grad_s \f$, the
 * derivative with respect to the sample locations. That is not the gradient
 * the minimizer works with; approximate() uses \f$ grad_t \f$, which additionally
 * carries the projection chain rule above.
 *
 *
 * \section example_gm_even_binf_raw Example (Raw Pointer)
 *
 * \code
 * gm_to_dirac_even_binf<double> approx;
 *
 * bool ok = approx.approximate(
 *     L,
 *     N,        // must be even
 *     x,        // initial guess / output (L x N), returned zero-mean
 *     wX,       // weights (optional)
 *     &result,
 *     options   // options.bMax is ignored
 * );
 * \endcode
 *
 *
 * \section example_gm_even_binf_distance Example (true distance)
 *
 * \code
 * gm_to_dirac_even_binf<double> approx;
 *
 * double distance = 0.0;
 * approx.modified_van_mises_distance_sq(&distance, L, N, x, wX);
 * // includes K_k, so this is the true bMax-free mCvM distance of the
 * // mean-corrected sample set
 * \endcode
 *
 *
 * \section gm_even_binf_math Closed forms
 *
 * The building blocks live in lcd_even_binf_closed_form.h:
 *
 * - lcd_binf_bkk_zero()        -> \f$ B_{k,k}(0, c) \f$, the attraction term
 * - lcd_binf_bkk_zero_deriv()  -> \f$ B_{k,k}'(0, c) \f$, its derivative
 * - lcd_binf_phi()             -> \f$ \phi_k(c) \f$, the per-sample potential
 * - lcd_binf_phi_deriv()       -> \f$ \phi_k'(c) \f$, its derivative
 * - lcd_binf_constant()        -> \f$ K_k \f$
 *
 * Ei is taken from lcd_ei() of lcd_even_closed_form.h.
 *
 * K_k is a pure constant. It does not move the optimum, and is carried so
 * that the reported value is the true distance; it plays the role
 * lcd_reported_offset() plays on the finite-bMax path.
 *
 */
