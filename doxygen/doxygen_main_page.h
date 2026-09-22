/**
 * \mainpage Dirac Mixture Approximation Library (C++)
 *
 * \section overview Overview
 *
 * This repository provides a high-performance C++ implementation for
 * Dirac mixture reduction and Gaussian-to-Dirac approximation.
 *
 * The library focuses on efficient, optimization-based approximation of
 * probability distributions using Localized Cumulative Distribution (LCD)
 * distances and gradient-based minimization.
 *
 * The implementation is designed for:
 * - High-dimensional problems
 * - Performance-critical applications
 * - Modular integration into estimation pipelines
 * - Threaded and non-threaded execution
 *
 * \section dirac_to_dirac Dirac-to-Dirac Reduction
 *
 * Reduces a high-resolution Dirac mixture to a compact representation.
 *
 * Implementations:
 * - \ref page_d2d_short "dirac_to_dirac_approx_short"
 * - \ref page_d2d_thread "dirac_to_dirac_approx_short_thread"
 * - \ref page_d2d_function "dirac_to_dirac_approx_short_function"
 *
 * All implementations follow the dirac_to_dirac_approx_i<T> interface,
 * except:
 * - dirac_to_dirac_approx_short_function, which follows
 *   dirac_to_dirac_approx_function_i<T>.
 *
 * Supported input formats:
 * - Raw pointers
 * - GSL vectors
 * - GSL matrices
 *
 * \section gm_to_dirac Gaussian-to-Dirac Approximation
 *
 * Approximates a multivariate Gaussian distribution by a Dirac mixture.
 *
 * Implementations:
 * - \ref page_gm_short "gm_to_dirac_short"
 * - \ref page_gm_stddev "gm_to_dirac_short_standard_normal_deviation"
 * - \ref page_gm_even "gm_to_dirac_even_closed_form" — quadrature-free;
 *   requires even N and a standard normal or isotropic target, but accepts
 *   any bMax > 0
 *
 * All implementations follow the gm_to_dirac_approx_i<T> interface, except:
 * - gm_to_dirac_short_standard_normal_deviation and
 *   gm_to_dirac_even_closed_form, which follow
 *   gm_to_dirac_approx_standard_normal_distribution_i<T>.
 *
 * \section samples Sample Gallery
 *
 * See the \ref page_samples "LCD Sample Gallery" for code-generated plots of
 * 2-D Gaussian densities overlaid with their deterministic Dirac sample points.
 *
 * \section design Design Highlights
 *
 * - Template-based numeric precision (float, double)
 * - Unified interface design for interchangeable implementations
 * - GSL-based minimization backend
 * - Analytical gradients
 * - Threaded performance variants
 * - Configurable optimization via \ref ApproximateOptions
 *
 * The library is structured to separate interfaces, algorithmic
 * implementations, and optimization backends, enabling clean extension
 * and experimentation.
 *
 */
