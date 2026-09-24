#ifndef GM_TO_DIRAC_EVEN_BINF_H
#define GM_TO_DIRAC_EVEN_BINF_H

#include "approximate_options.h"
#include "gm_to_dirac_even_binf_optimization_params.h"
#include "gsl_minimizer.h"
#include "gsl_vector_matrix_types.h"

/**
 * @brief bMax-free Gaussian-to-Dirac approximation for an even-dimensional standard normal target
 *
 * Same objective family as gm_to_dirac_even_closed_form, but the limit bMax -> infinity has been
 * taken analytically (see lcd_even_binf_closed_form.h), so no integration bound appears anywhere in the API.
 *
 * @note PRECONDITIONS
 *
 * - N must be EVEN and >= 2 and L >= 1. approximate() returns false and the void-returning entry points write NaN
 * to their output rather than leaving a stale buffer.
 * - The target must be the STANDARD normal N(0, I).
 * - The sample set must have ZERO MEAN, sum_i w_i s_i = 0. The limit exists only there; for a non-zero mean the
 * closed form still evaluates to a finite number that is NOT the distance.
 *
 * The minimizer optimises over an unconstrained t and every objective and
 * gradient evaluation begins by projecting
 *
 *     s = t - 1 * (w^T t)
 *
 * with the assembled gradient chain-ruled back by
 *
 *     grad_{t,q} = grad_{s,q} - w_q * sum_i grad_{s,i}
 *
 * so no iterate can ever leave the zero-mean manifold and the finite non-distance is unreachable.
 * The same projection runs inside the standalone distance and derivative entry points, which therefore report
 * the value for the MEAN-CORRECTED input rather than for the raw input.
 *
 * @note DEGENERATE SAMPLE SETS. phi_k is undefined at c = 0 and evaluating it there reaches lcd_ei(0),
 * which GSL's default error handler turns into a process abort. Any input whose projection puts a sample
 * exactly on the origin is therefore refused the same way odd N is: approximate() returns false before
 * the minimizer starts and the void entry points write NaN.
 * That covers L == 1, whose only zero-mean sample set is the single point 0 and inputs such as L identical locations.
 *
 * @tparam T float or double; computation is always performed in double and cast at the interface boundary
 */
template <typename T>
class gm_to_dirac_even_binf {
 public:
  using GSLVectorType = typename GSLTemplateTypeAlias<T>::VectorType;
  using GSLMatrixType = typename GSLTemplateTypeAlias<T>::MatrixType;

  // clang-format off
  /**
   * @brief approximate a standard normal by L Dirac components
   *
   * On return x holds the projected, zero-mean sample set s, not the raw
   * optimizer variable t.
   *
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x initial guess and output locations (L * N)
   * @param wX weights, nullptr for uniform
   * @param result minimizer result
   * @param options minimizer options; ApproximateOptions::bMax is IGNORED,
   * there is no integration bound on this path
   * @return true on success, false otherwise (including odd N)
   */
  bool approximate(size_t L,
                   size_t N,
                   T* x,
                   const T* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  /**
   * @brief true bMax-free mCvM distance, K_k included
   *
   * Reports the distance of the mean-corrected sample set
   *
   * @param distance output distance; set to NaN if the preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      T* x,
                                      const T* wX = nullptr);

  /**
   * @brief gradient of the true bMax-free mCvM distance w.r.t. the SAMPLES
   *
   * Returns grad_s, the derivative with respect to the sample locations themselves:
   *
   *     grad_s,q = -4 w_q phi_k'(c_q) s_q
   *              + 0.5 w_q sum_{j != q, T_qj > 0} w_j (s_q - s_j)
   *                                             * (ln(T_qj / 4) + 1)
   *
   * This is NOT the gradient the minimizer works with. approximate() varies
   * the unconstrained t behind s = t - 1 * (w^T t) and therefore uses
   * grad_t,q = grad_s,q - w_q * sum_i grad_s,i. The two agree only up to that
   * projection; grad_s is exposed here because it is the derivative of the
   * distance as a function of the sample set, which is what a caller asking
   * for "the gradient at x" means.
   *
   * @param gradient output gradient (L * N); filled with NaN if the
   * preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq_derivative(T* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 T* x,
                                                 const T* wX = nullptr);

  /**
   * @brief approximate a standard normal by L Dirac components
   *
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x initial guess and output locations (L * N)
   * @param wX weights, nullptr for uniform
   * @param result minimizer result
   * @param options minimizer options; ApproximateOptions::bMax is IGNORED
   * @return true on success, false otherwise (including odd N)
   */
  bool approximate(size_t L,
                   size_t N,
                   GSLVectorType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  /**
   * @brief true bMax-free mCvM distance, K_k included
   *
   * @param distance output distance; set to NaN if the preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      GSLVectorType* x,
                                      const GSLVectorType* wX = nullptr);

  /**
   * @brief gradient of the true bMax-free mCvM distance w.r.t. the SAMPLES
   *
   * Returns grad_s, not the minimizer's grad_t; see the raw-pointer overload.
   *
   * @param gradient output gradient (L * N); filled with NaN if the
   * preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq_derivative(GSLVectorType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 GSLVectorType* x,
                                                 const GSLVectorType* wX = nullptr);

  /**
   * @brief approximate a standard normal by L Dirac components
   *
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x initial guess and output locations (L rows, N columns)
   * @param wX weights, nullptr for uniform
   * @param result minimizer result
   * @param options minimizer options; ApproximateOptions::bMax is IGNORED
   * @return true on success, false otherwise (including odd N)
   */
  bool approximate(size_t L,
                   size_t N,
                   GSLMatrixType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  /**
   * @brief true bMax-free mCvM distance, K_k included
   *
   * @param distance output distance; set to NaN if the preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L rows, N columns)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      GSLMatrixType* x,
                                      const GSLVectorType* wX = nullptr);

  /**
   * @brief gradient of the true bMax-free mCvM distance w.r.t. the SAMPLES
   *
   * Returns grad_s, not the minimizer's grad_t; see the raw-pointer overload.
   *
   * @param gradient output gradient (L rows, N columns); filled with NaN if
   * the preconditions fail
   * @param L number of Dirac components, must be >= 1
   * @param N dimension, must be even and >= 2
   * @param x sample locations (L rows, N columns)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 GSLMatrixType* x,
                                                 const GSLVectorType* wX = nullptr);
  // clang-format on

 private:
  /// shared evaluation body; chainRuleToT selects grad_t (the minimizer's
  /// variable) over grad_s (the sample locations)
  static void evaluate(const gsl_vector* t,
                       GMToDiracEvenBinfOptimizationParams* params, double* f,
                       gsl_vector* grad, bool chainRuleToT);

  // gsl_minimizer entry points; these always work in t
  static double modified_van_mises_distance_sq(const gsl_vector* t,
                                               void* params);
  static void modified_van_mises_distance_sq_derivative(const gsl_vector* t,
                                                        void* params,
                                                        gsl_vector* grad);
  static void combined_distance_metric(const gsl_vector* t, void* params,
                                       double* f, gsl_vector* grad);

  static inline void calculateAttraction(
      GMToDiracEvenBinfOptimizationParams* params, double* f, gsl_vector* grad);

  static inline void calculateRepulsion(
      GMToDiracEvenBinfOptimizationParams* params, double* f, gsl_vector* grad);

  static inline void projectGradient(
      const GMToDiracEvenBinfOptimizationParams* params, gsl_vector* grad);

  static inline void correctMean(gsl_vector* x, const gsl_vector* wX, size_t L,
                                 size_t N);

  static inline bool preconditionsHold(size_t L, size_t N);
};

#include "gm_to_dirac_even_binf.tpp"

template <>
bool gm_to_dirac_even_binf<float>::approximate(
    size_t L, size_t N, gsl_vector_float* x, const gsl_vector_float* wX,
    GslminimizerResult* result, const ApproximateOptions& options);

template <>
bool gm_to_dirac_even_binf<double>::approximate(
    size_t L, size_t N, gsl_vector* x, const gsl_vector* wX,
    GslminimizerResult* result, const ApproximateOptions& options);

template <>
void gm_to_dirac_even_binf<float>::modified_van_mises_distance_sq(
    float* distance, size_t L, size_t N, gsl_vector_float* x,
    const gsl_vector_float* wX);

template <>
void gm_to_dirac_even_binf<double>::modified_van_mises_distance_sq(
    double* distance, size_t L, size_t N, gsl_vector* x, const gsl_vector* wX);

template <>
void gm_to_dirac_even_binf<float>::modified_van_mises_distance_sq_derivative(
    gsl_vector_float* gradient, size_t L, size_t N, gsl_vector_float* x,
    const gsl_vector_float* wX);

template <>
void gm_to_dirac_even_binf<double>::modified_van_mises_distance_sq_derivative(
    gsl_vector* gradient, size_t L, size_t N, gsl_vector* x,
    const gsl_vector* wX);

extern template class gm_to_dirac_even_binf<double>;
extern template class gm_to_dirac_even_binf<float>;

#endif  // GM_TO_DIRAC_EVEN_BINF_H
