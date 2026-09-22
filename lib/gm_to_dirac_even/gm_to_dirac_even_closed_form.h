#ifndef GM_TO_DIRAC_EVEN_CLOSED_FORM_H
#define GM_TO_DIRAC_EVEN_CLOSED_FORM_H

#include "gm_to_dirac_approx_standard_normal_distribution_i.h"
#include "gm_to_dirac_even_optimization_params.h"

/**
 * @brief quadrature-free Gaussian-to-Dirac approximation for an
 * even-dimensional standard normal target
 *
 * Same objective as gm_to_dirac_short_standard_normal_deviation, but every
 * integral over the kernel-width variable b is evaluated in closed form (see
 * lcd_even_closed_form.h).
 *
 * One further differences from the quadrature path:
 *
 * - the reported distance is the TRUE mCvM distance: it includes the D1 term,
 *   which gm_to_dirac_short omits.
 *
 * @note PRECONDITIONS: the target must be the STANDARD normal N(0, I) (or an
 * isotropic sigma^2 * I via approximate_isotropic()), and N must be EVEN and
 * >= 2. Odd N fails loudly rather than falling back.
 *
 */
template <typename T>
class gm_to_dirac_even_closed_form
    : public gm_to_dirac_approx_standard_normal_distribution_i<T> {
 public:
  using GSLVectorType = typename gm_to_dirac_approx_standard_normal_distribution_i<
      T>::GSLVectorType;
  using GSLMatrixType = typename gm_to_dirac_approx_standard_normal_distribution_i<
      T>::GSLMatrixType;

  /**
   * @brief carry the sample-independent constant inside the minimizer's
   * objective
   *
   * The constant does not depend on x, so omitting it leaves the gradient and
   * the minimum unchanged and saves one scalar evaluation per objective call.
   * With it, the objective IS the distance.
   */
  bool includeD1InObjective = true;

  // clang-format off
  /**
   * @brief approximate a standard normal by L Dirac components
   *
   * @param L number of Dirac components
   * @param N dimension, must be even and >= 2
   * @param bMax upper bound of the b-integral, must be > 0
   * @param x initial guess and output locations (L * N)
   * @param wX weights, nullptr for uniform
   * @param result minimizer result
   * @param options minimizer options; options.bMax is ignored here
   * @return true on success, false otherwise (including odd N)
   */
  bool approximate(size_t L,
                   size_t N,
                   double bMax,
                   T* x,
                   const T* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  /**
   * @brief true mCvM distance, D1 included
   *
   * @param distance output distance
   * @param L number of Dirac components
   * @param N dimension, must be even and >= 2
   * @param bMax upper bound of the b-integral, must be > 0
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      double bMax,
                                      T* x,
                                      const T* wX);

  /**
   * @brief gradient of the true mCvM distance with respect to x
   *
   * @param gradient output gradient (L * N)
   * @param L number of Dirac components
   * @param N dimension, must be even and >= 2
   * @param bMax upper bound of the b-integral, must be > 0
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq_derivative(T* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 double bMax,
                                                 T* x,
                                                 const T* wX);

  bool approximate(size_t L,
                   size_t N,
                   double bMax,
                   GSLVectorType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      double bMax,
                                      GSLVectorType* x,
                                      const GSLVectorType* wX);

  void modified_van_mises_distance_sq_derivative(GSLVectorType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 double bMax,
                                                 GSLVectorType* x,
                                                 const GSLVectorType* wX);

  bool approximate(size_t L,
                   size_t N,
                   double bMax,
                   GSLMatrixType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{});

  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      double bMax,
                                      GSLMatrixType* x,
                                      const GSLVectorType* wX);

  void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 double bMax,
                                                 GSLMatrixType* x,
                                                 const GSLVectorType* wX);
  // clang-format on

  // isotropic sigma^2 * I (scaling reduction) 

  // clang-format off
  /**
   * @brief approximate an isotropic Gaussian N(0, sigma^2 * I)
   *
   * Solves the standard-normal problem at bMax / sigma and scales the result
   * by sigma. This is why the new API takes a double bMax: bMax / sigma is
   * generally not an integer.
   *
   * @param L number of Dirac components
   * @param N dimension, must be even and >= 2
   * @param sigma standard deviation, must be > 0
   * @param bMax upper bound of the b-integral, must be > 0
   * @param x initial guess and output locations (L * N)
   * @param wX weights, nullptr for uniform
   * @param result minimizer result
   * @param options minimizer options; options.bMax is ignored here
   * @return true on success, false otherwise
   * @note only isotropic covariance is supported; general diagonal covariance
   * admits no such reduction
   */
  bool approximate_isotropic(size_t L,
                             size_t N,
                             double sigma,
                             double bMax,
                             T* x,
                             const T* wX = nullptr,
                             GslminimizerResult* result = nullptr,
                             const ApproximateOptions& options = ApproximateOptions{});

  /**
   * @brief true mCvM distance for an isotropic Gaussian N(0, sigma^2 * I)
   *
   * D(sigma, bMax) = sigma^2 * D(1, bMax / sigma).
   *
   * @param distance output distance
   * @param L number of Dirac components
   * @param N dimension, must be even and >= 2
   * @param sigma standard deviation, must be > 0
   * @param bMax upper bound of the b-integral, must be > 0
   * @param x sample locations (L * N)
   * @param wX weights, nullptr for uniform
   */
  void modified_van_mises_distance_sq_isotropic(T* distance,
                                                size_t L,
                                                size_t N,
                                                double sigma,
                                                double bMax,
                                                T* x,
                                                const T* wX);

  bool approximate(size_t L,
                   size_t N,
                   T* x,
                   const T* wX,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;

  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      T* x,
                                      const T* wX) override;

  void modified_van_mises_distance_sq_derivative(T* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 T* x,
                                                 const T* wX) override;

  bool approximate(size_t L,
                   size_t N,
                   GSLVectorType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;

  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      GSLVectorType* x,
                                      const GSLVectorType* wX) override;

  void modified_van_mises_distance_sq_derivative(GSLVectorType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 GSLVectorType* x,
                                                 const GSLVectorType* wX) override;

  bool approximate(size_t L,
                   size_t N,
                   GSLMatrixType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;

  void modified_van_mises_distance_sq(T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      GSLMatrixType* x,
                                      const GSLVectorType* wX) override;

  void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 GSLMatrixType* x,
                                                 const GSLVectorType* wX) override;
  // clang-format on

 private:
  static double modified_van_mises_distance_sq(const gsl_vector* x,
                                               void* params);
  static void modified_van_mises_distance_sq_derivative(const gsl_vector* x,
                                                        void* params,
                                                        gsl_vector* grad);
  static void combined_distance_metric(const gsl_vector* x, void* params,
                                       double* f, gsl_vector* grad);

  static inline void calculateD2(const gsl_vector* x,
                                 GMToDiracEvenOptimizationParams* params,
                                 double* f, gsl_vector* grad);

  static inline void calculateD3(const gsl_vector* x,
                                 GMToDiracEvenOptimizationParams* params,
                                 double* f, gsl_vector* grad);

  static inline void correctMean(gsl_vector* x, const gsl_vector* wX, size_t L,
                                 size_t N);

  static inline bool preconditionsHold(size_t L, size_t N, double bMax);
};

#include "gm_to_dirac_even_closed_form.tpp"

template <>
bool gm_to_dirac_even_closed_form<float>::approximate(
    size_t L, size_t N, double bMax, gsl_vector_float* x,
    const gsl_vector_float* wX, GslminimizerResult* result,
    const ApproximateOptions& options);

template <>
bool gm_to_dirac_even_closed_form<double>::approximate(
    size_t L, size_t N, double bMax, gsl_vector* x, const gsl_vector* wX,
    GslminimizerResult* result, const ApproximateOptions& options);

template <>
void gm_to_dirac_even_closed_form<float>::modified_van_mises_distance_sq(
    float* distance, size_t L, size_t N, double bMax, gsl_vector_float* x,
    const gsl_vector_float* wX);

template <>
void gm_to_dirac_even_closed_form<double>::modified_van_mises_distance_sq(
    double* distance, size_t L, size_t N, double bMax, gsl_vector* x,
    const gsl_vector* wX);

template <>
void gm_to_dirac_even_closed_form<
    float>::modified_van_mises_distance_sq_derivative(
    gsl_vector_float* gradient, size_t L, size_t N, double bMax,
    gsl_vector_float* x, const gsl_vector_float* wX);

template <>
void gm_to_dirac_even_closed_form<
    double>::modified_van_mises_distance_sq_derivative(
    gsl_vector* gradient, size_t L, size_t N, double bMax, gsl_vector* x,
    const gsl_vector* wX);

extern template class gm_to_dirac_even_closed_form<double>;
extern template class gm_to_dirac_even_closed_form<float>;

#endif  // GM_TO_DIRAC_EVEN_CLOSED_FORM_H
