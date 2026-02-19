#ifndef GM_TO_DIRAC_SHORT_H
#define GM_TO_DIRAC_SHORT_H

#include <gtest/gtest.h>

#include "gm_to_dirac_approx_i.h"
#include "gm_to_dirac_optimization_params.h"

template <typename T>
class gm_to_dirac_short : public gm_to_dirac_approx_i<T> {
 public:
  using GSLVectorType = typename gm_to_dirac_approx_i<T>::GSLVectorType;
  using GSLVectorViewType = typename gm_to_dirac_approx_i<T>::GSLVectorViewType;
  using GSLMatrixType = typename gm_to_dirac_approx_i<T>::GSLMatrixType;

  // clang-format off
  bool approximate(const T* covDiag,
                   size_t L,
                   size_t N,
                   size_t bMax,
                   T* x,
                   const T* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq(const T* covDiag,
                                      T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      T* x,
                                      const T* wX) override;
  // clang-format on              
                
  // clang-format off
  void modified_van_mises_distance_sq_derivative(const T* covDiag,
                                                 T* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 T* x,
                                                 const T* wX) override;
  // clang-format on

  // clang-format off
  bool approximate(const GSLVectorType* covDiag,
                   size_t L,
                   size_t N,
                   size_t bMax,
                   GSLVectorType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq(const GSLVectorType* covDiag,
                                      T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      GSLVectorType* x,
                                      const GSLVectorType* wX) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq_derivative(const GSLVectorType* covDiag,
                                                 GSLVectorType* gradient,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 GSLVectorType* x,
                                                 const GSLVectorType* wX) override;
  // clang-format on

  // clang-format off
  bool approximate(const GSLVectorType* covDiag,
                   size_t L,
                   size_t N,
                   size_t bMax,
                   GSLMatrixType* x,
                   const GSLVectorType* wX = nullptr,
                   GslminimizerResult* result = nullptr,
                   const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq(const GSLVectorType* covDiag,
                                      T* distance,
                                      size_t L,
                                      size_t N,
                                      size_t bMax,
                                      GSLMatrixType* x,
                                      const GSLVectorType* wX) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq_derivative(const GSLVectorType* covDiag,
                                                 GSLMatrixType* gradient,
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

  static inline double calculateP2(double b, void* params);
  static inline double calculateGradP2(double b, void* params);

  static double c_b(size_t bMax);
  static inline void calculateD2(const gsl_vector* x,
                                 GMToDiracConstWeightOptimizationParams* params,
                                 double* f, gsl_vector* grad);

  static inline void calculateD3(const gsl_vector* x,
                                 GMToDiracConstWeightOptimizationParams* params,
                                 double* f, gsl_vector* grad);

  static inline void correctMean(gsl_vector* x, const gsl_vector* wX, size_t L,
                                 size_t N);

  friend class benchmark_gm_to_dirac_short;
  FRIEND_TEST(
      gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative);
  FRIEND_TEST(
      gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_distance);
  FRIEND_TEST(
      gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_gradient);
};

#include "gm_to_dirac_short.tpp"

template <>
bool gm_to_dirac_short<float>::approximate(const gsl_vector_float* covDiag,
                                           size_t L, size_t N, size_t bMax,
                                           gsl_vector_float* x,
                                           const gsl_vector_float* wX,
                                           GslminimizerResult* result,
                                           const ApproximateOptions& options);

template <>
bool gm_to_dirac_short<double>::approximate(const gsl_vector* covDiag, size_t L,
                                            size_t N, size_t bMax,
                                            gsl_vector* x, const gsl_vector* wX,
                                            GslminimizerResult* result,
                                            const ApproximateOptions& options);

template <>
void gm_to_dirac_short<float>::modified_van_mises_distance_sq(
    const gsl_vector_float* covDiag, float* distance, size_t L, size_t N,
    size_t bMax, gsl_vector_float* x, const gsl_vector_float* wX);

template <>
void gm_to_dirac_short<double>::modified_van_mises_distance_sq(
    const gsl_vector* covDiag, double* distance, size_t L, size_t N,
    size_t bMax, gsl_vector* x, const gsl_vector* wX);

template <>
void gm_to_dirac_short<float>::modified_van_mises_distance_sq_derivative(
    const gsl_vector_float* covDiag, gsl_vector_float* gradient, size_t L,
    size_t N, size_t bMax, gsl_vector_float* x, const gsl_vector_float* wX);

template <>
void gm_to_dirac_short<double>::modified_van_mises_distance_sq_derivative(
    const gsl_vector* covDiag, gsl_vector* gradient, size_t L, size_t N,
    size_t bMax, gsl_vector* x, const gsl_vector* wX);

extern template class gm_to_dirac_short<double>;
extern template class gm_to_dirac_short<float>;

#endif  // GM_TO_DIRAC_SHORT_H
