#ifndef DIRAC_TO_DIRAC_SHORT_THREAD_H
#define DIRAC_TO_DIRAC_SHORT_THREAD_H

#include <gtest/gtest.h>

#include "dirac_to_dirac_approx_i.h"

template <typename T>
class dirac_to_dirac_approx_short_thread : public dirac_to_dirac_approx_i<T> {
 public:
  using GSLVectorType = typename dirac_to_dirac_approx_i<T>::GSLVectorType;
  using GSLVectorViewType =
      typename dirac_to_dirac_approx_i<T>::GSLVectorViewType;
  using GSLMatrixType = typename dirac_to_dirac_approx_i<T>::GSLMatrixType;

  // clang-format off
    bool approximate(const T* y,
                      size_t M,
                      size_t L,
                      size_t N,
                      size_t bMax,
                      T* x,
                      const T* wX = nullptr,
                      const T* wY = nullptr,
                      GslminimizerResult* result = nullptr,
                      const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
    bool approximate(const GSLVectorType* y,
                      size_t L,
                      size_t N,
                      size_t bMax,
                      GSLVectorType* x,
                      const GSLVectorType* wX = nullptr,
                      const GSLVectorType* wY = nullptr,
                      GslminimizerResult* result = nullptr,
                      const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
    bool approximate(GSLMatrixType* y,
                      size_t L,
                      size_t bMax,
                      GSLMatrixType* x,
                      const GSLVectorType* wX = nullptr,
                      const GSLVectorType* wY = nullptr,
                      GslminimizerResult* result = nullptr,
                      const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

 private:
  static double c_b(size_t bMax);
  static double modified_van_mises_distance_sq(const gsl_vector* x,
                                               void* params);
  static void modified_van_mises_distance_sq_derivative(const gsl_vector* x,
                                                        void* params,
                                                        gsl_vector* grad);
  static void combined_distance_metric(const gsl_vector* x, void* params,
                                       double* f, gsl_vector* grad);

  static inline void correctMean(const gsl_vector* meanY, gsl_vector* x,
                                 const gsl_vector* wX, size_t L, size_t N);

  FRIEND_TEST(
      dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative);
  FRIEND_TEST(dirac_to_dirac_approx_short_test_combined,
              parameterized_test_combined);
  friend class testero;
  friend class benchmark_dirac_to_dirac_approx_short_thread;
};

template <>
bool dirac_to_dirac_approx_short_thread<float>::approximate(
    const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const GSLVectorType* wX, const GSLVectorType* wY,
    GslminimizerResult* result, const ApproximateOptions& options);

template <>
bool dirac_to_dirac_approx_short_thread<double>::approximate(
    const gsl_vector* y, size_t L, size_t N, size_t bMax, gsl_vector* x,
    const GSLVectorType* wX, const GSLVectorType* wY,
    GslminimizerResult* result, const ApproximateOptions& options);

extern template class dirac_to_dirac_approx_short_thread<double>;
extern template class dirac_to_dirac_approx_short_thread<float>;

#endif  // DIRAC_TO_DIRAC_SHORT_H
