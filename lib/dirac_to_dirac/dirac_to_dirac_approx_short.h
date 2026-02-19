#ifndef DIRAC_TO_DIRAC_SHORT_H
#define DIRAC_TO_DIRAC_SHORT_H

#include <gtest/gtest.h>

#include "dirac_to_dirac_approx_i.h"

template <typename T>
class dirac_to_dirac_approx_short : public dirac_to_dirac_approx_i<T> {
 public:
  using GSLVectorType = typename dirac_to_dirac_approx_i<T>::GSLVectorType;
  using GSLVectorViewType =
      typename dirac_to_dirac_approx_i<T>::GSLVectorViewType;
  using GSLMatrixType = typename dirac_to_dirac_approx_i<T>::GSLMatrixType;
  using GSLMatrixViewType =
      typename dirac_to_dirac_approx_i<T>::GSLMatrixViewType;

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
    void modified_van_mises_distance_sq(T* distance,
                                        const T *y,
                                        size_t M,
                                        size_t L,
                                        size_t N,
                                        size_t bMax,
                                        T *x,
                                        const T *wX = nullptr,
                                        const T *wY = nullptr) override;
  // clang-format on

  // clang-format off
    void modified_van_mises_distance_sq_derivative(T* gradient,
                                                   const T *y,
                                                   size_t M,
                                                   size_t L,
                                                   size_t N,
                                                   size_t bMax,
                                                   T *x,
                                                   const T *wX = nullptr,
                                                   const T *wY = nullptr) override;
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
    void modified_van_mises_distance_sq(T* distance,
                                        const GSLVectorType *y,
                                        size_t L,
                                        size_t N,
                                        size_t bMax,
                                        GSLVectorType *x,
                                        const GSLVectorType *wX = nullptr,
                                        const GSLVectorType *wY = nullptr) override;
  // clang-format on

  // clang-format off
    void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                   const GSLVectorType *y,
                                                   size_t L,
                                                   size_t N,
                                                   size_t bMax,
                                                   GSLVectorType *x,
                                                   const GSLVectorType *wX = nullptr,
                                                   const GSLVectorType *wY = nullptr) override;
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

  // clang-format off
    void modified_van_mises_distance_sq(T* distance,
                                        GSLMatrixType *y,
                                        size_t L,
                                        size_t bMax,
                                        GSLMatrixType *x,
                                        const GSLVectorType *wX = nullptr,
                                        const GSLVectorType *wY = nullptr) override;
  // clang-format on

  // clang-format off
    void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                   GSLMatrixType *y,
                                                   size_t L,
                                                   size_t bMax,
                                                   GSLMatrixType *x,
                                                   const GSLVectorType *wX = nullptr,
                                                   const GSLVectorType *wY = nullptr) override;
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

  static inline void correctMean(const GSLVectorType* meanY, GSLVectorType* x,
                                 const GSLVectorType* wX, size_t L, size_t N);

  FRIEND_TEST(
      dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative);
  FRIEND_TEST(
      dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_distance);
  FRIEND_TEST(
      dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_gradient);
  FRIEND_TEST(dirac_to_dirac_approx_short_test_combined,
              parameterized_test_combined);
  friend class benchmark_dirac_to_dirac_approx_short;
};

template <>
bool dirac_to_dirac_approx_short<float>::approximate(
    const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const gsl_vector_float* wX, const gsl_vector_float* wY,
    GslminimizerResult* result, const ApproximateOptions& options);

template <>
bool dirac_to_dirac_approx_short<double>::approximate(
    const gsl_vector* y, size_t L, size_t N, size_t bMax, gsl_vector* x,
    const gsl_vector* wX, const gsl_vector* wY, GslminimizerResult* result,
    const ApproximateOptions& options);

template <>
void dirac_to_dirac_approx_short<float>::modified_van_mises_distance_sq(
    float* distance, const gsl_vector_float* y, size_t L, size_t N, size_t bMax,
    gsl_vector_float* x, const gsl_vector_float* wX,
    const gsl_vector_float* wY);

template <>
void dirac_to_dirac_approx_short<double>::modified_van_mises_distance_sq(
    double* distance, const gsl_vector* y, size_t L, size_t N, size_t bMax,
    gsl_vector* x, const gsl_vector* wX, const gsl_vector* wY);

template <>
void dirac_to_dirac_approx_short<float>::
    modified_van_mises_distance_sq_derivative(gsl_matrix_float* gradient,
                                              const gsl_vector_float* y,
                                              size_t L, size_t N, size_t bMax,
                                              gsl_vector_float* x,
                                              const gsl_vector_float* wX,
                                              const gsl_vector_float* wY);

template <>
void dirac_to_dirac_approx_short<
    double>::modified_van_mises_distance_sq_derivative(gsl_matrix* gradient,
                                                       const gsl_vector* y,
                                                       size_t L, size_t N,
                                                       size_t bMax,
                                                       gsl_vector* x,
                                                       const gsl_vector* wX,
                                                       const gsl_vector* wY);

extern template class dirac_to_dirac_approx_short<double>;
extern template class dirac_to_dirac_approx_short<float>;

#endif  // DIRAC_TO_DIRAC_SHORT_H
