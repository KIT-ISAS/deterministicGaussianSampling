#ifndef DIRAC_TO_DIRAC_SHORT_FUNCTION_H
#define DIRAC_TO_DIRAC_SHORT_FUNCTION_H

#include <gtest/gtest.h>

#include "dirac_to_dirac_approx_function_i.h"

template <typename T>
class dirac_to_dirac_approx_short_function
    : public dirac_to_dirac_approx_function_i<T> {
 public:
  using GSLVectorType =
      typename dirac_to_dirac_approx_function_i<T>::GSLVectorType;
  using GSLVectorViewType =
      typename dirac_to_dirac_approx_function_i<T>::GSLVectorViewType;
  using GSLMatrixType =
      typename dirac_to_dirac_approx_function_i<T>::GSLMatrixType;
  using wXf = typename dirac_to_dirac_approx_function_i<T>::wXf;
  using wXd = typename dirac_to_dirac_approx_function_i<T>::wXd;

  // clang-format off
  bool approximate(const T* y,
                    size_t M,
                    size_t L,
                    size_t N,
                    size_t bMax,
                    T* x,
                    wXf wXcallback,
                    wXd wXDcallback,
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
                                      wXf wXcallback,
                                      wXd wXDcallback) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq_derivative(T* gradient,
                                                 const T *y,
                                                 size_t M,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 T *x,
                                                 wXf wXcallback,
                                                 wXd wXDcallback) override;

  // clang-format off
  bool approximate(const GSLVectorType* y,
                    size_t L,
                    size_t N,
                    size_t bMax,
                    GSLVectorType* x,
                    wXf wXcallback,
                    wXd wXDcallback,
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
                                      wXf wXcallback,
                                      wXd wXDcallback) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                 const GSLVectorType *y,
                                                 size_t L,
                                                 size_t N,
                                                 size_t bMax,
                                                 GSLVectorType *x,
                                                 wXf wXcallback,
                                                 wXd wXDcallback) override;
  // clang-format on

  // clang-format off
  bool approximate(GSLMatrixType* y,
                    size_t L,
                    size_t bMax,
                    GSLMatrixType* x,
                    wXf wXcallback,
                    wXd wXDeriv,
                    GslminimizerResult* result = nullptr,
                    const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq(T* distance,
                                      GSLMatrixType *y,
                                      size_t L,
                                      size_t bMax,
                                      GSLMatrixType *x,
                                      wXf wXcallback,
                                      wXd wXDcallback) override;
  // clang-format on

  // clang-format off
  void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient,
                                                 GSLMatrixType *y,
                                                 size_t L,
                                                 size_t bMax,
                                                 GSLMatrixType *x,
                                                 wXf wXcallback,
                                                 wXd wXDcallback) override;
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

  FRIEND_TEST(
      dirac_to_dirac_approx_short_function_test_modified_van_mises_distance_sq_derivative,
      parameterized_test_modified_van_mises_distance_sq_derivative);
};

template <>
bool dirac_to_dirac_approx_short_function<double>::approximate(
    const gsl_vector* y, size_t L, size_t N, size_t bMax, gsl_vector* x,
    wXf wXcallback, wXd wXDcallback, GslminimizerResult* result,
    const ApproximateOptions& options);

template <>
void dirac_to_dirac_approx_short_function<
    double>::modified_van_mises_distance_sq(double* distance,
                                            const gsl_vector* y, size_t L,
                                            size_t N, size_t bMax,
                                            gsl_vector* x, wXf wXcallback,
                                            wXd wXDcallback);

template <>
void dirac_to_dirac_approx_short_function<
    double>::modified_van_mises_distance_sq_derivative(gsl_matrix* gradient,
                                                       const gsl_vector* y,
                                                       size_t L, size_t N,
                                                       size_t bMax,
                                                       gsl_vector* x,
                                                       wXf wXcallback,
                                                       wXd wXDcallback);

extern template class dirac_to_dirac_approx_short_function<double>;

#endif  // DIRAC_TO_DIRAC_SHORT_FUNCTION_H