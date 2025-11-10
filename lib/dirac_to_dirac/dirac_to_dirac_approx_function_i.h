#ifndef DIRAC_TO_DIRAC_APPROX_FUNCTION_I_H
#define DIRAC_TO_DIRAC_APPROX_FUNCTION_I_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_matrix_long_double.h>

#include <type_traits>

#include "approximate_options.h"
#include "gsl_minimizer.h"
#include "gsl_vector_matrix_types.h"

/**
 * @brief interface for the gausian mixture to dirac approximation with a custom
 * weight function
 *
 * @tparam T type of the vector (float, double, long double)
 */
template <typename T>
class dirac_to_dirac_approx_function_i {
 public:
  using GSLVectorType = typename GSLTemplateTypeAlias<T>::VectorType;
  using GSLVectorViewType = typename GSLTemplateTypeAlias<T>::VectorViewType;
  using GSLMatrixType = typename GSLTemplateTypeAlias<T>::MatrixType;
  using wXf = void (*)(const double* x, double* res, size_t L, size_t N);
  using wXd = wXf;  // (*)(const double* x, double* res, size_t L, size_t N);

  virtual ~dirac_to_dirac_approx_function_i() = default;

  /**
   * @brief reduce the data points using raw pointers
   *
   * @param y input data points
   * @param M number of input data points
   * @param L number of data points for reduction
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wXcallback callback for the weight function
   * @param wXDcallback callback for the gradient of the weight function
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(const T* y, size_t M, size_t L, size_t N,
                           size_t bMax, T* x, wXf wXcallback, wXd wXDcallback,
                           GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief reduce the data points using gsl vectors
   *
   * @param y input data points
   * @param L number of data points for reduction
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wXcallback callback for the weight function
   * @param wXDcallback callback for the gradient of the weight function
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(const GSLVectorType* y, size_t L, size_t N,
                           size_t bMax, GSLVectorType* x, wXf wXcallback,
                           wXd wXDcallback, GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief reduce the data points using gsl matricies where possible
   *
   * @param y input data points
   * @param L number of data points for reduction
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wXcallback callback for the weight function
   * @param wXDcallback callback for the gradient of the weight function
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(GSLMatrixType* y, size_t L, size_t bMax,
                           GSLMatrixType* x, wXf wXcallback, wXd wXDcallback,
                           GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;
};

#endif  // DIRAC_TO_DIRAC_APPROX_FUNCTION_I_H
