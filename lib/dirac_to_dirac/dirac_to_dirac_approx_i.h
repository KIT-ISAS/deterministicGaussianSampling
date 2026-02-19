#ifndef DIRAC_TO_DIRAC_APPROX_I_H
#define DIRAC_TO_DIRAC_APPROX_I_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_matrix_long_double.h>

#include <type_traits>

#include "approximate_options.h"
#include "gsl_minimizer.h"
#include "gsl_vector_matrix_types.h"

/**
 * @brief interface for the gausian mixture to dirac approximation
 *
 * @tparam T type of the vector (float, double)
 */
template <typename T>
class dirac_to_dirac_approx_i {
 public:
  using GSLVectorType = typename GSLTemplateTypeAlias<T>::VectorType;
  using GSLVectorViewType = typename GSLTemplateTypeAlias<T>::VectorViewType;
  using GSLMatrixType = typename GSLTemplateTypeAlias<T>::MatrixType;
  using GSLMatrixViewType = typename GSLTemplateTypeAlias<T>::MatrixViewType;

  virtual ~dirac_to_dirac_approx_i() = default;

  /**
   * @brief reduce the data points using raw pointers
   *
   * @param y input data points
   * @param M number of input data points
   * @param L number of data points for reduction
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(const T* y, size_t M, size_t L, size_t N,
                           size_t bMax, T* x, const T* wX, const T* wY,
                           GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param distance pointer to distance value to be calculated
   * @param y input data points
   * @param M number of elements in y
   * @param L number of elements in x
   * @param N dimension of the data
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(T* distance, const T* y, size_t M,
                                              size_t L, size_t N, size_t bMax,
                                              T* x, const T* wX,
                                              const T* wY) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param gradient pointer to gradient to be calculated
   * @param y input data points
   * @param M number of elements in y
   * @param L number of elements in x
   * @param N dimension of the data
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(
      T* gradient, const T* y, size_t M, size_t L, size_t N, size_t bMax, T* x,
      const T* wX, const T* wY) = 0;

  /**
   * @brief reduce the data points using gsl vectors
   *
   * @param y input data points
   * @param L number of data points for reduction
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(const GSLVectorType* y, size_t L, size_t N,
                           size_t bMax, GSLVectorType* x,
                           const GSLVectorType* wX, const GSLVectorType* wY,
                           GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param distance pointer to distance value to be calculated
   * @param y input data points
   * @param L number of elements in x
   * @param N dimension of the data
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(
      T* distance, const GSLVectorType* y, size_t L, size_t N, size_t bMax,
      GSLVectorType* x, const GSLVectorType* wX, const GSLVectorType* wY) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param gradient pointer to gradient to be calculated
   * @param y input data points
   * @param L number of elements in x
   * @param N dimension of the data
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(
      GSLMatrixType* gradient, const GSLVectorType* y, size_t L, size_t N,
      size_t bMax, GSLVectorType* x, const GSLVectorType* wX,
      const GSLVectorType* wY) = 0;

  /**
   * @brief reduce the data points using gsl matricies where possible
   *
   * @param y input data points
   * @param L number of data points for reduction
   * @param bMax bMax
   * @param x first guess for the reduction and return value
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(GSLMatrixType* y, size_t L, size_t bMax,
                           GSLMatrixType* x, const GSLVectorType* wX,
                           const GSLVectorType* wY, GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param distance pointer to distance value to be calculated
   * @param y input data points
   * @param L number of elements in x
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(T* distance, GSLMatrixType* y,
                                              size_t L, size_t bMax,
                                              GSLMatrixType* x,
                                              const GSLVectorType* wX,
                                              const GSLVectorType* wY) = 0;

  /**
   * @brief calculate modified van mises distance based on x and y
   *
   * @param gradient pointer to gradient to be calculated
   * @param y input data points
   * @param L number of elements in x
   * @param bMax bMax
   * @param x input data points
   * @param wX weights for the x data points
   * @param wY weights for the y data points
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(
      GSLMatrixType* gradient, GSLMatrixType* y, size_t L, size_t bMax,
      GSLMatrixType* x, const GSLVectorType* wX, const GSLVectorType* wY) = 0;
};

#endif  // DIRAC_TO_DIRAC_APPROX_I_H