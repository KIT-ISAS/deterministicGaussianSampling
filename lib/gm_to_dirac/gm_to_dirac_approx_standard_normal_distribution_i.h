#ifndef GM_TO_DIRAC_APPROX_STANDARD_NORMAL_DISTRIBUTION_I_H
#define GM_TO_DIRAC_APPROX_STANDARD_NORMAL_DISTRIBUTION_I_H

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
class gm_to_dirac_approx_standard_normal_distribution_i {
 public:
  using GSLVectorType = typename GSLTemplateTypeAlias<T>::VectorType;
  using GSLVectorViewType = typename GSLTemplateTypeAlias<T>::VectorViewType;
  using GSLMatrixType = typename GSLTemplateTypeAlias<T>::MatrixType;

  virtual ~gm_to_dirac_approx_standard_normal_distribution_i() = default;

  /**
   * @brief approximate using raw pointers
   *
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(size_t L, size_t N, size_t bMax, T* x, const T* wX,
                           GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param distance pointer to distance value to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                              size_t bMax, T* x,
                                              const T* wX) = 0;

  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param gradient pointer to gradient to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(T* gradient, size_t L,
                                                         size_t N, size_t bMax,
                                                         T* x, const T* wX) = 0;

  /**
   * @brief approximate using gsl vectors
   *
   * @param L number of data points for apprioximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param wX weights for the x data points
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(size_t L, size_t N, size_t bMax, GSLVectorType* x,
                           const GSLVectorType* wX, GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;

  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param distance pointer to distance value to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                              size_t bMax, GSLVectorType* x,
                                              const GSLVectorType* wX) = 0;

  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param gradient pointer to gradient to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(GSLVectorType* gradient, size_t L,
                                                         size_t N, size_t bMax,
                                                         GSLVectorType* x, const GSLVectorType* wX) = 0;

  /**
   * @brief approximate using gsl matricies where possible
   *
   * @param L number of data points for apprioximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param wX weights for the x data points
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual bool approximate(size_t L, size_t N, size_t bMax, GSLMatrixType* x,
                           const GSLVectorType* wX, GslminimizerResult* result,
                           const ApproximateOptions& options) = 0;
                           
  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param distance pointer to distance value to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq(T* distance, size_t L, size_t N,
                                              size_t bMax, GSLMatrixType* x,
                                              const GSLVectorType* wX) = 0;

  /**
   * @brief calculate modified van mises distance based on standard normal
   * deviation and x
   *
   * @param gradient pointer to gradient to be calculated
   * @param L number of data points for approximation
   * @param N dimension of the data
   * @param bMax bMax
   * @param x first guess for the approximation and return value
   * @param result minimizer result
   * @param options options for minimizer
   * @return true, on success, false otherwise
   */
  virtual void modified_van_mises_distance_sq_derivative(GSLMatrixType* gradient, size_t L,
                                                         size_t N, size_t bMax,
                                                         GSLMatrixType* x, const GSLVectorType* wX) = 0;
};

#endif  // GM_TO_DIRAC_APPROX_STANDARD_NORMAL_DISTRIBUTION_I_H
