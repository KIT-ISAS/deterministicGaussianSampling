#ifndef GSL_VECTOR_MATRIX_TYPES_H
#define GSL_VECTOR_MATRIX_TYPES_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <cstring>
#include <type_traits>
#include <vector>

// Helper class to handle GSL types and views for float and double only
template <typename T>
class GSLTemplateTypeAlias {
 public:
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "GSLTemplateTypeAlias only supports float and double");

  using MatrixType =
      typename std::conditional<std::is_same<T, double>::value, gsl_matrix,
                                gsl_matrix_float>::type;

  using VectorType =
      typename std::conditional<std::is_same<T, double>::value, gsl_vector,
                                gsl_vector_float>::type;

  using VectorViewType =
      typename std::conditional<std::is_same<T, double>::value, gsl_vector_view,
                                gsl_vector_float_view>::type;

  // Static method to create a vector view from a raw pointer
  static VectorViewType vector_view_from_array(const T* data, size_t size) {
    return vector_view_from_array(const_cast<T*>(data), size);
  }
  static VectorViewType vector_view_from_array(T* data, size_t size) {
    if constexpr (std::is_same<T, double>::value) {
      return gsl_vector_view_array(data, size);
    } else {
      return gsl_vector_float_view_array(data, size);
    }
  }

  // Static method to flatten a matrix into a vector view
  static VectorViewType flatten_matrix_to_vector(MatrixType* m) {
    return vector_view_from_array(m->data, m->size1 * m->size2);
  }
  static VectorViewType flatten_matrix_to_vector(const MatrixType* m) {
    return vector_view_from_array(m->data, m->size1 * m->size2);
  }

  // Static method to allocate / free a vector
  static VectorType* allocate_vector(size_t size) {
    if constexpr (std::is_same<T, double>::value) {
      return gsl_vector_alloc(size);
    } else {
      return gsl_vector_float_alloc(size);
    }
  }
  static void free_vector(VectorType* v) {
    if constexpr (std::is_same<T, double>::value) {
      gsl_vector_free(v);
    } else {
      gsl_vector_float_free(v);
    }
  }

  // Static method to allocate / free a matrix
  static MatrixType* allocate_matrix(size_t rows, size_t cols) {
    if constexpr (std::is_same<T, double>::value) {
      return gsl_matrix_alloc(rows, cols);
    } else {
      return gsl_matrix_float_alloc(rows, cols);
    }
  }
  static void free_matrix(MatrixType* m) {
    if constexpr (std::is_same<T, double>::value) {
      gsl_matrix_free(m);
    } else {
      gsl_matrix_float_free(m);
    }
  }
};

#endif  // GSL_VECTOR_MATRIX_TYPES_H
