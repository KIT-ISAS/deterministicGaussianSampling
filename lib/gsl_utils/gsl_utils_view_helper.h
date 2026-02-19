#ifndef GSL_UTILS_VIEW_HELPER_H
#define GSL_UTILS_VIEW_HELPER_H

#include <type_traits>

#include "gsl_vector_matrix_types.h"

template <typename T, bool IsMatrix>
class GSLViewHelper {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Only float and double supported");

 public:
  using GSLVectorType = typename GSLTemplateTypeAlias<T>::VectorType;
  using GSLVectorViewType = typename GSLTemplateTypeAlias<T>::VectorViewType;
  using GSLMatrixType = typename GSLTemplateTypeAlias<T>::MatrixType;
  using GSLMatrixViewType = typename GSLTemplateTypeAlias<T>::MatrixViewType;

  using GSLType =
      typename std::conditional<IsMatrix, GSLMatrixType, GSLVectorType>::type;

  using ViewType = typename std::conditional<IsMatrix, GSLMatrixViewType,
                                             GSLVectorViewType>::type;

  /**************************************************************************/
  /********************************* pointer ********************************/
  /**************************************************************************/
  template <typename U>
  GSLViewHelper(const U* ptr, size_t size) {
    static_assert(!IsMatrix, "Vector constructor used for matrix");
    static_assert(is_float_or_double<U>(), "Only float/double allowed");

    if (!ptr) {
      _ptr = nullptr;
      return;
    }

    construct_vector_from_ptr(ptr, size);
  }

  template <typename U>
  GSLViewHelper(const U* ptr, size_t rows, size_t cols) {
    static_assert(IsMatrix, "Matrix constructor used for vector");
    static_assert(is_float_or_double<U>(), "Only float/double allowed");

    if (!ptr) {
      _ptr = nullptr;
      return;
    }

    construct_matrix_from_ptr(ptr, rows, cols);
  }

  /**************************************************************************/
  /********************************* vector *********************************/
  /**************************************************************************/
  GSLViewHelper(const gsl_vector* v, size_t rows = 0, size_t cols = 0) {
    if (!v) {
      _ptr = nullptr;
      return;
    }

    if constexpr (!IsMatrix) {
      // internal storage = vector
      construct_vector_from_vector<double>(v);
    } else {
      // internal storage = matrix
      construct_matrix_from_vector<double>(v, rows, cols);
    }
  }

  GSLViewHelper(const gsl_vector_float* v, size_t rows = 0, size_t cols = 0) {
    if (!v) {
      _ptr = nullptr;
      return;
    }

    if constexpr (!IsMatrix) {
      // internal storage = vector
      construct_vector_from_vector<float>(v);
    } else {
      // internal storage = matrix
      construct_matrix_from_vector<float>(v, rows, cols);
    }
  }

  /**************************************************************************/
  /********************************* matrix *********************************/
  /**************************************************************************/
  GSLViewHelper(const gsl_matrix* m) {
    if (!m) {
      _ptr = nullptr;
      return;
    }

    if constexpr (IsMatrix) {
      // internal storage = matrix
      construct_matrix_from_matrix<double>(m);
    } else {
      // internal storage = vector
      construct_vector_from_matrix<double>(m);
    }
  }

  GSLViewHelper(const gsl_matrix_float* m) {
    if (!m) {
      _ptr = nullptr;
      return;
    }

    if constexpr (IsMatrix) {
      // internal storage = matrix
      construct_matrix_from_matrix<float>(m);
    } else {
      // internal storage = vector
      construct_vector_from_matrix<float>(m);
    }
  }

  /**************************************************************************/
  /******************************* destructor *******************************/
  /**************************************************************************/
  ~GSLViewHelper() {
    if (!_freeMemory || !_ptr) return;

    if constexpr (IsMatrix)
      GSLTemplateTypeAlias<T>::free_matrix(_ptr);
    else
      GSLTemplateTypeAlias<T>::free_vector(_ptr);
  }

  /**************************************************************************/
  /********************************* access *********************************/
  /**************************************************************************/
  GSLType* get() { return _ptr; }
  const GSLType* get() const { return _ptr; }

  operator GSLType*() { return _ptr; }
  operator const GSLType*() const { return _ptr; }

 private:
  template <typename U>
  void construct_vector_from_ptr(const U* ptr, size_t size) {
    if constexpr (std::is_same<U, T>::value) {
      _view = GSLTemplateTypeAlias<T>::vector_view_from_array(ptr, size);
      _ptr = &_view.vector;
    } else {
      _ptr = GSLTemplateTypeAlias<T>::allocate_vector(size);
      _freeMemory = true;

      for (size_t i = 0; i < size; ++i) _ptr->data[i] = static_cast<T>(ptr[i]);
    }
  }

  template <typename U>
  void construct_vector_from_vector(
      const typename GSLTemplateTypeAlias<U>::VectorType* v) {
    if (!v) {
      _ptr = nullptr;
      return;
    }

    if constexpr (std::is_same<U, T>::value) {
      _ptr = const_cast<GSLType*>(v);
    } else {
      _ptr = GSLTemplateTypeAlias<T>::allocate_vector(v->size);
      _freeMemory = true;

      for (size_t i = 0; i < v->size; ++i)
        _ptr->data[i] = static_cast<T>(v->data[i]);
    }
  }

  template <typename U>
  void construct_matrix_from_vector(
      const typename GSLTemplateTypeAlias<U>::VectorType* v, size_t rows,
      size_t cols) {
    if (!v) {
      _ptr = nullptr;
      return;
    }

    if (rows == 0 || cols == 0)
      throw std::runtime_error(
          "Matrix construction from vector requires rows and cols");

    if (v->size != rows * cols)
      throw std::runtime_error("Size mismatch in reshape");

    if constexpr (std::is_same<T, U>::value) {
      _view =
          GSLTemplateTypeAlias<T>::matrix_view_from_array(v->data, rows, cols);

      _ptr = &_view.matrix;
    } else {
      _ptr = GSLTemplateTypeAlias<T>::allocate_matrix(rows, cols);
      _freeMemory = true;

      for (size_t i = 0; i < v->size; ++i)
        _ptr->data[i] = static_cast<T>(v->data[i]);
    }
  }

  template <typename U>
  void construct_matrix_from_ptr(const U* ptr, size_t rows, size_t cols) {
    if constexpr (std::is_same<U, T>::value) {
      _view = GSLTemplateTypeAlias<T>::matrix_view_from_array(ptr, rows, cols);
      _ptr = &_view.matrix;
    } else {
      _ptr = GSLTemplateTypeAlias<T>::allocate_matrix(rows, cols);
      _freeMemory = true;

      size_t total = rows * cols;
      for (size_t i = 0; i < total; ++i) _ptr->data[i] = static_cast<T>(ptr[i]);
    }
  }

  template <typename U>
  void construct_matrix_from_matrix(
      const typename GSLTemplateTypeAlias<U>::MatrixType* m) {
    if (!m) {
      _ptr = nullptr;
      return;
    }

    if constexpr (std::is_same<U, T>::value) {
      _ptr = const_cast<GSLType*>(m);
    } else {
      _ptr = GSLTemplateTypeAlias<T>::allocate_matrix(m->size1, m->size2);
      _freeMemory = true;

      size_t total = m->size1 * m->size2;
      for (size_t i = 0; i < total; ++i)
        _ptr->data[i] = static_cast<T>(m->data[i]);
    }
  }

  template <typename U>
  void construct_vector_from_matrix(
      const typename GSLTemplateTypeAlias<U>::MatrixType* m) {
    if (!m) {
      _ptr = nullptr;
      return;
    }

    if constexpr (std::is_same<T, U>::value) {
      _view = GSLTemplateTypeAlias<T>::flatten_matrix_to_vector(m);
      _ptr = &_view.vector;
    } else {
      const size_t total = m->size1 * m->size2;
      _ptr = GSLTemplateTypeAlias<T>::allocate_vector(total);
      _freeMemory = true;

      for (size_t i = 0; i < total; ++i)
        _ptr->data[i] = static_cast<T>(m->data[i]);
    }
  }

  template <typename U>
  static constexpr bool is_float_or_double() {
    return std::is_same<U, float>::value || std::is_same<U, double>::value;
  }

 private:
  GSLType* _ptr = nullptr;
  bool _freeMemory = false;

  ViewType _view{};
};

template <typename T>
using GSLVectorView = GSLViewHelper<T, false>;

template <typename T>
using GSLMatrixView = GSLViewHelper<T, true>;

#endif  // GSL_UTILS_VIEW_HELPER_H