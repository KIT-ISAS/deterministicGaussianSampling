#ifndef GSL_UTILS_WEIGHT_HELPER_H
#define GSL_UTILS_WEIGHT_HELPER_H

#include <type_traits>
#include <stdexcept>

#include "gsl_vector_matrix_types.h"

template <typename T>
class GSLWeightHelper {
  static_assert(std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Only float and double supported");

 public:
  using GSLVectorType =
      typename GSLTemplateTypeAlias<T>::VectorType;

  GSLWeightHelper(const GSLVectorType* v, size_t size) {
    if (size == 0)
      throw std::runtime_error("Weight vector size must be > 0");

    if (!v) {
      _ownedPtr = GSLTemplateTypeAlias<T>::allocate_vector(size);
      _freeMemory = true;

      const T weight = static_cast<T>(1.0) / static_cast<T>(size);
      for (size_t i = 0; i < size; ++i)
        _ownedPtr->data[i] = weight;

      _ptr = _ownedPtr;
    } else {
      _ptr = v;
    }
  }

  ~GSLWeightHelper() {
    if (_freeMemory && _ownedPtr) {
      GSLTemplateTypeAlias<T>::free_vector(_ownedPtr);
    }
  }

  const GSLVectorType* get() const { return _ptr; }

  operator const GSLVectorType*() const { return _ptr; }

 private:
  const GSLVectorType* _ptr = nullptr;
  GSLVectorType* _ownedPtr = nullptr;
  bool _freeMemory = false;
};

#endif  // GSL_UTILS_WEIGHT_HELPER_H
