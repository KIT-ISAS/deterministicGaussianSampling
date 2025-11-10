#ifndef GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_H
#define GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_H

#include <gtest/gtest.h>

#include "gm_to_dirac_approx_standard_normal_distribution_i.h"
#include "gm_to_dirac_optimization_params.h"

template <typename T>
class gm_to_dirac_short_standard_normal_deviation
    : public gm_to_dirac_approx_standard_normal_distribution_i<T> {
 public:
  using GSLVectorType =
      typename gm_to_dirac_approx_standard_normal_distribution_i<
          T>::GSLVectorType;
  using GSLVectorViewType =
      typename gm_to_dirac_approx_standard_normal_distribution_i<
          T>::GSLVectorViewType;
  using GSLMatrixType =
      typename gm_to_dirac_approx_standard_normal_distribution_i<
          T>::GSLMatrixType;

  // clang-format off
  bool approximate(size_t L,
                    size_t N,
                    size_t bMax,
                    T* x,
                    const T* wX,
                    GslminimizerResult* result = nullptr,
                    const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  bool approximate(size_t L,
                    size_t N,
                    size_t bMax,
                    GSLVectorType* x,
                    const GSLVectorType* wX = nullptr,
                    GslminimizerResult* result = nullptr,
                    const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on

  // clang-format off
  bool approximate(size_t L,
                    size_t N,
                    size_t bMax,
                    GSLMatrixType* x,
                    const GSLVectorType* wX = nullptr,
                    GslminimizerResult* result = nullptr,
                    const ApproximateOptions& options = ApproximateOptions{}) override;
  // clang-format on
};

extern template class gm_to_dirac_short_standard_normal_deviation<double>;
extern template class gm_to_dirac_short_standard_normal_deviation<float>;

#endif  // GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_H
