#ifndef GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_C_H
#define GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_C_H

#include "gm_to_dirac_short_standard_normal_deviation.h"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

DLL_EXPORT void* create_gm_to_dirac_short_standard_normal_deviation_double() {
  return new gm_to_dirac_short_standard_normal_deviation<double>();
}

DLL_EXPORT void delete_gm_to_dirac_short_standard_normal_deviation_double(
    void* instance) {
  delete static_cast<gm_to_dirac_short_standard_normal_deviation<double>*>(
      instance);
}

DLL_EXPORT bool gm_to_dirac_short_standard_normal_deviation_double_approximate(
    void* instance, size_t L, size_t N, size_t bMax, double* x,
    const double* wX, GslminimizerResult* result,
    const ApproximateOptions* options) {
  auto* obj = static_cast<gm_to_dirac_short_standard_normal_deviation<double>*>(
      instance);
  return obj->approximate(L, N, bMax, x, wX, result,
                          options ? *options : ApproximateOptions{});
}

DLL_EXPORT void* create_gm_to_dirac_short_standard_normal_deviation_float() {
  return new gm_to_dirac_short_standard_normal_deviation<float>();
}

DLL_EXPORT void delete_gm_to_dirac_short_standard_normal_deviation_float(
    void* instance) {
  delete static_cast<gm_to_dirac_short_standard_normal_deviation<float>*>(
      instance);
}

DLL_EXPORT bool gm_to_dirac_short_standard_normal_deviation_float_approximate(
    void* instance, size_t L, size_t N, size_t bMax, float* x, const float* wX,
    GslminimizerResult* result, const ApproximateOptions* options) {
  auto* obj = static_cast<gm_to_dirac_short_standard_normal_deviation<float>*>(
      instance);
  return obj->approximate(L, N, bMax, x, wX, result,
                          options ? *options : ApproximateOptions{});
}

}  // extern "C"

#endif  // GM_TO_DIRAC_SHORT_STANDARD_NORMAL_DEVIATION_C_H
