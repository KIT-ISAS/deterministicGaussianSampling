#ifndef DIRAC_TO_DIRAC_APPROX_SHORT_THREAD_C_H
#define DIRAC_TO_DIRAC_APPROX_SHORT_THREAD_C_H

#include "dirac_to_dirac_approx_short_thread.h"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

DLL_EXPORT void* create_dirac_to_dirac_approx_short_thread_double() {
  return new dirac_to_dirac_approx_short_thread<double>();
}

DLL_EXPORT void delete_dirac_to_dirac_approx_short_thread_double(
    void* instance) {
  delete static_cast<dirac_to_dirac_approx_short_thread<double>*>(instance);
}

DLL_EXPORT bool dirac_to_dirac_approx_short_thread_double_approximate(
    void* instance, const double* y, size_t M, size_t L, size_t N, size_t bMax,
    double* x, const double* wX, const double* wY, GslminimizerResult* result,
    const ApproximateOptions* options) {
  auto* obj =
      static_cast<dirac_to_dirac_approx_short_thread<double>*>(instance);
  return obj->approximate(y, M, L, N, bMax, x, wX, wY, result,
                          options ? *options : ApproximateOptions{});
}

DLL_EXPORT void* create_dirac_to_dirac_approx_short_thread_float() {
  return new dirac_to_dirac_approx_short_thread<float>();
}

DLL_EXPORT void delete_dirac_to_dirac_approx_short_thread_float(
    void* instance) {
  delete static_cast<dirac_to_dirac_approx_short_thread<float>*>(instance);
}

DLL_EXPORT bool dirac_to_dirac_approx_short_thread_float_approximate(
    void* instance, const float* y, size_t M, size_t L, size_t N, size_t bMax,
    float* x, const float* wX, const float* wY, GslminimizerResult* result,
    const ApproximateOptions* options) {
  auto* obj = static_cast<dirac_to_dirac_approx_short_thread<float>*>(instance);
  return obj->approximate(y, M, L, N, bMax, x, wX, wY, result,
                          options ? *options : ApproximateOptions{});
}

}  // extern "C"

#endif  // DIRAC_TO_DIRAC_APPROX_SHORT_THREAD_C_H
