#ifndef DIRAC_TO_DIRAC_APPROX_SHORT_FUNCTION_C_H
#define DIRAC_TO_DIRAC_APPROX_SHORT_FUNCTION_C_H

#include "dirac_to_dirac_approx_short_function.h"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

DLL_EXPORT void* create_dirac_to_dirac_approx_short_function_double() {
  return new dirac_to_dirac_approx_short_function<double>();
}

DLL_EXPORT void delete_dirac_to_dirac_approx_short_function_double(
    void* instance) {
  delete static_cast<dirac_to_dirac_approx_short_function<double>*>(instance);
}

DLL_EXPORT bool dirac_to_dirac_approx_short_function_double_approximate(
    void* instance, const double* y, size_t M, size_t L, size_t N, size_t bMax,
    double* x,
    typename dirac_to_dirac_approx_function_i<double>::wXf wXcallback,
    typename dirac_to_dirac_approx_function_i<double>::wXd wXDcallback,
    GslminimizerResult* result, const ApproximateOptions* options) {
  auto* obj =
      static_cast<dirac_to_dirac_approx_short_function<double>*>(instance);
  return obj->approximate(y, M, L, N, bMax, x, wXcallback, wXDcallback, result,
                          *options);
}

}  // extern "C"

#endif  // DIRAC_TO_DIRAC_APPROX_SHORT_FUNCTION_C_H
