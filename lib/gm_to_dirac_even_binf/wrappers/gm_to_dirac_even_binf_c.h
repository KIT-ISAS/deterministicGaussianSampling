#ifndef GM_TO_DIRAC_EVEN_BINF_C_H
#define GM_TO_DIRAC_EVEN_BINF_C_H

#include "gm_to_dirac_even_binf.h"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {

// options->bMax is ignored by this class

DLL_EXPORT void* create_gm_to_dirac_even_binf_double() {
  return new gm_to_dirac_even_binf<double>();
}

DLL_EXPORT void delete_gm_to_dirac_even_binf_double(void* instance) {
  delete static_cast<gm_to_dirac_even_binf<double>*>(instance);
}

DLL_EXPORT bool gm_to_dirac_even_binf_double_approximate(
    void* instance, size_t L, size_t N, double* x, const double* wX,
    GslminimizerResult* result, const ApproximateOptions* options) {
  auto* obj = static_cast<gm_to_dirac_even_binf<double>*>(instance);
  return obj->approximate(L, N, x, wX, result,
                          options ? *options : ApproximateOptions{});
}

DLL_EXPORT void gm_to_dirac_even_binf_double_modified_van_mises_distance_sq(
    void* instance, double* distance, size_t L, size_t N, double* x,
    const double* wX) {
  auto* obj = static_cast<gm_to_dirac_even_binf<double>*>(instance);
  obj->modified_van_mises_distance_sq(distance, L, N, x, wX);
}

DLL_EXPORT void
gm_to_dirac_even_binf_double_modified_van_mises_distance_sq_derivative(
    void* instance, double* gradient, size_t L, size_t N, double* x,
    const double* wX) {
  auto* obj = static_cast<gm_to_dirac_even_binf<double>*>(instance);
  obj->modified_van_mises_distance_sq_derivative(gradient, L, N, x, wX);
}

}  // extern "C"

#endif  // GM_TO_DIRAC_EVEN_BINF_C_H
