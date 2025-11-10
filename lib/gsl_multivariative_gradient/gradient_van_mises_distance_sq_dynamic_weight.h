#ifndef GRADIENT_VAN_MISES_DISTANCE_SQ_DYNAMIC_WEIGHT_H
#define GRADIENT_VAN_MISES_DISTANCE_SQ_DYNAMIC_WEIGHT_H

#include "dirac_to_dirac_optimization_params.h"
#include "gsl_minimizer.h"
#include "gsl_multivariative_gradient_template.h"

class gradient_van_mises_distance_sq_dynamic_weight
    : public gsl_multivariative_gradient_template<
          DiracToDiracVariableWeightOptimizationParams> {
 public:
  static double scalarFunctionWrapper(double xI, void* params) {
    GradientParams* gradientParams = (GradientParams*)params;
    gsl_vector* xCopy = gsl_vector_alloc(gradientParams->xOriginal->size);
    gsl_vector_memcpy(xCopy, gradientParams->xOriginal);
    gsl_vector_set(xCopy, gradientParams->index, xI);
    double fValue = gradientParams->f(xCopy, gradientParams->params);
    gsl_vector_free(xCopy);
    return fValue;
  }

  void multivariativeGradient(
      const gsl_vector* x, gsl_vector* grad, VectorFunc f,
      DiracToDiracVariableWeightOptimizationParams* params,
      double h = 1e-6) override {
    GradientParams gp;
    gp.xOriginal = (gsl_vector*)x;  // Non-const because we make a copy inside
    gp.params = params;
    gp.f = f;
    gsl_function F;
    F.function =
        gradient_van_mises_distance_sq_dynamic_weight::scalarFunctionWrapper;
    F.params = &gp;

    for (size_t i = 0; i < x->size; ++i) {
      gp.index = i;

      double result, error;
      gsl_deriv_central(&F, gsl_vector_get(x, i), h, &result, &error);
      gsl_vector_set(grad, i, result);
    }
  }
};

#endif  // GRADIENT_VAN_MISES_DISTANCE_SQ_DYNAMIC_WEIGHT_H