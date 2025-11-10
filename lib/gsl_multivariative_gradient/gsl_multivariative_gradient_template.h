#ifndef GSL_MULTIVARIATIVE_GRADIENT_TEMPLATE_H
#define GSL_MULTIVARIATIVE_GRADIENT_TEMPLATE_H

#include <gsl/gsl_deriv.h>
#include <gsl/gsl_vector.h>

typedef double (*VectorFunc)(const gsl_vector* x, void* params);

template <typename S>
class gsl_multivariative_gradient_template {
 public:
  virtual void multivariativeGradient(const gsl_vector* x, gsl_vector* grad,
                                      VectorFunc f, S* params,
                                      double h = 1e-9) = 0;

 protected:
  struct GradientParams {
    S* params;
    gsl_vector* xOriginal;
    size_t index;
    VectorFunc f;
  };
};

#endif  // GSL_MULTIVARIATIVE_GRADIENT_TEMPLATE_H