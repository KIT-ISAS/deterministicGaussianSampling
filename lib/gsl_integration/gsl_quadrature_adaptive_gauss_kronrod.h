#ifndef GSL_QUADRATURE_ADAPTIVE_GAUSS_KRONROD_H
#define GSL_QUADRATURE_ADAPTIVE_GAUSS_KRONROD_H

#include <gsl/gsl_integration.h>
#include <omp.h>

#include <cassert>

/**
 * @brief helper class to use GSL adaptive Gauss-Kronrod quadrature
 *
 */
class GslQuadratureAdaptiveGaussKronrod {
 public:
  using fT = double (*)(double x, void* params);

  /**
   * @brief Construct a new Gsl Quadrature Adaptive Gauss Kronrod object
   *
   * @param epsabs absolute error tolerance
   * @param epsrel relative error tolerance
   */
  GslQuadratureAdaptiveGaussKronrod(double epsabs = 0.00, double epsrel = 1e-10)
      : epsabs(epsabs), epsrel(epsrel) {
    thread_num = (size_t)omp_get_max_threads();
    workspace = new gsl_integration_workspace*[thread_num];
    for (size_t i = 0; i < thread_num; ++i) {
      workspace[i] = gsl_integration_workspace_alloc(1000);
    }
  }

  /**
   * @brief Destroy the Gsl Quadrature Adaptive Gauss Kronrod object
   *
   */
  ~GslQuadratureAdaptiveGaussKronrod() {
    if (workspace) {
      for (size_t i = 0; i < thread_num; ++i) {
        gsl_integration_workspace_free(workspace[i]);
      }
      delete[] workspace;
    }
  }

  /**
   * @brief perform the integration using GSL adaptive Gauss-Kronrod
   *
   * @param f function to be integrated
   * @param params parameters for the function to be integrated
   * @param lowerLimit lower limit of the integration
   * @param upperLimit upper limit of the integration
   * @param result result of the integration
   * @param abserr absolute error of the integration
   * @return true, if the integration was successful within the given
   * tolerances, false otherwise
   */
  inline bool integrate(fT f, void* params, double lowerLimit,
                        double upperLimit, double* result,
                        double* abserr) const {
    gsl_function F;
    F.function = f;
    F.params = params;
    int tid = omp_get_thread_num();
    int status =
        gsl_integration_qag(&F, lowerLimit, upperLimit, epsabs, epsrel, 1000,
                            GSL_INTEG_GAUSS31, workspace[tid], result, abserr);
    return status == GSL_SUCCESS;
  }

 private:
  gsl_integration_workspace** workspace;
  size_t thread_num;
  double epsabs;
  double epsrel;
};

#endif  // GSL_QUADRATURE_ADAPTIVE_GAUSS_KRONROD_H