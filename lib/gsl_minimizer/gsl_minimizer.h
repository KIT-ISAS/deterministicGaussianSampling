#ifndef GSL_MINIMIZER_H
#define GSL_MINIMIZER_H

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "gsl/gsl_multimin.h"
#include "gsl/gsl_vector.h"

/**
 * @brief base struct minimizer needs to be setup with
 *
 * L * N is the number of parameters to be optimized
 *
 */
struct GslMinimizerOptimizationParams {
 public:
  GslMinimizerOptimizationParams(size_t L, size_t N) : L(L), N(N) {
    assert(L > 0);
    assert(N > 0);
  }

  size_t L;
  size_t N;
};

/**
 * @brief struct to hold the result of the minimization
 */
struct GslminimizerResult {
  double initalStepSize = 0.00;
  double stepTolerance = 0.00;
  double lastXtolAbs = 0.00;
  double lastXtolRel = 0.00;
  double lastFtolAbs = 0.00;
  double lastFtolRel = 0.00;
  double lastGtol = 0.00;
  double xtolAbs = 0.00;
  double xtolRel = 0.00;
  double ftolAbs = 0.00;
  double ftolRel = 0.00;
  double gtol = 0.00;
  size_t iterations = 0;
  size_t maxIterations = 0;
  size_t elapsedTimeMicro = 0;
};
std::ostream& operator<<(std::ostream& os, const GslminimizerResult& s);

/**
 * @brief helper class to use GSL fdf minimizer
 *
 */
class gsl_minimizer {
 public:
  /**
   * @brief Construct a new gsl minimizer object
   *
   * @param maxIterations max number of iterations to run
   * @param xtolAbs absolute tolerance for x
   * @param xtolRel relative tolerance for x
   * @param ftolAbs absolute tolerance for f
   * @param ftolRel relative tolerance for f
   * @param gtol tolerance for gradient-norm
   * @param optimiziationParams parameters for optimization passed to the
   * f/df/fdf-functions
   * @param f objective function
   * @param df gradient function
   * @param fdf function for f and df
   */
  gsl_minimizer(size_t maxIterations, double xtolAbs, double xtolRel,
                double ftolAbs, double ftolRel, double gtol,
                GslMinimizerOptimizationParams* optimiziationParams,
                double (*f)(const gsl_vector* x, void* params),
                void (*df)(const gsl_vector* x, void* params, gsl_vector* df),
                void (*fdf)(const gsl_vector* x, void* params, double* f,
                            gsl_vector* df))
      : maxIterations(maxIterations),
        xtolAbs(xtolAbs),
        xtolRel(xtolRel),
        ftolAbs(ftolAbs),
        ftolRel(ftolRel),
        gtol(gtol) {
    objectiveFunctionDefs.f = f;
    objectiveFunctionDefs.df = df;
    objectiveFunctionDefs.fdf = fdf;
    setOptimiziationParams(optimiziationParams);
  }

  /**
   * @brief set the maximum number of iterations to run
   *
   * @param iterations maximum number of iterations
   */
  void setMaxIterations(size_t iterations);

  /**
   * @brief set the absolute and relative tolerance for x
   *
   * @param ftolAbs absolute tolerance for f
   * @param ftolRel relative tolerance for f
   */
  void setFtol(double ftolAbs, double ftolRel);

  /**
   * @brief set the absolute and relative tolerance for x
   *
   * @param xtolAbs absolute tolerance for x
   * @param xtolRel relative tolerance for x
   */
  void setXtol(double xtolAbs, double xtolRel);

  /**
   * @brief set the tolerance for gradient-norm
   *
   * @param gtol tolerance for gradient-norm
   */
  void setGtol(double gtol);

  /**
   * @brief set the optimization passed to the f/df/fdf-functions
   *
   * @param optimiziationParams optimization parameters for the fdf-functions
   */
  void setOptimiziationParams(
      GslMinimizerOptimizationParams* optimiziationParams);

  /**
   * @brief minimize the function f with the given parameters
   *
   * @param x initial guess for the parameters to be optimized
   * @return int GSL_SUCCESS if successful, error code otherwise
   */
  int minimize(gsl_vector* x, GslminimizerResult* result = nullptr,
               bool verbose = false);

 protected:
  int checkXtol(const gsl_multimin_fdfminimizer* minimizer, gsl_vector* prevX,
                const double xtolAbs, const double xtolRel,
                GslminimizerResult* result);
  int checkFtol(const gsl_multimin_fdfminimizer* minimizer, double* prevF,
                const double ftolAbs, const double ftolRel,
                GslminimizerResult* result);
  int checkGtol(const gsl_multimin_fdfminimizer* minimizer, const double gtol,
                GslminimizerResult* result);

  GslMinimizerOptimizationParams* optimiziationParams;
  gsl_multimin_function_fdf objectiveFunctionDefs;

  size_t maxIterations;
  double xtolAbs, xtolRel;
  double ftolAbs, ftolRel;
  double gtol;
};

#endif  // GSL_MINIMIZER_H