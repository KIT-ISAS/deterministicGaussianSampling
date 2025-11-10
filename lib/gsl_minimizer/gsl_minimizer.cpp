#include "gsl_minimizer.h"

#include <gsl/gsl_blas.h>

#include <iomanip>
#include <sstream>

// #define SHOW_DEBUG

inline std::string format_elapsed_time(size_t elapsedMicro) {
  std::ostringstream oss;
  oss << std::fixed;

  if (elapsedMicro < 1'000) {
    oss << std::setprecision(1) << elapsedMicro << " µs";
  } else if (elapsedMicro < 1'000'000) {
    double ms = static_cast<double>(elapsedMicro) / 1'000.0;
    oss << std::setprecision(3) << ms << " ms";
  } else if (elapsedMicro < 60'000'000) {
    double sec = static_cast<double>(elapsedMicro) / 1'000'000.0;
    oss << std::setprecision(3) << sec << " s";
  } else {
    int minutes = static_cast<int>(elapsedMicro / 60'000'000);
    double seconds =
        static_cast<double>(elapsedMicro % 60'000'000) / 1'000'000.0;
    oss << minutes << "m " << std::setprecision(2) << seconds << "s";
  }

  return oss.str();
}

#define PRINT_COMP_SYMBOL(a, b) ((a) < (b) ? '<' : ((a) == (b) ? '=' : '>'))
std::ostream& operator<<(std::ostream& os, const GslminimizerResult& s) {
  std::ios oldState(nullptr);
  oldState.copyfmt(os);
  // clang-format off
  os << std::scientific << std::setprecision(6);
  os << "GslminimizerResult:\n"
     << "   initialStepSize: " << s.initalStepSize << "\n"
     << "   stepTolerance:   " << s.stepTolerance << "\n\n"
     << "   |x - x'|:               " << s.lastXtolAbs << " " << PRINT_COMP_SYMBOL(s.lastXtolAbs, s.xtolAbs) << " " << s.xtolAbs << "\n"
     << "   |x - x'|/|x'|:          " << s.lastXtolRel << " " << PRINT_COMP_SYMBOL(s.lastXtolRel, s.xtolRel) << " " << s.xtolRel << "\n"
     << "   |f(x) - f(x')|:         " << s.lastFtolAbs << " " << PRINT_COMP_SYMBOL(s.lastFtolAbs, s.ftolAbs) << " " << s.ftolAbs << "\n"
     << "   |f(x) - f(x')|/|f(x')|: " << s.lastFtolRel << " " << PRINT_COMP_SYMBOL(s.lastFtolRel, s.ftolRel) << " " << s.ftolRel << "\n"
     << "   |g(x)|:                 " << s.lastGtol << " " << PRINT_COMP_SYMBOL(s.lastGtol, s.gtol) << " " << s.gtol << "\n\n"
     << "   iterations: " << s.iterations << " of " << s.maxIterations << "\n\n";
     os << "   timeTaken: " << format_elapsed_time(s.elapsedTimeMicro) << "\n...\n";
     os.copyfmt(oldState);
  // clang-format on
  return os;
}

void gsl_minimizer::setMaxIterations(size_t iterations) {
  maxIterations = iterations;
}

void gsl_minimizer::setFtol(double ftolAbs, double ftolRel) {
  this->ftolAbs = ftolAbs;
  this->ftolRel = ftolRel;
}

void gsl_minimizer::setXtol(double xtolAbs, double xtolRel) {
  this->xtolAbs = xtolAbs;
  this->xtolRel = xtolRel;
}

void gsl_minimizer::setGtol(double gtol) { this->gtol = gtol; }

void gsl_minimizer::setOptimiziationParams(
    GslMinimizerOptimizationParams* optimiziationParams) {
  this->optimiziationParams = optimiziationParams;

  objectiveFunctionDefs.n = optimiziationParams->L * optimiziationParams->N;
  objectiveFunctionDefs.params = (void*)optimiziationParams;
}

int gsl_minimizer::minimize(gsl_vector* x, GslminimizerResult* result,
                            bool verbose) {
  auto start = std::chrono::high_resolution_clock::now();
  gsl_multimin_fdfminimizer* minimizer = gsl_multimin_fdfminimizer_alloc(
      gsl_multimin_fdfminimizer_vector_bfgs2,
      optimiziationParams->N * optimiziationParams->L);

  constexpr double initialStepSize = 0.1;
  constexpr double tol = 0.15;

  if (result) {
    result->xtolAbs = xtolAbs;
    result->xtolRel = xtolRel;
    result->ftolAbs = ftolAbs;
    result->ftolRel = ftolRel;
    result->gtol = gtol;
    result->maxIterations = maxIterations;
    result->initalStepSize = initialStepSize;
    result->stepTolerance = tol;
  }

  gsl_multimin_fdfminimizer_set(minimizer, &objectiveFunctionDefs, x,
                                initialStepSize, tol);

  int status;
  size_t iter = 0;

  double previousDistance = gsl_multimin_fdfminimizer_minimum(minimizer);
  gsl_vector* previousX = gsl_vector_alloc(x->size);
  gsl_vector_memcpy(previousX, minimizer->x);
  gsl_vector* xCopy = gsl_vector_alloc(x->size);

  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(minimizer);

    if (status == GSL_ENOPROG) {
      if (verbose)
        std::cout << ">>> GSL optimizer: No progress... done step[" << iter
                  << "]\n";
      status = GSL_SUCCESS;
    } else if (status) {
      if (verbose)
        std::cout << ">>> Error in GSL optimizer: " << gsl_strerror(status)
                  << "\n";
      break;
    } else {
      status = GSL_CONTINUE;

      // f delta stop crit
      if (checkFtol(minimizer, &previousDistance, ftolAbs, ftolRel, result) ==
          GSL_SUCCESS) {
        if (verbose)
          std::cout << ">>> Converged: Delta F is within tolerance step["
                    << iter << "]\n";
        status = GSL_SUCCESS;
        break;
      }

      // x delta stop crit
      if (checkXtol(minimizer, previousX, xtolAbs, xtolRel, result) ==
          GSL_SUCCESS) {
        if (verbose)
          std::cout << ">>> Converged: Delta X is within tolerance step["
                    << iter << "]\n";
        status = GSL_SUCCESS;
        break;
      }

      // gradient delta stop crit
      if (checkGtol(minimizer, gtol, result) == GSL_SUCCESS) {
        if (verbose)
          std::cout << ">>> Converged: Gradient is within tolerance step["
                    << iter << "]\n";
        status = GSL_SUCCESS;
        break;
      }
    }

  } while (status == GSL_CONTINUE && iter < maxIterations);
  gsl_vector_memcpy(x, minimizer->x);

  if (iter >= maxIterations) {
    if (verbose)
      std::cout << ">>> GSL optimizer: Max iterations reached: "
                << maxIterations << "\n";
  }

  gsl_vector_free(previousX);
  gsl_vector_free(xCopy);
  gsl_multimin_fdfminimizer_free(minimizer);

  if (result) {
    result->iterations = iter;
    result->elapsedTimeMicro =
        (size_t)(std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count());
  }
  return status;
}

int gsl_minimizer::checkXtol(const gsl_multimin_fdfminimizer* minimizer,
                             gsl_vector* prevX, const double xtolAbs,
                             const double xtolRel, GslminimizerResult* result) {
  if (prevX == nullptr) return GSL_CONTINUE;

  const double prevXNorm = gsl_blas_dnrm2(prevX);
  // use prevX as intermideate storage
  gsl_vector_sub(prevX, minimizer->x);
  const double xDiffNormAbs = gsl_blas_dnrm2(prevX);
  const double xDiffNormRel = xDiffNormAbs / prevXNorm;
  gsl_vector_memcpy(prevX, minimizer->x);
  if (result) {
    result->lastXtolAbs = xDiffNormAbs;
    result->lastXtolRel = xDiffNormRel;
  }
#ifdef SHOW_DEBUG
  printf("deltaX abs: %.6e\n", xDiffNormAbs);
  printf("deltaX rel: %.6e\n", xDiffNormAbs / prevXNorm);
#endif
  if (xDiffNormAbs <= xtolAbs || xDiffNormRel <= xtolRel) {
    return GSL_SUCCESS;
  }
  return GSL_CONTINUE;
}

int gsl_minimizer::checkFtol(const gsl_multimin_fdfminimizer* minimizer,
                             double* prevF, const double ftolAbs,
                             const double ftolRel, GslminimizerResult* result) {
  if (prevF == nullptr) return GSL_CONTINUE;

  const double curF = gsl_multimin_fdfminimizer_minimum(minimizer);
  const double deltaF = std::abs(curF - *prevF);
  const double deltaFRel = deltaF / std::abs(*prevF);
  *prevF = curF;
  if (result) {
    result->lastFtolAbs = deltaF;
    result->lastFtolRel = deltaFRel;
  }
#ifdef SHOW_DEBUG
  printf("deltaF abs: %.6e\n", deltaF);
  printf("deltaF rel: %.6e\n", deltaF / std::abs(*prevF));
#endif
  if (deltaF <= ftolAbs || deltaFRel <= ftolRel) {
    return GSL_SUCCESS;
  }
  return GSL_CONTINUE;
}

inline int gsl_minimizer::checkGtol(const gsl_multimin_fdfminimizer* minimizer,
                                    const double gtol,
                                    GslminimizerResult* result) {
  const double gNorm = gsl_blas_dnrm2(minimizer->gradient);
  if (result) {
    result->lastGtol = gNorm;
  }
#ifdef SHOW_DEBUG
  printf("gradient norm: %.6e\n", gNorm);
#endif
  if (gNorm <= gtol) {
    return GSL_SUCCESS;
  }
  return GSL_CONTINUE;
}
