#ifndef APPROXIMATE_OPTIONS_H
#define APPROXIMATE_OPTIONS_H

#include <cstddef>

struct ApproximateOptions {
  double xtolAbs = 0;            // Absolute tolerance for x
  double xtolRel = 0;            // Relative tolerance for x
  double ftolAbs = 0;            // Absolute tolerance for f
  double ftolRel = 1e-10;        // Relative tolerance for f
  double gtol = 1e-6;            // Tolerance for gradient-norm
  bool initialX = false;         // True if x should be used as initial guess
  size_t maxIterations = 10000;  // Maximum number of iterations
  bool verbose = false;          // True if verbose output is needed
};

#endif