#ifndef DIRAC_TO_DIRAC_TEST_CASE_PARAMS_H
#define DIRAC_TO_DIRAC_TEST_CASE_PARAMS_H

#include <vector>

struct DiracToDiracTestCaseParams {
  std::vector<double> wX;
  std::vector<double> x;
  size_t L;
  std::vector<double> wY;
  std::vector<double> y;
  size_t M;
  size_t N;
  size_t bMax;
};

struct DiracToDiracTestCaseExpectedParams {
  std::vector<double> wX;
  std::vector<double> x;
  size_t L;
  std::vector<double> wY;
  std::vector<double> y;
  size_t M;
  size_t N;
  size_t bMax;
  double expected;
};

#endif  // DIRAC_TO_DIRAC_TEST_CASE_PARAMS_H