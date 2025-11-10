#ifndef GM_TO_DIRAC_TEST_CASE_PARAMS_H
#define GM_TO_DIRAC_TEST_CASE_PARAMS_H

#include <vector>

struct GmToDiracTestCaseParams {
  std::vector<double> covDiag;
  std::vector<double> wX;
  std::vector<double> x;
  size_t L;
  size_t N;
  size_t bMax;
};

#endif  // GM_TO_DIRAC_TEST_CASE_PARAMS_H