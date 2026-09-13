#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gm_to_dirac_even_closed_form.h"

int main(int argc, char** argv) {
  if (std::getenv("GSL_RNG_SEED") == nullptr) {
#ifdef _WIN32
    _putenv_s("GSL_RNG_SEED", "42");
#else
    setenv("GSL_RNG_SEED", "42", 1);
#endif
  }

  if (argc < 5 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " <N> <L> <out_csv> <sigma> [bMax]\n"
              << "  N      dimension, must be even and >= 2\n"
              << "  L      number of Dirac components\n"
              << "  sigma  isotropic standard deviation (1 = standard normal)\n"
              << "  bMax   integration bound, any value > 0 (default 100)\n";
    return 1;
  }

  const size_t N = static_cast<size_t>(std::stoi(argv[1]));
  const size_t L = static_cast<size_t>(std::stoi(argv[2]));
  const char* outPath = argv[3];
  const double sigma = std::strtod(argv[4], nullptr);
  const double bMax = (argc == 6) ? std::strtod(argv[5], nullptr) : 100.00;

  if (N == 0 || N % 2 != 0) {
    std::cerr << "error: N must be even and > 0 (got " << N
              << "); the closed form has no odd-N case\n";
    return 1;
  }
  if (L == 0 || !(sigma > 0.00) || !(bMax > 0.00)) {
    std::cerr << "error: L, sigma and bMax must all be > 0\n";
    return 1;
  }

  std::vector<double> x(L * N, 0.00);
  GslminimizerResult result;
  gm_to_dirac_even_closed_form<double> approx;

  // sigma == 1 reduces to the plain standard-normal solve
  const bool ok = approx.approximate_isotropic(L, N, sigma, bMax, x.data(),
                                               nullptr, &result, {});
  if (!ok) {
    std::cerr << "error: LCD approximation did not converge for N=" << N
              << ", L=" << L << ", sigma=" << sigma << ", bMax=" << bMax
              << "\n";
    return 1;
  }

  std::ofstream out(outPath);
  if (!out) {
    std::cerr << "error: cannot open '" << outPath << "' for writing\n";
    return 1;
  }
  out << std::setprecision(17);
  for (size_t i = 0; i < L; ++i) {
    for (size_t k = 0; k < N; ++k) {
      if (k != 0) out << ',';
      out << x[i * N + k];
    }
    out << '\n';
  }

  std::cerr << "converged in " << result.iterations << " iterations ("
            << result.elapsedTimeMicro << " us)\n";
  return 0;
}
