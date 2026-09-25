#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gm_to_dirac_even_binf.h"

int main(int argc, char** argv) {
  if (std::getenv("GSL_RNG_SEED") == nullptr) {
#ifdef _WIN32
    _putenv_s("GSL_RNG_SEED", "42");
#else
    setenv("GSL_RNG_SEED", "42", 1);
#endif
  }

  if (argc != 4) {
    std::cerr
        << "usage: " << argv[0] << " <N> <L> <out_csv>\n"
        << "  N  dimension, must be even and >= 2\n"
        << "  L  number of Dirac components, must be >= 2\n"
        << "\n";
    return 1;
  }

  const size_t N = static_cast<size_t>(std::stoi(argv[1]));
  const size_t L = static_cast<size_t>(std::stoi(argv[2]));
  const char* outPath = argv[3];

  if (N == 0 || N % 2 != 0) {
    std::cerr << "error: N must be even and > 0 (got " << N
              << "); the closed form has no odd-N case\n";
    return 1;
  }
  if (L < 2) {
    std::cerr << "error: L must be >= 2; L = 1 has no non-degenerate "
                 "zero-mean sample set\n";
    return 1;
  }

  std::vector<double> x(L * N, 0.00);
  GslminimizerResult result;
  gm_to_dirac_even_binf<double> approx;

  if (!approx.approximate(L, N, x.data(), nullptr, &result, {})) {
    std::cerr << "error: LCD approximation did not converge for N=" << N
              << ", L=" << L << "\n";
    return 1;
  }

  double distance = 0.00;
  approx.modified_van_mises_distance_sq(&distance, L, N, x.data(), nullptr);

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

  std::cerr << "distance = " << std::setprecision(17) << distance << '\n';
  std::cerr << "converged in " << result.iterations << " iterations ("
            << result.elapsedTimeMicro << " us)\n";
  return 0;
}
