#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <stddef.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "capture_time.h"
#include "dirac_to_dirac_approx_short.h"
#include "dirac_to_dirac_approx_short_function.h"
#include "dirac_to_dirac_approx_short_thread.h"
#include "dirac_to_dirac_optimization_params.h"
#include "gm_to_dirac_optimization_params.h"
#include "gm_to_dirac_short.h"
#include "gradient_van_mises_distance_sq_const_weight.h"
#include "gsl_quadrature_adaptive_gauss_kronrod.h"
#include "squared_euclidean_distance_utils.h"

void save2DMatrix(double* mat, size_t rows, size_t cols,
                  const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return;
  }
  if (cols != 2) {
    std::cerr << "Error: Only 2D matrices are supported." << std::endl;
    file.close();
    return;
  }
  char header[2] = {'x', 'y'};
  for (size_t c = 0; c < cols; ++c) {
    file << header[c];
    if (c < cols - 1) file << ",";
  }
  file << "\n";
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      file << mat[r * cols + c];
      if (c < cols - 1) file << ",";
    }
    file << "\n";
  }
  file.close();
}

void generateEllipsePoints(double* points, size_t numPoints, double a,
                           double b) {
  std::srand(static_cast<unsigned int>(
      std::time(nullptr)));  // Seed random number generator

  for (size_t i = 0; i < numPoints; i++) {
    double theta = (std::rand() / (double)RAND_MAX) * 2 * M_PI;  // Random angle
    double r =
        sqrt(std::rand() / (double)RAND_MAX);  // Random normalized radius

    double x = r * a * cos(theta);
    double y = r * b * sin(theta);

    points[2 * i] = x;
    points[2 * i + 1] = y;
  }
}

static void wXcallback(const double* x, double* res, size_t L, size_t N) {
  for (size_t i = 0; i < L; i++) {
    double sed = 0.00;
    for (size_t k = 0; k < N; k++) {
      const double xik = x[i * N + k];
      sed += xik * xik;
    }
    res[i] = std::exp(0.5 * sed);
  }
}
static void wXDcallback(const double* x, double* grad, size_t L, size_t N) {
  for (size_t xi = 0; xi < L; xi++) {
    double sed = 0.00;
    for (size_t k = 0; k < N; k++) {
      const double xik = x[xi * N + k];
      sed += xik * xik;
    }
    const double expSed = std::exp(0.5 * sed);
    for (size_t eta = 0; eta < N; ++eta) {
      grad[xi * N + eta] = x[xi * N + eta] * expSed;
    }
  }
}

void runRandomApproximation() {
  dirac_to_dirac_approx_short<double> dirac_to_dirac;
  dirac_to_dirac_approx_short_thread<double> dirac_to_dirac_threaded;
  dirac_to_dirac_approx_short_function<double> dirac_to_dirac_weighted;
  gm_to_dirac_short<double> gausian_to_dirac;

  size_t N = 2;
  size_t L = 8;
  size_t M = 1000;

  double* y = new double[M * N];
  double* x_dirac_to_dirac = new double[L * N];
  double* x_dirac_to_dirac_threaded = new double[L * N];
  double* x_dirac_to_dirac_weighted = new double[L * N];
  double* x_gausian_to_dirac = new double[L * N];

  if (!y || !x_dirac_to_dirac || !x_dirac_to_dirac_threaded ||
      !x_dirac_to_dirac_weighted || !x_gausian_to_dirac) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return;
  }

  double* covDiag = new double[N];
  covDiag[0] = 1.0;
  covDiag[1] = 1.0;

  gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(r, static_cast<unsigned long>(time(NULL)));
  for (size_t j = 0; j < M; j++) {
    for (size_t k = 0; k < N; k++) {
      y[j * N + k] = gsl_ran_gaussian(r, covDiag[k]);
    }
  }
  gsl_rng_free(r);

  // dirac_to_dirac
  std::cout << ">> running dirac_to_dirac<double> L:" << L << ", N:" << N
            << ", M:" << M << "\n";
  GslminimizerResult resultDiracToDirac;
  CaptureTime::start("dirac_to_dirac<double>");
  dirac_to_dirac.approximate(y, M, L, N, 100, x_dirac_to_dirac, nullptr,
                             nullptr, &resultDiracToDirac);
  std::cout << resultDiracToDirac;
  CaptureTime::stop("dirac_to_dirac<double>");

  // dirac_to_dirac_threaded
  std::cout << ">> running dirac_to_dirac_threaded<double> L:" << L
            << ", N:" << N << ", M:" << M << "\n";
  GslminimizerResult resultDiracToDiracThreaded;
  CaptureTime::start("dirac_to_dirac_threaded<double>");
  dirac_to_dirac_threaded.approximate(y, M, L, N, 100,
                                      x_dirac_to_dirac_threaded, nullptr,
                                      nullptr, &resultDiracToDiracThreaded);
  std::cout << resultDiracToDiracThreaded;
  CaptureTime::stop("dirac_to_dirac_threaded<double>");

  // dirac_to_dirac_weighted
  std::cout << ">> running dirac_to_dirac_weighted<double> L:" << L
            << ", N:" << N << ", M:" << M << "\n";
  GslminimizerResult resultDiracToDiracWeighted;
  CaptureTime::start("dirac_to_dirac_weighted<double>");
  dirac_to_dirac_weighted.approximate(y, M, L, N, 100,
                                      x_dirac_to_dirac_weighted, wXcallback,
                                      wXDcallback, &resultDiracToDiracWeighted);
  std::cout << resultDiracToDiracWeighted;
  CaptureTime::stop("dirac_to_dirac_weighted<double>");

  // gausian_to_dirac
  std::cout << ">> running gausian_to_dirac<double> L:" << L << ", N:" << N
            << "\n";
  GslminimizerResult resultGausianToDirac;
  CaptureTime::start("gausian_to_dirac<double>");
  gausian_to_dirac.approximate(covDiag, L, N, 100, x_gausian_to_dirac, nullptr,
                               &resultGausianToDirac);
  std::cout << resultGausianToDirac;
  CaptureTime::stop("gausian_to_dirac<double>");

  if (N == 2) {
    std::cout << ">> saving results to csv files\n";
    save2DMatrix(y, M, N, "y.csv");
    save2DMatrix(x_dirac_to_dirac, L, N, "x_dirac_to_dirac.csv");
    save2DMatrix(x_dirac_to_dirac_threaded, L, N,
                 "x_dirac_to_dirac_threaded.csv");
    save2DMatrix(x_dirac_to_dirac_weighted, L, N,
                 "x_dirac_to_dirac_weighted.csv");
    save2DMatrix(x_gausian_to_dirac, L, N, "x_gausian_to_dirac.csv");
  }
  if (y) delete[] y;
  if (x_dirac_to_dirac) delete[] x_dirac_to_dirac;
  if (x_dirac_to_dirac_threaded) delete[] x_dirac_to_dirac_threaded;
  if (x_dirac_to_dirac_weighted) delete[] x_dirac_to_dirac_weighted;
  if (x_gausian_to_dirac) delete[] x_gausian_to_dirac;
}

void euclideanDistanceTest() {
  const size_t L = 1000;
  const size_t M = 10000;
  const size_t N = 25;
  const gsl_matrix* x = gsl_matrix_calloc(L, N);
  const gsl_matrix* y = gsl_matrix_calloc(M, N);

  {
    std::cout << ">> SquaredEuclideanDistance_LL_matrix L: " << L
              << ", N: " << N << "\n";
    auto llm = SquaredEuclideanDistance_LL_matrix(L, N);
    CaptureTime::start("SquaredEuclideanDistance_LL_matrix");
    llm.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LL_matrix");
  }

  {
    std::cout << ">> SquaredEuclideanDistance_LL_matrix_optimized L: " << L
              << ", N: " << N << "\n";
    auto llmo = SquaredEuclideanDistance_LL_matrix_optimized(L, N);
    CaptureTime::start("SquaredEuclideanDistance_LL_matrix_optimized");
    llmo.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LL_matrix_optimized");
  }

  {
    std::cout << ">> SquaredEuclideanDistance_LL_vectorized L: " << L
              << ", N: " << N << "\n";
    auto llv = SquaredEuclideanDistance_LL_vectorized(L, N);
    CaptureTime::start("SquaredEuclideanDistance_LL_vectorized");
    llv.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LL_vectorized");
  }

  {
    std::cout << ">> SquaredEuclideanDistance_LL_vectorized_optimized L: " << L
              << ", N: " << N << "\n";
    auto llvo = SquaredEuclideanDistance_LL_vectorized_optimized(L, N);
    CaptureTime::start("SquaredEuclideanDistance_LL_vectorized_optimized");
    llvo.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LL_vectorized_optimized");
  }

  {
    std::cout << ">> SquaredEuclideanDistance_LM_matrix L: " << L
              << ", M: " << M << ", N: " << N << "\n";
    auto lmm = SquaredEuclideanDistance_LM_matrix(L, M, N);
    CaptureTime::start("SquaredEuclideanDistance_LM_matrix");
    lmm.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LM_matrix");
  }

  {
    std::cout << ">> SquaredEuclideanDistance_LM_matrix_optimized L: " << L
              << ", M: " << M << ", N: " << N << "\n";
    auto lmmo = SquaredEuclideanDistance_LM_matrix_optimized(L, M, N);
    CaptureTime::start("SquaredEuclideanDistance_LM_matrix_optimized");
    lmmo.calculateDistance(x, y);
    CaptureTime::stop("SquaredEuclideanDistance_LM_matrix_optimized");
  }

  if (x) gsl_matrix_free(const_cast<gsl_matrix*>(x));
  if (y) gsl_matrix_free(const_cast<gsl_matrix*>(y));
}

int main() {
  std::cout << std::fixed << std::setprecision(15);

  runRandomApproximation();
  euclideanDistanceTest();
  CaptureTime::printAllStatistics();

  return 0;
}
