#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gtest/gtest.h>

#include "dirac_to_dirac_approx_short_function.h"
#include "dirac_to_dirac_optimization_params.h"
#include "dirac_to_dirac_test_case_params.h"
#include "gradient_van_mises_distance_sq_dynamic_weight.h"
#include "gsl_utils_allocation.h"
#include "gtest_compare_vec.h"

class
    dirac_to_dirac_approx_short_function_test_modified_van_mises_distance_sq_derivative
    : public ::testing::TestWithParam<DiracToDiracTestCaseParams> {
 protected:
  static void wXcallback(const double* x, double* res, size_t L, size_t N) {
    for (size_t i = 0; i < L; ++i) {
      res[i] = 0.00;
      for (size_t k = 0; k < N; k++) {
        const double xik = x[i * N + k];
        res[i] += xik * xik;
      }
      res[i] = std::exp(0.5 * res[i]);
    }
  }
  static void wXDcallback(const double* x, double* grad, size_t L, size_t N) {
    for (size_t i = 0; i < L; ++i) {
      double norm_sq = 0.0;
      for (size_t k = 0; k < N; ++k) {
        double xik = x[i * N + k];
        norm_sq += xik * xik;
      }
      double wi = std::exp(0.5 * norm_sq);

      for (size_t eta = 0; eta < N; ++eta) {
        grad[i * N + eta] = x[i * N + eta] * wi;
      }
    }
  }

  static void gsl_numerical_gradient(
      const gsl_vector* x, DiracToDiracVariableWeightOptimizationParams* params,
      gsl_vector* grad, double (*f)(const gsl_vector*, void*)) {
    gradVanMisesDistanceSqDynamicWeight.multivariativeGradient(x, grad, f,
                                                               params);
  }

  void SetUp() override {
    DiracToDiracTestCaseParams p = GetParam();
    wX = create_gsl_vector(p.wX);
    x = create_gsl_vector(p.x);
    wY = create_gsl_vector(p.wY);
    y = create_gsl_vector(p.y);
    numericalGrad = gsl_vector_alloc(p.L * p.N);
    analyticalGrad = gsl_vector_alloc(p.L * p.N);
    if (wX == nullptr || x == nullptr || wY == nullptr || y == nullptr ||
        numericalGrad == nullptr || analyticalGrad == nullptr) {
      GTEST_SKIP() << "Failed to allocate memory for vectors.";
    }
  }

  void TearDown() override {
    gsl_vector_free(wX);
    gsl_vector_free(x);
    gsl_vector_free(wY);
    gsl_vector_free(y);
    gsl_vector_free(numericalGrad);
    gsl_vector_free(analyticalGrad);
  }

  gsl_vector* wX;
  gsl_vector* x;
  gsl_vector* wY;
  gsl_vector* y;
  gsl_vector* numericalGrad;
  gsl_vector* analyticalGrad;

  const size_t reps = 2;
  const double eps = 1e-5;

 private:
  static gradient_van_mises_distance_sq_dynamic_weight
      gradVanMisesDistanceSqDynamicWeight;
};

static double wXcallbackWrapper(const gsl_vector* x, void* params) {
  DiracToDiracVariableWeightOptimizationParams* p =
      static_cast<DiracToDiracVariableWeightOptimizationParams*>(params);
  gsl_vector* wX = gsl_vector_alloc(p->L);
  p->wXcallback(x->data, wX->data, p->L, p->N);
  double sum = 0.0;
  for (size_t i = 0; i < p->L; ++i) {
    sum += gsl_vector_get(wX, i);
  }
  gsl_vector_free(wX);
  return sum;
}

TEST_P(
    dirac_to_dirac_approx_short_function_test_modified_van_mises_distance_sq_derivative,
    parameterized_test_modified_van_mises_distance_sq_derivative) {
  DiracToDiracTestCaseParams p = GetParam();

  for (size_t i = 0; i < p.L * p.N; i++) {
    x->data[i] = x->data[i] * 0.001;
  }

  gsl_vector* wX = gsl_vector_alloc(p.L);
  gsl_vector* wXderiv = gsl_vector_alloc(p.L * p.N);
  if (wX == nullptr || wXderiv == nullptr) {
    GTEST_SKIP() << "Failed to allocate memory for vectors.";
  }

  DiracToDiracVariableWeightOptimizationParams paramsWxGradientCheck(
      wXcallback, wXDcallback, y, p.N, p.M, p.L, p.bMax, 0.0);

  gsl_numerical_gradient(x, &paramsWxGradientCheck, numericalGrad,
                         wXcallbackWrapper);
  wXDcallback(x->data, wXderiv->data, p.L, p.N);
  ASSERT_TRUE(assert_gsl_vectors_close(wXderiv, numericalGrad));

  gsl_vector_free(wX);
  gsl_vector_free(wXderiv);

  const double c_b = dirac_to_dirac_approx_short_function<double>::c_b(p.bMax);
  for (size_t i = 0; i < reps; i++) {
    DiracToDiracVariableWeightOptimizationParams params(
        wXcallback, wXDcallback, y, p.N, p.M, p.L, p.bMax, c_b);

    gsl_numerical_gradient(x, &params, numericalGrad,
                           dirac_to_dirac_approx_short_function<
                               double>::modified_van_mises_distance_sq);

    dirac_to_dirac_approx_short_function<
        double>::modified_van_mises_distance_sq_derivative(x, &params,
                                                           analyticalGrad);
    ASSERT_TRUE(assert_gsl_vectors_close(analyticalGrad, numericalGrad));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ModifiedVanMisesDistanceDerivativeParameterizedTest,
    dirac_to_dirac_approx_short_function_test_modified_van_mises_distance_sq_derivative,
    ::testing::Values(
        DiracToDiracTestCaseParams{
            {0.5, 0.5},            // wX
            {1.0, 1.0, 3.0, 3.0},  // x (2x2)
            2,                     // L
            {0.5, 0.5},            // wY
            {2.5, 2.0, 4.0, 4.0},  // y (2x2)
            2,                     // M
            2,                     // N
            100                    // bMax
        },
        DiracToDiracTestCaseParams{
            {1.0},       // wX
            {1.0, 1.0},  // x (1x2)
            1,           // L
            {1.0},       // wY
            {3.0, 4.0},  // y (1x2)
            1,           // M
            2,           // N
            100          // bMax
        },
        DiracToDiracTestCaseParams{
            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},  // wX
            {2.4, 1.7, 9.10, 9.10, 4.9,  6.1, 4.3, 4.4, 10.2, 7.9,
             2.1, 7.2, 9.5,  2.9,  8.6,  6.1, 3.5, 8.3, 5.2,  2.2,
             4.9, 4.9, 1.2,  2.9,  10.9, 3.7, 1.8, 5.4, 2.6,  6.10,
             7.1, 9.3, 9.7,  2.7,  4.7,  7.5, 7.8, 2.6, 7.6,  10.0},  // x(10x4)
            10,                                                       // L
            {0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05},  // wY
            {5.5, 10.9, 10.5, 3.3,  6.5,  8.10, 3.7,   3.10, 7.8, 3.7,  5.1,
             2.1, 6.4,  8.4,  1.2,  1.6,  5.2,  7.8,   10.7, 5.3, 6.10, 5.9,
             5.1, 3.3,  8.2,  7.10, 10.9, 8.9,  9.10,  3.1,  4.8, 8.2,  4.10,
             8.1, 9.9,  3.5,  3.10, 2.2,  3.2,  10.10, 10.8, 7.9, 3.9,  1.4,
             1.7, 10.4, 3.9,  2.5,  2.10, 7.3,  9.1,   4.4,  5.1, 10.6, 6.5,
             8.1, 5.9,  9.3,  2.2,  1.2,  1.6,  8.10,  1.10, 3.8, 3.8,  10.8,
             2.4, 4.8,  9.1,  1.2,  5.1,  2.1,  1.4,   5.6,  1.1, 7.1,  9.9,
             8.7, 1.7,  10.0},  // y (20x4)
            20,                 // M
            4,                  // N
            100                 // bMax
        }));