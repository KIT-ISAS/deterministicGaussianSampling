#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gtest/gtest.h>

#include "dirac_to_dirac_approx_short.h"
#include "dirac_to_dirac_test_case_params.h"
#include "gradient_van_mises_distance_sq_const_weight.h"
#include "gsl_utils_allocation.h"
#include "gtest_compare_vec.h"

class dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative
    : public ::testing::TestWithParam<DiracToDiracTestCaseParams> {
 protected:
  static void gsl_numerical_gradient(
      const gsl_vector* x, DiracToDiracConstWeightOptimizationParams* params,
      gsl_vector* grad, double (*f)(const gsl_vector*, void*)) {
    gradVanMisesDistanceSqConstWeight.multivariativeGradient(x, grad, f,
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

  const double eps = 1e-5;
  const size_t reps = 1;

 private:
  static gradient_van_mises_distance_sq_const_weight
      gradVanMisesDistanceSqConstWeight;
};

TEST_P(
    dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
    parameterized_test_modified_van_mises_distance_sq_derivative) {
  DiracToDiracTestCaseParams p = GetParam();

  const double c_b = dirac_to_dirac_approx_short<double>::c_b(p.bMax);
  DiracToDiracConstWeightOptimizationParams params(y, p.N, p.M, p.L, p.bMax,
                                                   c_b);
  for (size_t i = 0; i < reps; i++) {
    gsl_numerical_gradient(
        x, &params, numericalGrad,
        dirac_to_dirac_approx_short<double>::modified_van_mises_distance_sq);

    dirac_to_dirac_approx_short<
        double>::modified_van_mises_distance_sq_derivative(x, &params,
                                                           analyticalGrad);
    ASSERT_TRUE(assert_gsl_vectors_close(analyticalGrad, numericalGrad, eps));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ModifiedVanMisesDistanceDerivativeParameterizedTest,
    dirac_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
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