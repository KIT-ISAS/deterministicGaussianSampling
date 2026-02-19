#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gtest/gtest.h>

#include "gm_to_dirac_optimization_params.h"
#include "gm_to_dirac_short.h"
#include "gm_to_dirac_test_case_params.h"
#include "gradient_van_mises_distance.h"
#include "gsl_utils_allocation.h"
#include "gsl_utils_compare.h"
#include "gtest_compare_vec.h"

class gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative
    : public ::testing::TestWithParam<GmToDiracTestCaseParams> {
 protected:
  static void gsl_numerical_gradient(
      const gsl_vector* x, GMToDiracConstWeightOptimizationParams* params,
      gsl_vector* grad, double (*f)(const gsl_vector*, void*)) {
    gradVanMisesDistance.multivariativeGradient(x, grad, f, params);
  }

  void SetUp() override {
    GmToDiracTestCaseParams p = GetParam();
    covDiag = create_gsl_vector(p.covDiag);
    wX = create_gsl_vector(p.wX);
    x = create_gsl_vector(p.x);
    numericalGrad = gsl_vector_alloc(p.L * p.N);
    analyticalGrad = gsl_vector_alloc(p.L * p.N);
    if (wX == nullptr || x == nullptr || covDiag == nullptr ||
        numericalGrad == nullptr || analyticalGrad == nullptr) {
      GTEST_SKIP() << "Failed to allocate memory for vectors.";
    }
  }

  void TearDown() override {
    gsl_vector_free(covDiag);
    gsl_vector_free(wX);
    gsl_vector_free(x);
    gsl_vector_free(numericalGrad);
    gsl_vector_free(analyticalGrad);
  }

  gsl_vector* covDiag;
  gsl_vector* wX;
  gsl_vector* x;
  gsl_vector* numericalGrad;
  gsl_vector* analyticalGrad;

  const double eps = 1e-5;
  const size_t reps = 5;

 private:
  static gradient_van_mises_distance gradVanMisesDistance;
};

gradient_van_mises_distance
    gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative::
        gradVanMisesDistance;

TEST_P(gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
       parameterized_test_modified_van_mises_distance_sq_derivative) {
  GmToDiracTestCaseParams p = GetParam();

  const double c_b = gm_to_dirac_short<double>::c_b(p.bMax);
  GMToDiracConstWeightOptimizationParams params(covDiag, wX, p.N, p.L, p.bMax,
                                                c_b);
  for (size_t i = 0; i < reps; i++) {
    gsl_numerical_gradient(
        x, &params, numericalGrad,
        gm_to_dirac_short<double>::modified_van_mises_distance_sq);

    gm_to_dirac_short<double>::modified_van_mises_distance_sq_derivative(
        x, &params, analyticalGrad);
    ASSERT_TRUE(assert_gsl_vectors_close(analyticalGrad, numericalGrad, eps));
  }
}

TEST_P(
    gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
    parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_distance) {
  GmToDiracTestCaseParams p = GetParam();

  const double c_b = gm_to_dirac_short<double>::c_b(p.bMax);
  GMToDiracConstWeightOptimizationParams params(covDiag, wX, p.N, p.L, p.bMax,
                                                c_b);

  // wrapper
  double distance_wrapper = 0;
  auto gm2d = gm_to_dirac_short<double>();
  gm2d.modified_van_mises_distance_sq(covDiag, &distance_wrapper, p.L, p.N,
                                      p.bMax, x, wX);
  // internal impl
  double distance_internal = 1;
  distance_internal =
      gm_to_dirac_short<double>::modified_van_mises_distance_sq(x, &params);

  ASSERT_TRUE(distance_wrapper == distance_internal);
}

TEST_P(
    gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
    parameterized_test_modified_van_mises_distance_sq_derivative_wrapper_gradient) {
  GmToDiracTestCaseParams p = GetParam();

  const double c_b = gm_to_dirac_short<double>::c_b(p.bMax);
  GMToDiracConstWeightOptimizationParams params(covDiag, wX, p.N, p.L, p.bMax,
                                                c_b);

  // wrapper
  auto gm2d = gm_to_dirac_short<double>();
  gm2d.modified_van_mises_distance_sq_derivative(covDiag, numericalGrad, p.L, p.N,
                                      p.bMax, x, wX);

  // internal impl
  gm_to_dirac_short<double>::modified_van_mises_distance_sq_derivative(
      x, &params, analyticalGrad);
  ASSERT_TRUE(assert_gsl_vectors_close(analyticalGrad, numericalGrad, eps));
}

INSTANTIATE_TEST_SUITE_P(
    ModifiedVanMisesDistanceDerivativeParameterizedTest,
    gm_to_dirac_approx_short_test_modified_van_mises_distance_sq_derivative,
    ::testing::Values(
        GmToDiracTestCaseParams{
            {1.0, 1.0},            // covDiag (2x1)
            {0.5, 0.5},            // wX
            {1.0, 1.0, 3.0, 3.0},  // x (2x2)
            2,                     // L
            2,                     // N
            100                    // bMax
        },
        GmToDiracTestCaseParams{
            {1.0},       // covDiag (1x1)
            {1.0},       // wX
            {0.0, 0.0},  // x (1x2)
            1,           // L
            2,           // N
            100          // bMax
        },
        GmToDiracTestCaseParams{
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0},                                          // covDiag (10x1)
            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},  // wX
            {2.4, 1.7, 9.10, 9.10, 4.9,  6.1, 4.3, 4.4, 10.2, 7.9,
             2.1, 7.2, 9.5,  2.9,  8.6,  6.1, 3.5, 8.3, 5.2,  2.2,
             4.9, 4.9, 1.2,  2.9,  10.9, 3.7, 1.8, 5.4, 2.6,  6.10,
             7.1, 9.3, 9.7,  2.7,  4.7,  7.5, 7.8, 2.6, 7.6,  10.0},  // x (2x2)
            10,                                                       // L
            4,                                                        // N
            100                                                       // bMax
        }));