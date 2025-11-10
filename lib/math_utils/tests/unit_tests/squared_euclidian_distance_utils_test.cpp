#include <gsl/gsl_matrix.h>
#include <gtest/gtest.h>

#include "squared_euclidean_distance_utils.h"

size_t linSpacingi(size_t max, size_t min, size_t i, size_t numTests) {
  return ((max - min) / numTests) * i + min;
}

::testing::AssertionResult ASSERT_SquaredEuclidianDistanceUtils(
    SquaredEuclideanDistanceUtils* ref, SquaredEuclideanDistanceUtils* test) {
  if (ref->getL() != test->getL()) {
    return ::testing::AssertionFailure()
           << "Size mismatch L: ref: " << ref->getL()
           << ", test: " << test->getL();
  }
  if (ref->getM() != test->getM()) {
    return ::testing::AssertionFailure()
           << "Size mismatch M: ref: " << ref->getM()
           << ", test: " << test->getM();
  }
  if (ref->getN() != test->getN()) {
    return ::testing::AssertionFailure()
           << "Size mismatch N: ref: " << ref->getN()
           << ", test: " << test->getN();
  }

  for (size_t i = 0; i < ref->getL(); i++) {
    for (size_t j = 0; j < ref->getL(); j++) {
      const double refDist = ref->getDistance(i, j);
      const double testDist = test->getDistance(i, j);
      if (std::abs(refDist - testDist) > 1e-10) {
        return ::testing::AssertionFailure()
               << "Distance mismatch at (" << i << ", " << j << "): "
               << "ref: " << refDist << ", test: " << testDist;
      }
    }
  }

  return ::testing::AssertionSuccess();
}

TEST(SquaredEuclidianDistanceUtilsTest, TestDistance) {
  const size_t minL = 100;
  const size_t maxL = 1000;
  const size_t minN = 1;
  const size_t maxN = 25;
  const size_t minM = 1000;
  const size_t maxM = 10000;
  const size_t numTests = 10;

  gsl_matrix* x = gsl_matrix_alloc(maxL, maxN);
  gsl_matrix* y = gsl_matrix_alloc(maxM, maxN);
  if (!x || !y) {
    GTEST_SKIP() << "Could not allocate memory for matrices";
  }

  for (size_t i = 0; i < maxL; ++i) {
    for (size_t j = 0; j < maxN; ++j) {
      gsl_matrix_set(x, i, j, static_cast<double>(i * maxN + j));
    }
  }
  for (size_t i = 0; i < maxM; ++i) {
    for (size_t j = 0; j < maxN; ++j) {
      gsl_matrix_set(y, i, j, static_cast<double>(i * maxN + j + maxM * maxN));
    }
  }

  for (size_t i = 0; i < numTests; ++i) {
    const size_t curL = linSpacingi(maxL, minL, i, numTests);
    const size_t curN = linSpacingi(maxN, minN, i, numTests);
    const size_t curM = linSpacingi(maxM, minM, i, numTests);

    gsl_matrix_view xView = gsl_matrix_submatrix(x, 0, 0, curL, curN);
    gsl_matrix_view yView = gsl_matrix_submatrix(y, 0, 0, curM, curN);

    SquaredEuclideanDistance_LL_matrix llm(curL, curN);
    llm.calculateDistance(&xView.matrix, &yView.matrix);

    {  // LL_matrix_optimized
      SquaredEuclideanDistance_LL_matrix_optimized llmo(curL, curN);
      llmo.calculateDistance(&xView.matrix, &yView.matrix);
      EXPECT_TRUE(ASSERT_SquaredEuclidianDistanceUtils(&llm, &llmo));
    }
    {  // LL_vectorized
      SquaredEuclideanDistance_LL_vectorized llv(curL, curN);
      llv.calculateDistance(&xView.matrix, &yView.matrix);
      EXPECT_TRUE(ASSERT_SquaredEuclidianDistanceUtils(&llm, &llv));
    }
    {  // LL_vectorized_optimized
      SquaredEuclideanDistance_LL_vectorized_optimized llvo(curL, curN);
      llvo.calculateDistance(&xView.matrix, &yView.matrix);
      EXPECT_TRUE(ASSERT_SquaredEuclidianDistanceUtils(&llm, &llvo));
    }

    SquaredEuclideanDistance_LM_matrix llm2(curL, curM, curN);
    llm2.calculateDistance(&xView.matrix, &yView.matrix);

    {  // LM_matrix_optimized
      SquaredEuclideanDistance_LM_matrix_optimized llmo2(curL, curM, curN);
      llmo2.calculateDistance(&xView.matrix, &yView.matrix);
      EXPECT_TRUE(ASSERT_SquaredEuclidianDistanceUtils(&llm2, &llmo2));
    }
  }

  if (x) gsl_matrix_free(x);
  if (y) gsl_matrix_free(y);
}