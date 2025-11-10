#ifndef SQUARED_EUCLIDIAN_DISTANCE_UTILS_H
#define SQUARED_EUCLIDIAN_DISTANCE_UTILS_H

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <iostream>

// #define USE_OMP

/**
 * @brief helper class to calculate squared euclidean distance between
 *  all linear combinations of two vector sets
 */
class SquaredEuclideanDistanceUtils {
 public:
  SquaredEuclideanDistanceUtils(size_t L, size_t M, size_t N)
      : L(L), M(M), N(N) {}

  virtual ~SquaredEuclideanDistanceUtils() {}

  virtual double getDistance(size_t xi, size_t yi) const = 0;

  size_t getL() const { return L; }
  size_t getM() const { return M; }
  size_t getN() const { return N; }
  /**
   * @brief calculate all squared euclidean distance between vectors contained
   * as columns in x and y
   *
   * @param x LxN Matrix with L vectors of size N
   * @param y MxN Matrix with M vectors of size M
   */
  virtual void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) = 0;

 protected:
  size_t L;
  size_t M;
  size_t N;

 protected:
  /**
   * @brief convert an gsl_vector to a N column gsl_matrix
   *
   * @param v vector to be converted
   * @param N number of columns
   * @return the matrix view using same memory as the vector
   */
  static gsl_matrix convToMatrix(const gsl_vector* v, size_t N) noexcept {
    assert(v->size % N == 0);
    size_t rows = v->size / N;
    return gsl_matrix_view_array(v->data, rows, N).matrix;
  }
};

/**
 * @brief base class for implementations calculating squared euclidean distances
 * with just one set of vectors
 */
class SquaredEuclideanDistanceUtilsLL : public SquaredEuclideanDistanceUtils {
 public:
  SquaredEuclideanDistanceUtilsLL(size_t L, size_t N)
      : SquaredEuclideanDistanceUtils(L, L, N) {}

  ~SquaredEuclideanDistanceUtilsLL() override {}
};

/**
 * @brief base class for implementations calculating squared euclidean distances
 * with two set of vectors
 */
class SquaredEuclideanDistanceUtilsLM : public SquaredEuclideanDistanceUtils {
 public:
  SquaredEuclideanDistanceUtilsLM(size_t L, size_t M, size_t N)
      : SquaredEuclideanDistanceUtils(L, M, N) {}

  ~SquaredEuclideanDistanceUtilsLM() override {}
};

/**
 * @brief implements the naive calculation of all squared euclidean distances
 * store in a LxL matrix
 */
class SquaredEuclideanDistance_LL_matrix
    : public SquaredEuclideanDistanceUtilsLL {
 public:
  SquaredEuclideanDistance_LL_matrix(size_t L, size_t N)
      : SquaredEuclideanDistanceUtilsLL(L, N) {
    distanceMatrix = gsl_matrix_alloc(L, L);
  }

  ~SquaredEuclideanDistance_LL_matrix() {
    if (distanceMatrix) {
      gsl_matrix_free(distanceMatrix);
    }
  }

  inline double getDistance(size_t xi, size_t xj) const override {
    assert(xi < L && xj < L);
    return gsl_matrix_get(distanceMatrix, xi, xj);
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) override {
    (void)y;
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = 0; j < L; ++j) {
        if (i == j) {
          gsl_matrix_set(distanceMatrix, i, j, 0.00);
          continue;
        }
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
          const double diff = gsl_matrix_get(x, i, k) - gsl_matrix_get(x, j, k);
          sum += diff * diff;
        }
        gsl_matrix_set(distanceMatrix, i, j, sum);
      }
    }
  }

 private:
  gsl_matrix* distanceMatrix = nullptr;
};

/**
 * @brief implements a optimized version for calculating squared euclidean
 * distances store in a LxL matrix
 *
 * uses ||xi - xj||² = ||xi||² + ||xj||² - 2 * xiᵀ * xj
 *
 * with cblas functions
 */
class SquaredEuclideanDistance_LL_matrix_optimized
    : public SquaredEuclideanDistanceUtilsLL {
 public:
  SquaredEuclideanDistance_LL_matrix_optimized(size_t L, size_t N)
      : SquaredEuclideanDistanceUtilsLL(L, N) {
    distanceMatrix = gsl_matrix_alloc(L, L);
    means = gsl_vector_alloc(L);
    xxt = gsl_matrix_alloc(L, L);
  }

  ~SquaredEuclideanDistance_LL_matrix_optimized() {
    if (distanceMatrix) gsl_matrix_free(distanceMatrix);
    if (means) gsl_vector_free(means);
    if (xxt) gsl_matrix_free(xxt);
  }

  inline double getDistance(size_t xi, size_t xj) const override {
    assert(xi < L && xj < L);
    return gsl_matrix_get(distanceMatrix, xi, xj);
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix*) override {
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < L; i++) {
      gsl_vector_const_view xi = gsl_matrix_const_row(x, i);
      double xi_sq_norm = 0.0;
      gsl_blas_ddot(&(xi.vector), &(xi.vector), &xi_sq_norm);
      gsl_vector_set(means, i, xi_sq_norm);
    }

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, x, x, 0.0, xxt);

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = 0; j < L; ++j) {
        const double dist = gsl_vector_get(means, i) +
                            gsl_vector_get(means, j) -
                            2 * gsl_matrix_get(xxt, i, j);
        gsl_matrix_set(distanceMatrix, i, j, dist);
      }
    }
  }

 private:
  gsl_matrix* distanceMatrix = nullptr;
  gsl_vector* means = nullptr;
  gsl_matrix* xxt = nullptr;
};

/**
 * @brief implements a vectorized version of calculating squared euclidean
 * distances store in a vector
 *
 * uses symmetrie: ||xi - xj||² = ||xj - xi||²
 *
 * only needs to calculate (L² - L) / 2 elements
 */
class SquaredEuclideanDistance_LL_vectorized
    : public SquaredEuclideanDistanceUtilsLL {
 public:
  SquaredEuclideanDistance_LL_vectorized(size_t L, size_t N)
      : SquaredEuclideanDistanceUtilsLL(L, N) {
    distanceVector = gsl_vector_alloc(((L * L) - L) / 2);
  }

  ~SquaredEuclideanDistance_LL_vectorized() {
    if (distanceVector) gsl_vector_free(distanceVector);
  }

  inline double getDistance(size_t xi, size_t xj) const override {
    assert(xi < L && xj < L);
    if (xi == xj) return 0.00;
    if (xi < xj) std::swap(xi, xj);
    return gsl_vector_get(distanceVector, distIndex(xi, xj));
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) override {
    (void)y;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 1; i < L; i++) {
      for (size_t j = 0; j < i; j++) {
        const size_t distanceIndex = distIndex(i, j);
        double sum = 0.0;
        for (size_t k = 0; k < N; k++) {
          const double diff = gsl_matrix_get(x, i, k) - gsl_matrix_get(x, j, k);
          sum += diff * diff;
        }
        gsl_vector_set(distanceVector, distanceIndex, sum);
      }
    }
  }

 private:
  inline size_t distIndex(size_t i, size_t j) const {
    return ((i - 1) * i) / 2 + j;
  }
  inline size_t diffIndex(size_t i, size_t j, size_t k) const {
    return distIndex(i, j) * N + k;
  }
  inline size_t diffIndex(size_t distIndex, size_t k) const {
    return distIndex * N + k;
  }

  gsl_vector* distanceVector = nullptr;
};

/**
 * @brief implements a vectorized version of calculating squared euclidean
 * distances store in a vector
 *
 * uses symmetrie: ||xi - xj||² = ||xj - xi||²
 * and ||xi - xj||² = ||xi||² + ||xj||² - 2 * xiᵀ * xj
 *
 * with cblas functions
 *
 * only needs to calculate (L² - L) / 2 elements
 */
class SquaredEuclideanDistance_LL_vectorized_optimized
    : public SquaredEuclideanDistanceUtilsLL {
 public:
  SquaredEuclideanDistance_LL_vectorized_optimized(size_t L, size_t N)
      : SquaredEuclideanDistanceUtilsLL(L, N) {
    distanceVector = gsl_vector_alloc(((L * L) - L) / 2);
    means = gsl_vector_alloc(L);
  }

  ~SquaredEuclideanDistance_LL_vectorized_optimized() {
    if (distanceVector) gsl_vector_free(distanceVector);
  };

  double getDistance(size_t xi, size_t xj) const override {
    assert(xi < L && xj < L);
    if (xi == xj) return 0.00;
    if (xi < xj) std::swap(xi, xj);
    return gsl_vector_get(distanceVector, ((xi - 1) * xi) / 2 + xj);
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) override {
    (void)y;
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < L; i++) {
      gsl_vector_const_view xi = gsl_matrix_const_row(x, i);
      double xi_sq_norm = 0.0;
      gsl_blas_ddot(&(xi.vector), &(xi.vector), &xi_sq_norm);
      gsl_vector_set(means, i, xi_sq_norm);
    }

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 1; i < L; i++) {
      for (size_t j = 0; j < i; j++) {
        gsl_vector_const_view xi = gsl_matrix_const_row(x, i);
        gsl_vector_const_view xj = gsl_matrix_const_row(x, j);

        double xi_sq_norm = gsl_vector_get(means, i) + gsl_vector_get(means, j);

        double xi_xj_dot = 0.0;
        gsl_blas_ddot(&(xi.vector), &(xj.vector), &xi_xj_dot);

        double dist = xi_sq_norm - 2.0 * xi_xj_dot;
        if (dist < 0.0) dist = 0.0;

        gsl_vector_set(distanceVector, ((i - 1) * i) / 2 + j, dist);
      }
    }
  }

 private:
  gsl_vector* distanceVector = nullptr;
  gsl_vector* means = nullptr;
};

/**
 * @brief implements the naive calculation of all squared euclidean distances
 * store in a LxM matrix
 */
class SquaredEuclideanDistance_LM_matrix
    : public SquaredEuclideanDistanceUtilsLM {
 public:
  SquaredEuclideanDistance_LM_matrix(size_t L, size_t M, size_t N)

      : SquaredEuclideanDistanceUtilsLM(L, M, N) {
    distanceMatrix = gsl_matrix_alloc(L, M);
  }

  ~SquaredEuclideanDistance_LM_matrix() {
    if (distanceMatrix) gsl_matrix_free(distanceMatrix);
  }

  double getDistance(size_t xi, size_t yi) const override {
    assert(xi < L && yi < M);
    return gsl_matrix_get(distanceMatrix, xi, yi);
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) override {
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < L; i++) {
      for (size_t j = 0; j < M; j++) {
        double sum = 0.0;
        for (size_t k = 0; k < N; k++) {
          const double diff = gsl_matrix_get(x, i, k) - gsl_matrix_get(y, j, k);
          sum += diff * diff;
        }
        gsl_matrix_set(distanceMatrix, i, j, sum);
      }
    }
  }

 private:
  gsl_matrix* distanceMatrix = nullptr;
};

/**
 * @brief implements a optimized version for calculating squared euclidean
 * distances store in a LxM matrix
 *
 * uses ||xi - yj||² = ||xi||² + ||yj||² - 2 * xiᵀ * yj
 *
 * with cblas functions
 */
class SquaredEuclideanDistance_LM_matrix_optimized
    : public SquaredEuclideanDistanceUtilsLM {
 public:
  SquaredEuclideanDistance_LM_matrix_optimized(size_t L, size_t M, size_t N)
      : SquaredEuclideanDistanceUtilsLM(L, M, N) {
    distanceMatrix = gsl_matrix_alloc(L, M);
    meansX = gsl_vector_alloc(L);
    meansY = gsl_vector_alloc(M);
    xyt = gsl_matrix_alloc(L, M);
  }

  ~SquaredEuclideanDistance_LM_matrix_optimized() {
    if (distanceMatrix) gsl_matrix_free(distanceMatrix);
    if (meansX) gsl_vector_free(meansX);
    if (meansY) gsl_vector_free(meansY);
    if (xyt) gsl_matrix_free(xyt);
  }

  double getDistance(size_t xi, size_t yi) const override {
    assert(xi < L && yi < M);
    return gsl_matrix_get(distanceMatrix, xi, yi);
  }

  void calculateDistance(const gsl_matrix* x, const gsl_matrix* y) override {
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < L; i++) {
      gsl_vector_const_view xi = gsl_matrix_const_row(x, i);
      double xi_sq_norm = 0.0;
      gsl_blas_ddot(&(xi.vector), &(xi.vector), &xi_sq_norm);
      gsl_vector_set(meansX, i, xi_sq_norm);
    }

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < M; j++) {
      gsl_vector_const_view yj = gsl_matrix_const_row(y, j);
      double yj_sq_norm = 0.0;
      gsl_blas_ddot(&(yj.vector), &(yj.vector), &yj_sq_norm);
      gsl_vector_set(meansY, j, yj_sq_norm);
    }

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, x, y, 0.0, xyt);

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < L; ++i) {
      for (size_t j = 0; j < M; ++j) {
        const double dist = gsl_vector_get(meansX, i) +
                            gsl_vector_get(meansY, j) -
                            2 * gsl_matrix_get(xyt, i, j);
        gsl_matrix_set(distanceMatrix, i, j, dist);
      }
    }
  }

 private:
  gsl_matrix* distanceMatrix = nullptr;
  gsl_vector* meansX = nullptr;
  gsl_vector* meansY = nullptr;
  gsl_matrix* xyt = nullptr;
};

#endif  // SQUARED_EUCLIDIAN_DISTANCE_UTILS_H