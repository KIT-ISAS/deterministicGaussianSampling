#include <benchmark/benchmark.h>

#include <type_traits>

#include "squared_euclidean_distance_utils.h"

template <typename SquaredEuclideanDistanceUtilsType>
static void SquaredEuclideanDistanceUtilsBenchmark(benchmark::State& state) {
  const size_t L = (size_t)state.range(0);
  const size_t M = (size_t)(state.range(1) * state.range(0));
  const size_t N = (size_t)state.range(2);

  gsl_matrix* x = nullptr;
  gsl_matrix* y = nullptr;

  SquaredEuclideanDistanceUtilsType* sedistanceUtils = nullptr;
  if constexpr (std::is_constructible_v<SquaredEuclideanDistanceUtilsType,
                                        size_t, size_t, size_t>) {
    x = gsl_matrix_alloc(L, N);
    y = gsl_matrix_alloc(M, N);
    if (!x || !y) {
      state.SkipWithError("Could not allocate memory for matrices");
      return;
    }
    for (size_t i = 0; i < (L * N); ++i)
      x->data[i] = (std::rand() / (double)RAND_MAX);
    for (size_t i = 0; i < (M * N); ++i)
      y->data[i] = (std::rand() / (double)RAND_MAX);

    sedistanceUtils = new SquaredEuclideanDistanceUtilsType(L, M, N);
    state.SetComplexityN(static_cast<benchmark::ComplexityN>(L * M * N));
  } else if constexpr (std::is_constructible_v<
                           SquaredEuclideanDistanceUtilsType, size_t, size_t>) {
    x = gsl_matrix_alloc(L, N);
    if (!x) {
      state.SkipWithError("Could not allocate memory for matrices");
      return;
    }
    for (size_t i = 0; i < (L * N); ++i)
      x->data[i] = (std::rand() / (double)RAND_MAX);

    sedistanceUtils = new SquaredEuclideanDistanceUtilsType(L, N);
    state.SetComplexityN(static_cast<benchmark::ComplexityN>(L * L * N));
  } else {
    static_assert(sizeof(SquaredEuclideanDistanceUtilsType) == 0,
                  "SquaredEuclideanDistanceUtilsType must be constructible "
                  "with (x, y) or (x)");
  }

  for (auto _ : state) {
    sedistanceUtils->calculateDistance(x, y);
  }

  state.counters["L"] = (double)L;
  state.counters["M"] = (double)M;
  state.counters["N"] = (double)N;
  state.counters["M/L"] = (double)M / (double)L;

  if (x) gsl_matrix_free(x);
  if (y) gsl_matrix_free(y);
  delete sedistanceUtils;
}

static const long long maxM = 50;
static const long long minM = 10;
static const long long stepM = 20;
static const long long maxL = 1000;
static const long long minL = 100;
static const long long stepL = 100;
static const long long maxN = 25;
static const long long minN = 1;
static const long long stepN = 1;

static void SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLM(
    benchmark::Benchmark* b) {
  for (int L = minL; L <= maxL; L += stepL)
    for (int M = minM; M <= maxM; M += stepM)
      for (int N = minN; N <= maxN; N += stepN) b->Args({L, M, N});
}
static void SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLL(
    benchmark::Benchmark* b) {
  for (int L = minL; L <= maxL; L += stepL)
    for (int N = minN; N <= maxN; N += stepN) b->Args({L, 0, N});
}

void RegisterBenchmarks_SquaredEuclideanDistanceUtilsBenchmark() {
  const long long stateIterations = 10;
  const auto timeUnit = benchmark::kMicrosecond;

  // SquaredEuclideanDistance_LL_matrix
  benchmark::RegisterBenchmark("SquaredEuclideanDistance_LL_matrix",
                               &SquaredEuclideanDistanceUtilsBenchmark<
                                   SquaredEuclideanDistance_LL_matrix>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLL)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();

  // SquaredEuclideanDistance_LL_matrix_optimized
  benchmark::RegisterBenchmark(
      "SquaredEuclideanDistance_LL_matrix_optimized",
      &SquaredEuclideanDistanceUtilsBenchmark<
          SquaredEuclideanDistance_LL_matrix_optimized>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLL)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();

  // SquaredEuclideanDistance_LL_vectorized
  benchmark::RegisterBenchmark("SquaredEuclideanDistance_LL_vectorized",
                               &SquaredEuclideanDistanceUtilsBenchmark<
                                   SquaredEuclideanDistance_LL_vectorized>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLL)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();

  // SquaredEuclideanDistance_LL_vectorized_optimized
  benchmark::RegisterBenchmark(
      "SquaredEuclideanDistance_LL_vectorized_optimized",
      &SquaredEuclideanDistanceUtilsBenchmark<
          SquaredEuclideanDistance_LL_vectorized_optimized>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLL)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();

  // SquaredEuclideanDistance_LM_matrix
  benchmark::RegisterBenchmark("SquaredEuclideanDistance_LM_matrix",
                               &SquaredEuclideanDistanceUtilsBenchmark<
                                   SquaredEuclideanDistance_LM_matrix>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLM)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();

  // SquaredEuclideanDistance_LM_matrix_optimized
  benchmark::RegisterBenchmark(
      "SquaredEuclideanDistance_LM_matrix_optimized",
      &SquaredEuclideanDistanceUtilsBenchmark<
          SquaredEuclideanDistance_LM_matrix_optimized>)
      ->Apply(SquaredEuclideanDistanceUtilsBenchmark_CustomArgumentsLM)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->Complexity()
      ->UseRealTime();
}

// trick to register benchmarks
struct RegisterAllBenchmarks_SquaredEuclideanDistanceUtilsBenchmark {
  RegisterAllBenchmarks_SquaredEuclideanDistanceUtilsBenchmark() {
    RegisterBenchmarks_SquaredEuclideanDistanceUtilsBenchmark();
  }
};
RegisterAllBenchmarks_SquaredEuclideanDistanceUtilsBenchmark
    instance_SquaredEuclideanDistanceUtilsBenchmark;
