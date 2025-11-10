#include <benchmark/benchmark.h>
#include <gsl/gsl_vector.h>

#include <cmath>

#include "gm_to_dirac_short.h"

class benchmark_gm_to_dirac_short {
 public:
  static gm_to_dirac_short<double> gmToDiracShortThreadDoubleInstance;

  static double calculateD1(size_t bMax, const gsl_vector* covDiag, size_t N) {
    return GMToDiracBaseOptimizationParams::calculateD1(bMax, covDiag, N);
  }

  static void calculateD2(const gsl_vector* x,
                          GMToDiracConstWeightOptimizationParams* params,
                          double* f, gsl_vector* grad) {
    return gmToDiracShortThreadDoubleInstance.calculateD2(x, params, f, grad);
  }

  static void calculateD3(const gsl_vector* x,
                          GMToDiracConstWeightOptimizationParams* params,
                          double* f, gsl_vector* grad) {
    return gmToDiracShortThreadDoubleInstance.calculateD3(x, params, f, grad);
  }
};
gm_to_dirac_short<double>
    benchmark_gm_to_dirac_short::gmToDiracShortThreadDoubleInstance;

static void RegisterBenchmarks_runtimeD1(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  const size_t bMax = 100;

  gsl_vector* covDiag = gsl_vector_alloc(N);
  gsl_vector_set_all(covDiag, 1.00);

  for (auto _ : state) {
    double result = benchmark_gm_to_dirac_short::calculateD1(bMax, covDiag, N);
    benchmark::DoNotOptimize(result);
  }

  state.counters["N"] = (double)N;

  gsl_vector_free(covDiag);
}

static void RegisterBenchmarks_runtimeD2(benchmark::State& state) {
  const size_t L = (size_t)state.range(0);
  const size_t N = (size_t)state.range(1);
  const size_t bMax = 100;
  double distance = 0.00;
  gsl_vector* x = gsl_vector_alloc(L * N);
  std::srand(42);
  for (size_t i = 0; i < L * N; i++) {
    x->data[i] = (std::rand() / (double)RAND_MAX);
  }
  gsl_vector* grad = gsl_vector_alloc(L * N);
  gsl_vector* covDiag = gsl_vector_alloc(N);
  gsl_vector_set_all(covDiag, 1.00);
  GMToDiracConstWeightOptimizationParams params =
      GMToDiracConstWeightOptimizationParams(covDiag, N, L, bMax, 10.00);

  params.integrationParams->reset(x);

  for (auto _ : state) {
    benchmark_gm_to_dirac_short::calculateD2(x, &params, &distance, grad);
    benchmark::DoNotOptimize(distance);
    benchmark::DoNotOptimize(grad);
  }

  state.counters["L"] = (double)L;
  state.counters["N"] = (double)N;

  gsl_vector_free(x);
  gsl_vector_free(grad);
  gsl_vector_free(covDiag);
}

static void RegisterBenchmarks_runtimeD3(benchmark::State& state) {
  const size_t L = (size_t)state.range(0);
  const size_t N = (size_t)state.range(1);
  const size_t bMax = 100;
  double distance = 0.00;
  gsl_vector* x = gsl_vector_alloc(L * N);
  std::srand(42);
  for (size_t i = 0; i < L * N; i++) {
    x->data[i] = (std::rand() / (double)RAND_MAX);
  }
  gsl_vector* grad = gsl_vector_alloc(L * N);
  gsl_vector* covDiag = gsl_vector_alloc(N);
  gsl_vector_set_all(covDiag, 1.00);
  GMToDiracConstWeightOptimizationParams params =
      GMToDiracConstWeightOptimizationParams(covDiag, N, L, bMax, 10.00);

  params.integrationParams->reset(x);

  for (auto _ : state) {
    benchmark_gm_to_dirac_short::calculateD3(x, &params, &distance, grad);
    benchmark::DoNotOptimize(distance);
    benchmark::DoNotOptimize(grad);
  }

  state.counters["L"] = (double)L;
  state.counters["N"] = (double)N;

  gsl_vector_free(x);
  gsl_vector_free(grad);
  gsl_vector_free(covDiag);
}

static const long long maxL = 1000;
static const long long minL = 100;
static const long long stepL = 100;
static const long long maxN = 25;
static const long long minN = 1;
static const long long stepN = 4;
static const long long minAcc = 60;
static const long long maxAcc = 130;

static void gm_to_dirac_short_CustomArguments_LN(
    benchmark::internal::Benchmark* b) {
  for (int L = minL; L <= maxL; L += stepL)
    for (int N = minN; N <= maxN; N += stepN) b->Args({L, N});
}

static void gm_to_dirac_short_CustomArguments_N(
    benchmark::internal::Benchmark* b) {
  for (int N = minN; N <= maxN; N += stepN) b->Args({N});
}

void RegisterBenchmarks_gm_to_dirac_short() {
  const long long stateIterations = 50;
  const auto timeUnit = benchmark::kMicrosecond;

  benchmark::RegisterBenchmark("gm_to_dirac_short/runtimeD1",
                               &RegisterBenchmarks_runtimeD1)
      ->Apply(gm_to_dirac_short_CustomArguments_N)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("gm_to_dirac_short/runtimeD2",
                               &RegisterBenchmarks_runtimeD2)
      ->Apply(gm_to_dirac_short_CustomArguments_LN)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("gm_to_dirac_short/runtimeD3",
                               &RegisterBenchmarks_runtimeD3)
      ->Apply(gm_to_dirac_short_CustomArguments_LN)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();
}

// trick to register benchmarks
struct RegisterAllBenchmarks_gm_to_dirac_short {
  RegisterAllBenchmarks_gm_to_dirac_short() {
    RegisterBenchmarks_gm_to_dirac_short();
  }
};
RegisterAllBenchmarks_gm_to_dirac_short instance_gm_to_dirac_short;
