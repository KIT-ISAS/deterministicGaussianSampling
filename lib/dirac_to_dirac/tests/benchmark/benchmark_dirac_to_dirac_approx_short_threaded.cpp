#include <benchmark/benchmark.h>
#include <dirac_to_dirac_optimization_params.h>
#include <gsl/gsl_vector.h>

#include "dirac_to_dirac_approx_short_thread.h"

class benchmark_dirac_to_dirac_approx_short_thread {
 public:
  static dirac_to_dirac_approx_short_thread<double>
      diracToDiracApproxShortThreadDoubleInstance;

  static double modified_van_mises_distance_sq(const gsl_vector* x,
                                               void* params) {
    return diracToDiracApproxShortThreadDoubleInstance
        .modified_van_mises_distance_sq(x, params);
  }

  static void modified_van_mises_distance_sq_derivative(const gsl_vector* x,
                                                        void* params,
                                                        gsl_vector* grad) {
    diracToDiracApproxShortThreadDoubleInstance
        .modified_van_mises_distance_sq_derivative(x, params, grad);
  }
};
dirac_to_dirac_approx_short_thread<double>
    benchmark_dirac_to_dirac_approx_short_thread::
        diracToDiracApproxShortThreadDoubleInstance;

static void DiracToDiracApproxShortThreadBenchmarkApproximate(
    benchmark::State& state) {
  const size_t ratioMN = (size_t)state.range(0);
  const size_t M = (size_t)state.range(1);
  const size_t N = (size_t)state.range(2);
  const size_t L = (M / ratioMN);

  gsl_vector* x = gsl_vector_alloc(L * N);
  gsl_vector* y = gsl_vector_alloc(M * N);

  if (!x || !y) {
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }

  for (size_t i = 0; i < (M * N); ++i)
    y->data[i] = (std::rand() / (double)RAND_MAX);

  for (auto _ : state) {
    benchmark_dirac_to_dirac_approx_short_thread::
        diracToDiracApproxShortThreadDoubleInstance.approximate(y, L, N, 100,
                                                                x);
  }

  state.counters["L"] = (double)L;
  state.counters["M"] = (double)M;
  state.counters["N"] = (double)N;
  if (x) gsl_vector_free(x);
  if (y) gsl_vector_free(y);
}

static void DiracToDiracApproxShortThreadBenchmarkDistance(
    benchmark::State& state) {
  const size_t ratioMN = (size_t)state.range(0);
  const size_t M = (size_t)state.range(1);
  const size_t N = (size_t)state.range(2);
  const size_t L = (M / ratioMN);

  gsl_vector* x = gsl_vector_alloc(L * N);
  gsl_vector* y = gsl_vector_alloc(M * N);

  if (!x || !y) {
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }

  for (size_t i = 0; i < (M * N); i++)
    y->data[i] = (std::rand() / (double)RAND_MAX);
  for (size_t i = 0; i < (L * N); i++)
    x->data[i] = (std::rand() / (double)RAND_MAX);

  DiracToDiracConstWeightOptimizationParams params = {y, N, M, L, 100, 10.0};

  for (auto _ : state) {
    benchmark_dirac_to_dirac_approx_short_thread::
        modified_van_mises_distance_sq(x, &params);
  }

  state.counters["L"] = (double)L;
  state.counters["M"] = (double)M;
  state.counters["N"] = (double)N;
  if (x) gsl_vector_free(x);
  if (y) gsl_vector_free(y);
}

static void DiracToDiracApproxShortThreadBenchmarkGradient(
    benchmark::State& state) {
  const size_t ratioMN = (size_t)state.range(0);
  const size_t M = (size_t)state.range(1);
  const size_t N = (size_t)state.range(2);
  const size_t L = (M / ratioMN);

  gsl_vector* x = gsl_vector_alloc(L * N);
  gsl_vector* y = gsl_vector_alloc(M * N);
  gsl_vector* grad = gsl_vector_alloc(L * N);

  if (!x || !y || !grad) {
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }

  for (size_t i = 0; i < (M * N); i++)
    y->data[i] = (std::rand() / (double)RAND_MAX);
  for (size_t i = 0; i < (L * N); i++)
    x->data[i] = (std::rand() / (double)RAND_MAX);

  DiracToDiracConstWeightOptimizationParams params = {y, N, M, L, 100, 10.0};

  for (auto _ : state) {
    benchmark_dirac_to_dirac_approx_short_thread::
        modified_van_mises_distance_sq_derivative(x, &params, grad);
  }

  state.counters["L"] = (double)L;
  state.counters["M"] = (double)M;
  state.counters["N"] = (double)N;
  if (x) gsl_vector_free(x);
  if (y) gsl_vector_free(y);
  if (grad) gsl_vector_free(grad);
}

static const long long maxM = 2000;
static const long long minM = 500;
static const long long stepM = 50;
static const long long maxN = 20;
static const long long minN = 10;
static const long long stepN = 10;
static const float minRatioMN = 100.0f;
static const float maxRatioMN = 100.0f;
static const float stepRatioMN = 10.0f;

static void D2D_CustomArguments(benchmark::internal::Benchmark* b) {
  for (float MN = minRatioMN; MN <= maxRatioMN; MN += stepRatioMN)
    for (float M = minM; M <= maxM; M += stepM)
      for (float N = minN; N <= maxN; N += stepN)
        b->Args({(long long)MN, (long long)M, (long long)N});
}

void RegisterBenchmarks_DiracToDiracApproxShortThreadedBenchmark() {
  const long long stateIterations = 10;
  const auto timeUnit = benchmark::kMillisecond;

  benchmark::RegisterBenchmark(
      "DiracToDiracApproxShortThread-Approximate",
      &DiracToDiracApproxShortThreadBenchmarkApproximate)
      ->Apply(D2D_CustomArguments)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("DiracToDiracApproxShortThread-Distance",
                               &DiracToDiracApproxShortThreadBenchmarkDistance)
      ->Apply(D2D_CustomArguments)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("DiracToDiracApproxShortThread-Gradient",
                               &DiracToDiracApproxShortThreadBenchmarkGradient)
      ->Apply(D2D_CustomArguments)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();
}

// trick to register benchmarks
struct RegisterAllBenchmarks_DiracToDiracApproxShortThreadedBenchmark {
  RegisterAllBenchmarks_DiracToDiracApproxShortThreadedBenchmark() {
    RegisterBenchmarks_DiracToDiracApproxShortThreadedBenchmark();
  }
};
RegisterAllBenchmarks_DiracToDiracApproxShortThreadedBenchmark
    instance_DiracToDiracApproxShortThreadedBenchmark;
