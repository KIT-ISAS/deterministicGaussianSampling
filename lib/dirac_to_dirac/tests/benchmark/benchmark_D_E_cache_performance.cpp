#include <benchmark/benchmark.h>
#include <gsl/gsl_vector.h>

static void RegisterBenchmarks_DeCacheRedorderingNaive(
    benchmark::State& state) {
  const size_t L = (size_t)state.range(0);
  const size_t N = (size_t)state.range(1);

  gsl_vector* x = gsl_vector_alloc(L * N);
  if (!x) {
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }
  gsl_vector* wX = gsl_vector_alloc(L);
  if (!wX) {
    gsl_vector_free(x);
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }

  for (auto _ : state) {
    double meanSqrd = 0.0;
    std::vector<double> meanX(N, 0.0);
    for (size_t k = 0; k < N; k++) {
      for (size_t i = 0; i < L; i++) {
        meanX[k] += wX->data[i] * x->data[i * N + k];
      }
    }
    for (size_t k = 0; k < N; k++) {
      meanSqrd += meanX[k] * meanX[k];
    }
    benchmark::DoNotOptimize(meanSqrd);
  }

  state.counters["L"] = (double)L;
  state.counters["N"] = (double)N;

  gsl_vector_free(x);
  gsl_vector_free(wX);
}

static void RegisterBenchmarks_DeCacheRedorderingImproved(
    benchmark::State& state) {
  const size_t L = (size_t)state.range(0);
  const size_t N = (size_t)state.range(1);

  gsl_vector* x = gsl_vector_alloc(L * N);
  if (!x) {
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }
  gsl_vector* wX = gsl_vector_alloc(L);
  if (!wX) {
    gsl_vector_free(x);
    state.SkipWithError("Could not allocate memory for matrices");
    return;
  }

  for (auto _ : state) {
    double meanSqrd = 0.0;
    std::vector<double> meanX(N, 0.0);
    for (size_t i = 0; i < L; i++) {
      const double wXi = wX->data[i];
      for (size_t k = 0; k < N; k++) {
        meanX[k] += wXi * x->data[i * N + k];
      }
    }
    for (size_t k = 0; k < N; k++) {
      meanSqrd += meanX[k] * meanX[k];
    }
    benchmark::DoNotOptimize(meanSqrd);
  }

  state.counters["L"] = (double)L;
  state.counters["N"] = (double)N;

  gsl_vector_free(x);
  gsl_vector_free(wX);
}

static const long long maxL = 10000;
static const long long minL = 100;
static const long long stepL = 100;
static const long long maxN = 25;
static const long long minN = 1;
static const long long stepN = 1;

static void D_E_CustomArguments(benchmark::internal::Benchmark* b) {
  for (int L = minL; L <= maxL; L += stepL)
    for (int N = minN; N <= maxN; N += stepN) b->Args({L, N});
}

void RegisterBenchmarks_DeCacheRedordering() {
  const long long stateIterations = 100;
  const auto timeUnit = benchmark::kMicrosecond;

  benchmark::RegisterBenchmark("BenchmarkDeNaiveOrder",
                               &RegisterBenchmarks_DeCacheRedorderingNaive)
      ->Apply(D_E_CustomArguments)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("BenchmarkDeImprovedOrder",
                               &RegisterBenchmarks_DeCacheRedorderingImproved)
      ->Apply(D_E_CustomArguments)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();
}

// trick to register benchmarks
struct RegisterAllBenchmarks_DeCacheRedordering {
  RegisterAllBenchmarks_DeCacheRedordering() {
    RegisterBenchmarks_DeCacheRedordering();
  }
};
RegisterAllBenchmarks_DeCacheRedordering instance_DeCacheRedordering;
