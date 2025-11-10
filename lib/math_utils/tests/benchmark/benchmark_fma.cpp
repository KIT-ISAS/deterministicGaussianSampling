#include <benchmark/benchmark.h>
#include <omp.h>

#include <cmath>

constexpr int kLoopSize = 1'000'000'000;

static void RegisterBenchmarks_fused_multiply_add(benchmark::State& state) {
  double a = 1.23456789, b = 9.87654321, c = 0.00012345;
  double result = 0.0;

  for (auto _ : state) {
#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < kLoopSize; ++i) {
      result += std::fma(a, b, c);
    }
    benchmark::DoNotOptimize(result);
  }
}

static void RegisterBenchmarks_fused_multiply_add_naive(
    benchmark::State& state) {
  double a = 1.23456789, b = 9.87654321, c = 0.00012345;
  double result = 0.0;

  for (auto _ : state) {
#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < kLoopSize; ++i) {
      result += a * b + c;
    }
    benchmark::DoNotOptimize(result);
  }
}

void RegisterBenchmarks_fused_multiply_add() {
  const long long stateIterations = 100;
  const auto timeUnit = benchmark::kMicrosecond;

  benchmark::RegisterBenchmark("fused_multiply_add/naive",
                               &RegisterBenchmarks_fused_multiply_add_naive)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();

  benchmark::RegisterBenchmark("fused_multiply_add/std::fma",
                               &RegisterBenchmarks_fused_multiply_add)
      ->Unit(timeUnit)
      ->Iterations(stateIterations)
      ->UseRealTime();
}

struct RegisterAllBenchmarks_fused_multiply_add {
  RegisterAllBenchmarks_fused_multiply_add() {
    RegisterBenchmarks_fused_multiply_add();
  }
};
RegisterAllBenchmarks_fused_multiply_add instance_fused_multiply_add;