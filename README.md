# Deterministic Gaussian Sampling (C++)

Deterministic approximation and reduction of multivariate **Dirac mixtures** and **Gaussian distributions** using a high-performance C++ implementation.

The library provides optimization-based approximation algorithms with analytical gradients and optional multi-threaded execution.

📖 **Full API Documentation:**  
https://kit-isas.github.io/deterministicGaussianSampling/

Prebuilt binaries for **Linux** and **Windows** are automatically generated via GitHub Actions and attached to tagged releases.

---

# Requirements

## Core

- C++17 or newer  
- CMake ≥ 3.15  
- OpenMP  
- GSL (GNU Scientific Library)

## Optional

- GoogleTest (unit tests)
- Google Benchmark (benchmarks)

---

# Build

The project supports:

- Native Windows build (MinGW + vcpkg)
- Linux build (Docker-based)
- CI-based release artifacts

---

# 🔹 Windows Build (MinGW + vcpkg)

Dependencies are managed via **vcpkg**.

## 1️⃣ Install MinGW

Install via Chocolatey:

```powershell
choco install mingw -y
```

Ensure MinGW is in your `PATH`.

---

## 2️⃣ Install Dependencies (vcpkg)

```powershell
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat

.\vcpkg\vcpkg install `
    gsl:x64-mingw-static `
    gsl:x64-mingw-dynamic `
    gtest:x64-mingw-static `
    benchmark:x64-mingw-static
```

---

## 3️⃣ Configure and Build

```powershell
mkdir build
cd build

cmake .. -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="../vcpkg/installed/x64-mingw-static;../vcpkg/installed/x64-mingw-dynamic" `
  -DOpenMP_C_FLAGS="-fopenmp" `
  -DOpenMP_CXX_FLAGS="-fopenmp"

cmake --build . --target install
```

---

## Run Tests

```powershell
.\lib\unit.exe
```

---

# 🔹 Linux Build (Docker)

Linux builds are performed inside a Docker container to ensure reproducibility.

## Build Docker Image

```bash
docker build -f Dockerfile.linux -t libapproxlcd-builder .
```

## Run Build Container

```bash
docker run --rm \
  -e BUILD_DIR="/work/build" \
  -e ARTIFACT_DIR="/work/artifacts" \
  -v "$PWD:/work" \
  libapproxlcd-builder
```

Artifacts will be placed in:

```
artifacts/linux
```

---

# 🔹 Prebuilt Binaries

Every tagged release (`v*`) automatically generates:

- `linux.zip`
- `windows.zip`

These archives contain:

- Installed headers
- Libraries
- Unit test executable
- Benchmark executable
- Example application (main)

Download them from the **GitHub Releases** page.

---

# Overview

The library provides the following main components:

```cpp
dirac_to_dirac_approx_short
dirac_to_dirac_approx_short_thread
dirac_to_dirac_approx_short_function
gm_to_dirac_short
gm_to_dirac_short_standard_normal_deviation
```

They allow you to:

- Reduce large Dirac mixtures to compact deterministic representations
- Approximate Gaussian distributions with optimized Dirac support points
- Compute the modified van Mises distance
- Compute the analytic gradient
- Use custom weight functions
- Switch between single-threaded and multi-threaded execution

---

# 1️⃣ Dirac-to-Dirac Reduction

Reduce `M` Dirac components in ℝᴺ to `L < M` optimized deterministic components.

---

## Basic Example (Single-Threaded)

```cpp
#include <vector>
#include <iostream>
#include "dirac_to_dirac_approx_short.h"

size_t M = 3000;
size_t N = 2;
size_t L = 12;
size_t bMax = 10;

std::vector<double> y(M * N);
std::vector<double> x(L * N);

dirac_to_dirac_approx_short<double> reducer;

GslminimizerResult result;

bool ok = reducer.approximate(
    y.data(),
    M,
    L,
    N,
    bMax,
    x.data()
);

std::cout << "Success: " << ok << std::endl;
```

---

## Multi-Threaded Version

```cpp
#include "dirac_to_dirac_approx_short_thread.h"

dirac_to_dirac_approx_short_thread<double> reducer;

bool ok = reducer.approximate(
    y.data(),
    M,
    L,
    N,
    bMax,
    x.data()
);
```

Produces the same result as the single-threaded version, but parallelizes internal computations.

---

## Custom Weight Functions

```cpp
#include <cmath>
#include "dirac_to_dirac_approx_short_function.h"

static void wXcallback(const double* x,
                       double* res,
                       size_t L,
                       size_t N)
{
    for (size_t i = 0; i < L; ++i) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            double v = x[i * N + k];
            sum += v * v;
        }
        res[i] = std::exp(-sum);
    }
}

static void wXDcallback(const double* x,
                        double* grad,
                        size_t L,
                        size_t N)
{
    for (size_t i = 0; i < L; ++i)
        for (size_t k = 0; k < N; ++k)
            grad[i * N + k] = -2.0 * x[i * N + k];
}
```

---

# 2️⃣ Gaussian-to-Dirac Approximation

Approximate a multivariate Gaussian distribution with `L` deterministic Dirac points.

---

## Standard Normal Deviation Variant

```cpp
#include "gm_to_dirac_short_standard_normal_deviation.h"

gm_to_dirac_short_standard_normal_deviation<double> approx;

std::vector<double> x(L * N);

bool ok = approx.approximate(
    L,
    N,
    bMax,
    x.data()
);
```

---

## Diagonal Covariance Variant

```cpp
#include "gm_to_dirac_short.h"

std::vector<double> covDiag = {2.0, 1.5};
std::vector<double> x(L * N);

gm_to_dirac_short<double> approx;

bool ok = approx.approximate(
    covDiag.data(),
    L,
    N,
    bMax,
    x.data()
);
```

---

# Distance and Gradient

### Compute Distance

```cpp
double distance = 0.0;

reducer.modified_van_mises_distance_sq(
    &distance,
    y.data(),
    M,
    L,
    N,
    bMax,
    x.data()
);
```

### Compute Analytic Gradient

```cpp
std::vector<double> gradient(L * N);

reducer.modified_van_mises_distance_sq_derivative(
    gradient.data(),
    y.data(),
    M,
    L,
    N,
    bMax,
    x.data()
);
```

---

# Notes

- High-performance C++ implementation  
- Analytical gradients  
- GSL-based optimization backend  
- OpenMP-enabled threaded variants  
- Docker-based reproducible Linux builds  
- Automated Windows + Linux release packaging  
- Designed for high-dimensional and performance-critical applications  
