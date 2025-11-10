#ifndef GM_TO_DIRAC_SHORT_TPP
#define GM_TO_DIRAC_SHORT_TPP

// #define USE_CACHE_MANAGER

template <typename T>
double gm_to_dirac_short<T>::calculateP2(double b, void* params) {
  GMToDiracConstWeightOptimizationParams* optiParams =
      static_cast<GMToDiracConstWeightOptimizationParams*>(params);
  const size_t N = optiParams->N;
  const size_t L = optiParams->L;
  const gsl_vector* wX = optiParams->wX;
  const gsl_vector* covDiagSqrd = optiParams->covDiagSqrd;
  const double twoPiNHalf = optiParams->twoPiNHalf;

  GMToDiracIntegrationParams* integrationParams = optiParams->integrationParams;
  const gsl_vector* x = integrationParams->x;
#ifdef USE_CACHE_MANAGER
  floatingBucketCacheManager* cacheManagerPrefactor =
      integrationParams->cacheManagerPrefactor;
  floatingBucketCacheManagerIntKey* cacheManagerInnerSum =
      integrationParams->cacheManagerInnerSum;
#endif

  const double twoBSqrd = 2.00 * b * b;

  double prefactor;
#ifdef USE_CACHE_MANAGER
  if (!cacheManagerPrefactor || !cacheManagerPrefactor->get(b, &prefactor)) {
#endif
    prefactor = twoPiNHalf * gsl_sf_pow_int(b, (int)(N + 1));
    for (size_t k = 0; k < N; k++) {
      prefactor *= 1.00 / std::sqrt(covDiagSqrd->data[k] + twoBSqrd);
    }
#ifdef USE_CACHE_MANAGER
    if (cacheManagerPrefactor) cacheManagerPrefactor->set(b, prefactor);
  }
#endif

  double sum = 0.00;
  for (size_t i = 0; i < L; i++) {
    const double wXi = wX->data[i];
    double innerSum = 0.00;
    for (size_t k = 0; k < N; k++) {
      const double xikSqrd = x->data[i * N + k] * x->data[i * N + k];
      innerSum += xikSqrd / (covDiagSqrd->data[k] + twoBSqrd);
    }
    const double expValuei = wXi * std::exp(-0.5 * innerSum);
#ifdef USE_CACHE_MANAGER
    if (cacheManagerInnerSum) cacheManagerInnerSum->set(b, i, expValuei);
#endif
    sum += expValuei;
  }

  const double ret = prefactor * sum;
  return ret;
}

template <typename T>
double gm_to_dirac_short<T>::calculateGradP2(double b, void* params) {
  GMToDiracConstWeightOptimizationParams* optiParams =
      static_cast<GMToDiracConstWeightOptimizationParams*>(params);
  const size_t N = optiParams->N;
  const gsl_vector* covDiagSqrd = optiParams->covDiagSqrd;

  GMToDiracIntegrationParams* integrationParams = optiParams->integrationParams;
  int tid = omp_get_thread_num();
  const size_t xi = integrationParams->xi[tid];
  const size_t eta = integrationParams->eta[tid];
  const gsl_vector* x = integrationParams->x;
#ifdef USE_CACHE_MANAGER
  floatingBucketCacheManager* cacheManagerPrefactor =
      integrationParams->cacheManagerPrefactor;
  floatingBucketCacheManagerIntKey* cacheManagerInnerSum =
      integrationParams->cacheManagerInnerSum;
#endif

  const double twoBSqrd = 2.00 * b * b;

  double prefactor;
#ifdef USE_CACHE_MANAGER
  if (!cacheManagerPrefactor || !cacheManagerPrefactor->get(b, &prefactor)) {
#endif
    prefactor =
        gsl_sf_pow_int(b, (int)(N + 1)) / (covDiagSqrd->data[eta] + twoBSqrd);
    for (size_t k = 0; k < N; k++) {
      prefactor *= 1.00 / std::sqrt(covDiagSqrd->data[k] + twoBSqrd);
    }
#ifdef USE_CACHE_MANAGER
  }
#endif

  double expValuei = 0.00;
#ifdef USE_CACHE_MANAGER
  if (!cacheManagerInnerSum || !cacheManagerInnerSum->get(b, xi, &expValuei)) {
#endif
    double innerSum = 0.00;
    for (size_t k = 0; k < N; k++) {
      const double xikSqrd = x->data[xi * N + k] * x->data[xi * N + k];
      innerSum += xikSqrd / (covDiagSqrd->data[k] + twoBSqrd);
    }
    expValuei = std::exp(-0.5 * innerSum);
#ifdef USE_CACHE_MANAGER
  }
#endif

  const double ret = prefactor * expValuei;
  return ret;
}

template <typename T>
void gm_to_dirac_short<T>::calculateD2(
    const gsl_vector* x, GMToDiracConstWeightOptimizationParams* params,
    double* f, gsl_vector* grad) {
  const size_t L = params->L;
  const size_t N = params->N;
  const gsl_vector* wX = params->wX;
  const double twoPiNHalf = params->twoPiNHalf;
  const GslQuadratureAdaptiveGaussKronrod* integrationUtils =
      params->gaussKronrod;
  params->integrationParams->reset(x);

  double result = 0.00;
  double abserr = 0.00;

  if (f) {
    integrationUtils->integrate(calculateP2, params, 0, (double)(params->bMax),
                                &result, &abserr);
    *f += -2.00 * result;
  }

  if (grad) {
#pragma omp parallel for
    for (size_t i = 0; i < L; i++) {
      for (size_t k = 0; k < N; k++) {
        double localResult = 0.00;
        double localAbserr = 0.00;
        int tid = omp_get_thread_num();
        params->integrationParams->xi[tid] = i;
        params->integrationParams->eta[tid] = k;
        integrationUtils->integrate(calculateGradP2, params, 0,
                                    (double)(params->bMax), &localResult,
                                    &localAbserr);
        grad->data[i * N + k] +=
            2.00 * twoPiNHalf * wX->data[i] * x->data[i * N + k] * localResult;
      }
    }
  }
}

template <typename T>
inline void gm_to_dirac_short<T>::calculateD3(
    const gsl_vector* x, GMToDiracConstWeightOptimizationParams* params,
    double* f, gsl_vector* grad) {
  const size_t N = params->N;
  const size_t L = params->L;
  const gsl_vector* wX = params->wX;
  const size_t bMax = params->bMax;
  gsl_vector* localDist = params->vecN;
  const double bMaxSqrd = static_cast<double>(bMax * bMax);
  const double bMaxSqrdHalf = bMaxSqrd / 2.00;
  const double cB = params->cB;
  (void)cB;

  const double logApprox = 0.5772156649015 - 1.00 - 2.00 * std::log(2.00);

  const int numThreads = omp_get_max_threads();
  std::vector<double> d3((size_t)numThreads, 0.00);

  for (size_t i = 0; i < L; i++) {
    const int tid = omp_get_thread_num();
    const double wXi = wX->data[i];
    for (size_t j = 0; j < L; j++) {
      const double wXiwXj = wXi * wX->data[j];
      double localDistSq = 0.00;
      if (i == j) {
        if (f) d3[(size_t)tid] += bMaxSqrdHalf * wXiwXj;
        continue;
      }
      for (size_t k = 0; k < N; k++) {
        localDist->data[k] = x->data[i * N + k] - x->data[j * N + k];
        localDistSq += localDist->data[k] * localDist->data[k];
      }
      if (localDistSq <= 0.0) {
        if (f) d3[(size_t)tid] += bMaxSqrdHalf * wXiwXj;
        continue;
      }
      const double logLocalDistSq = std::log(localDistSq);
      if (f) {
        d3[(size_t)tid] += 0.125 * wXiwXj *
                           (4.00 * bMaxSqrd +
                            (logApprox - 2.00 * std::log(bMax)) * localDistSq +
                            (localDistSq * logLocalDistSq));
      }
      if (grad) {
        const double constFactor =
            0.5 * wXiwXj *
            (0.5772156649015 - 2.00 * std::log(2.00 * (double)(bMax)) +
             logLocalDistSq);
        for (size_t k = 0; k < N; ++k) {
          grad->data[i * N + k] += constFactor * localDist->data[k];
        }
      }
    }
  }
  if (f) {
    for (size_t i = 0; i < (size_t)numThreads; ++i) {
      *f += d3[i];
    }
  }
}

template <typename T>
inline void gm_to_dirac_short<T>::correctMean(gsl_vector* x,
                                              const gsl_vector* wX, size_t L,
                                              size_t N) {
  std::vector<double> mean(N, 0.00);
  for (size_t i = 0; i < L; ++i) {
    const double wXi = wX->data[i];
    for (size_t k = 0; k < N; ++k) {
      mean[k] += wXi * x->data[i * N + k];
    }
  }
  for (size_t i = 0; i < L; i++) {
    for (size_t k = 0; k < N; k++) {
      x->data[i * N + k] -= mean[k];
    }
  }
}

#endif  // GM_TO_DIRAC_SHORT_TPP