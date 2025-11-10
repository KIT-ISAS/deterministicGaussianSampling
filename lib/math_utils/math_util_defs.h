#ifndef MATH_UTIL_DEFS_H
#define MATH_UTIL_DEFS_H

#include <cmath>

// Epsilons
#define EPS_1 1e-1
#define EPS_2 1e-2
#define EPS_3 1e-3
#define EPS_4 1e-4
#define EPS_5 1e-5
#define EPS_6 1e-6
#define EPS_7 1e-7
#define EPS_8 1e-8
#define EPS_9 1e-9
#define EPS_10 1e-10
#define EPS_11 1e-11
#define EPS_12 1e-12
#define EPS_13 1e-13
#define EPS_14 1e-14
#define EPS_15 1e-15
#define EPS_16 1e-16
#define EPS_17 1e-17
#define EPS_18 1e-18
#define EPS_19 1e-19
#define EPS_20 1e-20
#define EPS_21 1e-21
#define EPS_22 1e-22

// Diagamma constant
#define diagamma_1 -0.5772156649015328606065120900824

// fma
#define FMA_ACC(a, b, c) c = std::fma(a, b, c)

#endif  // MATH_UTIL_DEFS_H