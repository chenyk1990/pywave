#ifndef blas_h
#define blas_h

// #include "seis_blas.h"

double cblas_dsdot(int n, const float *x, int sx, const float *y, int sy);
/*< x'y float -> double >*/

void cblas_saxpy(int n, float a, const float *x, int sx, float *y, int sy);
/*< y += a*x >*/

void cblas_sswap(int n, float *x, int sx, float* y, int sy);
/*< swap x and y >*/

float cblas_sdot(int n, const float *x, int sx, const float *y, int sy);
/*< x'y float -> complex >*/

double cblas_dsdot(int n, const float *x, int sx, const float *y, int sy);
/*< x'y float -> complex >*/

float cblas_snrm2 (int n, const float* x, int sx);
/*< sum x_i^2 >*/

void cblas_sscal(int n, float alpha, float *x, int sx);
/*< x = alpha*x >*/

#endif

