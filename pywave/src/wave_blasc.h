#ifndef blasc_h
#define blasc_h

// #include "seis_blas.h"

float cblas_scnrm2 (int n, const void* x, int sx);
/*< sum |x_i|^2 >*/

void cblas_csscal(int n, float alpha, void *x, int sx);
/*< x = alpha*x >*/

void cblas_cdotc_sub(int n, 
		     const void *x, int sx,
		     const void *y, int sy, void *dot);
/*< complex hermitian dot product >*/


#endif