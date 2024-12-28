#include "wave_blasc.h"
#include "wave_komplex.h"

float cblas_scnrm2 (int n, const void* x, int sx) 
/*< sum |x_i|^2 >*/
{
    int i, ix;
    float xn;
    const np_complex* c;

    c = (const np_complex*) x;

    xn = 0.0;

    for (i=0; i < n; i++) {
	ix = i*sx;
	xn += crealf(np_cmul(c[ix],conjf(c[ix])));
    }
    return xn;
}

void cblas_csscal(int n, float alpha, void *x, int sx)
/*< x = alpha*x >*/
{
    int i, ix;
    np_complex* c;

    c = (np_complex*) x;

    for (i=0; i < n; i++) {
        ix = i*sx;
	c[ix] = np_crmul(c[ix],alpha);
    }
}


void cblas_cdotc_sub(int n, 
		     const void *x, int sx,
		     const void *y, int sy, void *dot)
/*< complex hermitian dot product >*/
{
    np_complex *cx, *cy, xi, yi;
    np_double_complex prod, pi;
    int i, ix, iy;
    
    cx = (np_complex*) x;
    cy = (np_complex*) y;

    prod = np_dcmplx(0.,0.);

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	xi = cx[ix];
	yi = cy[iy];
	pi = np_dcmplx((double) crealf(xi)*crealf(yi) + 
		       (double) cimagf(xi)*cimagf(yi), 
		       (double) crealf(xi)*cimagf(yi) - 
		       (double) cimagf(xi)*crealf(yi));
	prod = np_dcadd(prod,pi);
    }

    *((np_complex *) dot) = np_cmplx(creal(prod),cimag(prod));
}


