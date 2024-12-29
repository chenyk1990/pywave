/* Butterworth filtering. */

/* Implementation is inspired by D. Hale and J.F. Claerbout, 1983, Butterworth
 * dip filters: Geophysics, 48, 1033-1038. */
    
// #include "_bool.h"
// /*^*/
// 
// #include "butter.h"
// #include "_defs.h"
// #include "c99.h"
#include <stdbool.h>
#include "wave_alloc.h"
#include "wave_butter.h"
#define np_PI (3.14159265358979323846264338328)

np_butter np_butter_init(bool low     /* low-pass (or high-pass) */, 
		   float cutoff /* cut off frequency */, 
		   int nn       /* number of poles */)
/*< initialize >*/
{
    int j;
    float arg, ss, sinw, cosw, fact;
    np_butter bw;

    arg = 2.*np_PI*cutoff;
    sinw = sinf(arg);
    cosw = cosf(arg);

    bw = (np_butter) np_alloc (1,sizeof(*bw));
    bw->nn = nn;
    bw->low = low;
    bw->den = np_floatalloc2(2,(nn+1)/2);

    if (nn%2) {
	if (low) {
	    fact = (1.+cosw)/sinw;
	    bw->den[nn/2][0] = 1./(1.+fact);
	    bw->den[nn/2][1] = 1.-fact;
	} else {
	    fact = sinw/(1.+cosw);
	    bw->den[nn/2][0] = 1./(fact+1.);
	    bw->den[nn/2][1] = fact-1.;
	}
    }

    fact = low? sinf(0.5*arg): cosf(0.5*arg);
    fact *= fact;
    
    for (j=0; j < nn/2; j++) {
	ss = sinf(np_PI*(2*j+1)/(2*nn))*sinw;
	bw->den[j][0] = fact/(1.+ss);
	bw->den[j][1] = (1.-ss)/fact;
    }
    bw->mid = -2.*cosw/fact;

    return bw;
}

void np_butter_close(np_butter bw)
/*< Free allocated storage >*/
{
    free(bw->den[0]);
    free(bw->den);
    free(bw);
}

void np_butter_apply (const np_butter bw, int nx, float *x /* data [nx] */)
/*< filter the data (in place) >*/
{
    int ix, j, nn;
    float d0, d1, d2, x0, x1, x2, y0, y1, y2;

    d1 = bw->mid;
    nn = bw->nn;

    if (nn%2) {
	d0 = bw->den[nn/2][0];
	d2 = bw->den[nn/2][1];
	x0 = y1 = 0.;
	for (ix=0; ix < nx; ix++) { 
	    x1 = x0; x0 = x[ix];
	    y0 = (bw->low)? 
		(x0 + x1 - d2 * y1)*d0:
		(x0 - x1 - d2 * y1)*d0;
	    x[ix] = y1 = y0;
	}
    }

    for (j=0; j < nn/2; j++) {
	d0 = bw->den[j][0];
	d2 = bw->den[j][1];
	x1 = x0 = y1 = y2 = 0.;
	for (ix=0; ix < nx; ix++) { 
	    x2 = x1; x1 = x0; x0 = x[ix];
	    y0 = (bw->low)? 
		(x0 + 2*x1 + x2 - d1 * y1 - d2 * y2)*d0:
		(x0 - 2*x1 + x2 - d1 * y1 - d2 * y2)*d0;
	    y2 = y1; x[ix] = y1 = y0;
	}
    }
}

void np_reverse (int n1, float* trace)
/*< reverse a trace >*/
{
    int i1;
    float t;

    for (i1=0; i1 < n1/2; i1++) { 
        t=trace[i1];
        trace[i1]=trace[n1-1-i1];
        trace[n1-1-i1]=t;
    }
}

