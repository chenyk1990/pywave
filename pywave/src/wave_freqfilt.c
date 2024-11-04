/* Frequency-domain filtering. */
#include <math.h>
#include <stdbool.h>

#include "wave_freqfilt.h"

// #include "_bool.h"
// #include "kiss_fft.h"
/*^*/

// #include "alloc.h"
// #include "komplex.h"
// #include "error.h"
// #include "adjnull.h"
// #include "kiss_fftr.h"
// #include "_kiss_fft_guts.h"

#include "wave_komplex.h"
#include "wave_alloc.h"
#include "wave_conjgrad.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

static int nfft, nw;
static kiss_fft_cpx *cdata, *shape=NULL;
static float *tmp;
static kiss_fftr_cfg forw, invs;

void np_freqfilt_init(int nfft1 /* time samples (possibly padded) */, 
		      int nw1   /* frequency samples */)
/*< Initialize >*/
{
    nfft = nfft1;
    nw = nw1;

    cdata = (kiss_fft_cpx*) np_alloc(nw,sizeof(kiss_fft_cpx));
    tmp = np_floatalloc(nfft);
    forw = kiss_fftr_alloc(nfft,0,NULL,NULL);
    invs = kiss_fftr_alloc(nfft,1,NULL,NULL);
    if (NULL == forw || NULL == invs) printf("KISS FFT allocation problem\n");
}

void np_freqfilt_set(float *filt /* frequency filter [nw] */)
/*< Initialize filter (zero-phase) >*/
{
    int iw;
    
    if (NULL==shape) shape = (kiss_fft_cpx*) np_alloc(nw,sizeof(kiss_fft_cpx));

    for (iw=0; iw < nw; iw++) {
	shape[iw].r = filt[iw];
	shape[iw].i = 0.;
    }
}

/* #ifndef __cplusplus */
/*^*/

void np_freqfilt_cset(kiss_fft_cpx *filt /* frequency filter [nw] */)
/*< Initialize filter >*/
{
    shape = filt;
}

void np_freqfilt_close(void) 
/*< Free allocated storage >*/
{
    free(cdata);
    free(tmp);
    free(forw);
    free(invs);
}

void np_freqfilt(int nx, float* x)
/*< Filtering in place >*/
{
    kiss_fft_cpx c;
    int iw;

    for (iw=0; iw < nx; iw++) {
	tmp[iw] = x[iw];
    }
    for (iw=nx; iw < nfft; iw++) {
	tmp[iw] = 0.;
    }

    kiss_fftr(forw, tmp, cdata);
    for (iw=0; iw < nw; iw++) {
	C_MUL(c,cdata[iw],shape[iw]);
	cdata[iw]=c;
    }
    kiss_fftri(invs, cdata, tmp);

    for (iw=0; iw < nx; iw++) {
	x[iw] = tmp[iw];
    } 
}

void np_freqfilt_lop (bool adj, bool add, int nx, int ny, float* x, float* y) 
/*< Filtering as linear operator >*/
{
    kiss_fft_cpx c;
    int iw;

    np_adjnull(adj,add,nx,ny,x,y);

    for (iw=0; iw < nx; iw++) {
	tmp[iw] = adj? y[iw] : x[iw];
    }
    for (iw=nx; iw < nfft; iw++) {
	tmp[iw] = 0.;
    }

    kiss_fftr(forw, tmp, cdata);
    for (iw=0; iw < nw; iw++) {
        if (adj) {
	    C_MUL(c,cdata[iw],np_conjf(shape[iw]));
        } else {
            C_MUL(c,cdata[iw],shape[iw]);
        }
	cdata[iw]=c;
    }
    kiss_fftri(invs, cdata, tmp);

    for (iw=0; iw < nx; iw++) {	    
	if (adj) {
	    x[iw] += tmp[iw];
	} else {
	    y[iw] += tmp[iw];
	}
    } 
}




