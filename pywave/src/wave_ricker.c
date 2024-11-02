#include "wave_dtype.h"
#include "wave_komplex.h"


/** Part I: Ricker wavelet ********/
float Ricker(float t, float f0, float t0, float A) 
/*< ricker wavelet:
 * f0: peak frequency
 * t0: time lag
 * A: amplitude
 * ************************>*/
{
        float x=pow(SF_PI*f0*(t-t0),2);
        return -A*exp(-x)*(1-2*x);
}

static kiss_fft_cpx *shape;

void ricker_init(int nfft   /* time samples */, 
		 float freq /* frequency */,
		 int order  /* derivative order */)
/*< initialize >*/
{
    int iw, nw;
    float dw, w;
    kiss_fft_cpx cw;

    /* determine frequency sampling (for real to complex FFT) */
    nw = nfft/2+1;
    dw = 1./(nfft*freq);
 
    shape = (kiss_fft_cpx*) np_complexalloc(nw);

    for (iw=0; iw < nw; iw++) {
	w = iw*dw;
	w *= w;

	switch (order) {
	    case 2: /* half-order derivative */
		cw.r = 2*SF_PI/nfft;
		cw.i = iw*2*SF_PI/nfft;
		cw = np_csqrtf(cw);
		shape[iw].r = cw.r*w*expf(1-w)/nfft;
		shape[iw].i = cw.i*w*expf(1-w)/nfft;
		break;
	    case 0:
	    default:
		shape[iw].r = w*expf(1-w)/nfft;
		shape[iw].i = 0.;
		break;
	}
    }

    np_freqfilt_init(nfft,nw);
    np_freqfilt_cset(shape);
}

void ricker_close(void) 
/*< free allocated storage >*/
{
    free(shape);
    np_freqfilt_close();
}