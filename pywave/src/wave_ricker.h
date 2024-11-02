#ifndef _ricker_h
#define _ricker_h

// #include "wave_dtype.h"
// #include "wave_komplex.h"


/** Part I: Ricker wavelet ********/
float Ricker(float t, float f0, float t0, float A);
/*< ricker wavelet:
 * f0: peak frequency
 * t0: time lag
 * A: amplitude
 * ************************>*/

// static kiss_fft_cpx *shape;

void ricker_init(int nfft   /* time samples */, 
		 float freq /* frequency */,
		 int order  /* derivative order */);
/*< initialize >*/

void ricker_close(void);
/*< free allocated storage >*/


#endif