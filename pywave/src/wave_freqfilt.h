#ifndef freqfilt_h
#define freqfilt_h

#include <stdbool.h>
#include "wave_dtype.h"

// #include "_bool.h"
// #include "kiss_fft.h"
/*^*/

// #include "alloc.h"
// #include "komplex.h"
// #include "error.h"
// #include "adjnull.h"
// #include "kiss_fftr.h"
// #include "_kiss_fft_guts.h"

// #include "wave_komplex.h"
// #include "wave_alloc.h"
// 
// #ifndef KISS_FFT_H
// #include "wave_kissfft.h"
// #endif

void np_freqfilt_init(int nfft1 /* time samples (possibly padded) */, 
		      int nw1   /* frequency samples */);
/*< Initialize >*/


void np_freqfilt_set(float *filt /* frequency filter [nw] */);
/*< Initialize filter (zero-phase) >*/


/* #ifndef __cplusplus */
/*^*/

void np_freqfilt_cset(kiss_fft_cpx *filt /* frequency filter [nw] */);
/*< Initialize filter >*/


void np_freqfilt_close(void); 
/*< Free allocated storage >*/


void np_freqfilt(int nx, float* x);
/*< Filtering in place >*/


void np_freqfilt_lop (bool adj, bool add, int nx, int ny, float* x, float* y);
/*< Filtering as linear operator >*/


#endif

