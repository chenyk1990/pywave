
#ifndef DTYPE_H
#define DTYPE_H

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

typedef struct {
    double r, i;
} np_double_complex;

typedef kiss_fft_cpx np_complex;


#endif



