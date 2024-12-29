/*below is the including part*/
// #include <math.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include "wave_alloc.h"
// #include "wave_ricker.h"
// #include "wave_abs.h"

#ifndef _fwigradient_h
#define _fwigradient_h

#include <stdio.h>
#include "wave_fwiutil.h"

void lstri_op(float **dd, float **dwt, float ***ww, float ***mwt, np_acqui acpar, np_vec array, np_pas paspar, bool verb);
/*< ls TRI operator >*/

void gradient_pas_init(float ***data, float ****src, float ***mwt, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_pas paspar, bool verb1);
/*< initialize >*/

void gradient_pas_av(float *x, float *fcost, float *grad);
/*< acoustic velocity gradient >*/

#endif