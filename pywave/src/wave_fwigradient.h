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

#endif