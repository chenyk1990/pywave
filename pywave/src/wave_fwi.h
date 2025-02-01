
#ifndef _fwi_h
#define _fwi_h

#include "wave_fwiutil.h"

void lstri(float ***data, float ***mwt, float ****src, np_acqui acpar, np_vec array, np_pas paspar, bool verb);
/*< passive source inversion >*/

void pfwi(float ***data, float **vinv, float *grad, float ***mwt, float ****src, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_optim optpar, np_pas paspar, bool verb);
/*< passive fwi >*/

void fwi(float ***data, float **vinv, float *grad, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_optim optpar, bool verb, int media);
/*< fwi >*/

#endif