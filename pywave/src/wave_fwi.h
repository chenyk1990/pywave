
#ifndef _fwi_h
#define _fwi_h

#include "wave_fwiutil.h"

void lstri(float ***data, float ***mwt, float ****src, np_acqui acpar, np_vec array, np_pas paspar, bool verb);
/*< passive source inversion >*/

#endif