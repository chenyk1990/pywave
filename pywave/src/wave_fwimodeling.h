
#ifndef _fwimodeling_h
#define _fwimodeling_h

#include "wave_fwiutil.h"

void forward_modeling_a(float ***data, np_sou soupar, np_acqui acpar, np_vec array, bool verb);
/*< acoustic forward modeling >*/

void forward_modeling(float ***data, np_sou soupar, np_acqui acpar, np_vec array, bool verb);
/*< visco-acoustic forward modeling >*/

#endif