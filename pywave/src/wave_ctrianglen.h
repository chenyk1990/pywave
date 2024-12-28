#ifndef ctrianglen_h
#define ctrianglen_h

#include "wave_dtype.h"

void np_ctrianglen_init (int ndim  /* number of dimensions */, 
			int *nbox /* triangle radius [ndim] */, 
			int *ndat /* data dimensions [ndim] */);
/*< initialize >*/


void np_ctrianglen_lop (bool adj, bool add, int nx, int ny, np_complex* x, np_complex* y);
/*< linear operator >*/


void np_ctrianglen_close(void);
/*< free allocated storage >*/

#endif

