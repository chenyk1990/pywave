#ifndef ntriangle_h
#define ntriangle_h


#include "wave_dtype.h"

struct CNtriangle {
    np_complex *tmp;
    int np, nb, nx;
};

typedef struct CNtriangle *cntriangle;
/* abstract data type */

cntriangle cntriangle_init (int nbox /* maximum triangle length */, 
			  int ndat /* data length */);
/*< initialize >*/


void cnsmooth (cntriangle tr /* smoothing object */, 
	      int o, int d /* sampling. o: starting index, d: stride in samples for 1/2/3rd dimension; all refer to a correct index in a 1D vector  */, 
	      bool der     /* derivative flag */, 
	      const float *t /* triangle lengths */, 
	      const int *s /* triangle shifts */,
	      np_complex *x     /* data (smoothed in place) */);
/*< smooth >*/


void cnsmooth2 (cntriangle tr /* smoothing object */, 
	       int o, int d /* sampling */, 
	       bool der     /* derivative flag */, 
	       const float *t /* triangle lengths */,
	       const int *s /* triangle shifts */,
	       np_complex *x     /* data (smoothed in place) */);
/*< alternative smooth >*/


void  cntriangle_close(cntriangle tr);
/*< free allocated storage >*/

#endif
