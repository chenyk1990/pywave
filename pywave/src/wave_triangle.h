#ifndef triangle_h
#define triangle_h


// #include "_bool.h"
// #include "komplex.h"

#include "npre_dtype.h"


struct np_Ctriangle {
    np_complex *tmp;
    float wt;
    int np, nb, nx;
    bool box;
};

typedef struct np_Ctriangle *np_ctriangle;
/* abstract data type */
/*^*/


np_ctriangle np_ctriangle_init (int nbox /* triangle length */, 
				int ndat /* data length */,
				bool box /* if box instead of triangle */);
/*< initialize >*/


void np_csmooth (np_ctriangle tr    /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 np_complex *x   /* data (smoothed in place) */);
/*< apply adjoint triangle smoothing >*/


void  np_ctriangle_close(np_ctriangle tr);
/*< free allocated storage >*/

#endif


