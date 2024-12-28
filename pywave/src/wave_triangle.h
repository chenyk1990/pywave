#ifndef triangle_h
#define triangle_h


#include<stdbool.h>

typedef struct np_Triangle *np_triangle;
/* abstract data type */

struct np_Triangle {
    float *tmp, wt;
    int np, nb, nx;
    bool box;
};

np_triangle np_triangle_init (int  nbox /* triangle length */, 
			      int  ndat /* data length */,
                              bool box  /* if box instead of triangle */);
/*< initialize >*/


void np_smooth (np_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */);
/*< apply triangle smoothing >*/


void np_dsmooth (np_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */);
/*< apply triangle smoothing >*/


void np_smooth2 (np_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */);
/*< apply adjoint triangle smoothing >*/


void np_dsmooth2 (np_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */);
/*< apply adjoint triangle smoothing >*/


void  np_triangle_close(np_triangle tr);
/*< free allocated storage >*/

#endif
