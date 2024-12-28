#ifndef cgstep_h
#define cgstep_h


#include <stdbool.h>


void np_cgstep( bool forget     /* restart flag */, 
		int nx, int ny  /* model size, data size */, 
		float* x        /* current model [nx] */, 
		const float* g  /* gradient [nx] */, 
		float* rr       /* data residual [ny] */, 
		const float* gg /* conjugate gradient [ny] */);
/*< Step of conjugate-gradient iteration. >*/


void np_cgstep_close (void);
/*< Free allocated space. >*/

#endif
