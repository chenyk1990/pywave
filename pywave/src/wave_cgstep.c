/* Claerbout's conjugate-gradient iteration. */

#include <stdlib.h>
#include <stdbool.h>
#include "wave_cgstep.h"
#include "wave_alloc.h"
#include "wave_blas.h"

/* precision */
static const float EPSILON=1.e-12;

static float* S;  /* model step */
static float* Ss; /* residual step */
static bool Allocated = false; /* if S and Ss are allocated */

void np_cgstep( bool forget     /* restart flag */, 
		int nx, int ny  /* model size, data size */, 
		float* x        /* current model [nx] */, 
		const float* g  /* gradient [nx] */, 
		float* rr       /* data residual [ny] */, 
		const float* gg /* conjugate gradient [ny] */) 
/*< Step of conjugate-gradient iteration. >*/
{
    double sds, gdg, gds, determ, gdr, sdr, alfa, beta;
    int i;
    if (!Allocated) {
	Allocated = forget = true;
	S  = np_floatalloc (nx);
	Ss = np_floatalloc (ny);
    }
    if (forget) {
	for (i = 0; i < nx; i++) S[i] = 0.;
	for (i = 0; i < ny; i++) Ss[i] = 0.;
	beta = 0.0;
	alfa = cblas_dsdot( ny, gg, 1, gg, 1);
	/* Solve G . ( R + G*alfa) = 0 */
	if (alfa <= 0.) return;
	alfa = - cblas_dsdot( ny, gg, 1, rr, 1) / alfa;
    } else {
	/* search plane by solving 2-by-2
	   G . (R + G*alfa + S*beta) = 0
	   S . (R + G*alfa + S*beta) = 0 */
	gdg = cblas_dsdot( ny, gg, 1, gg, 1);       
	sds = cblas_dsdot( ny, Ss, 1, Ss, 1);       
	gds = cblas_dsdot( ny, gg, 1, Ss, 1);       
	if (gdg == 0. || sds == 0.) return;
	determ = 1.0 - (gds/gdg)*(gds/sds);
	if (determ > EPSILON) determ *= gdg * sds;
	else determ = gdg * sds * EPSILON;
	gdr = - cblas_dsdot( ny, gg, 1, rr, 1);
	sdr = - cblas_dsdot( ny, Ss, 1, rr, 1);
	alfa = ( sds * gdr - gds * sdr ) / determ;
	beta = (-gds * gdr + gdg * sdr ) / determ;
    }
    cblas_sscal(nx,beta,S,1);
    cblas_saxpy(nx,alfa,g,1,S,1);

    cblas_sscal(ny,beta,Ss,1);
    cblas_saxpy(ny,alfa,gg,1,Ss,1);

    for (i = 0; i < nx; i++) {
	x[i] +=  S[i];
    }
    for (i = 0; i < ny; i++) {
	rr[i] += Ss[i];
    }
}

void np_cgstep_close (void) 
/*< Free allocated space. >*/ 
{
    if (Allocated) {
	free (S);
	free (Ss);
	Allocated = false;
    }
}

