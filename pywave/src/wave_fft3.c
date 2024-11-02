/** Part III: Fourier transform ********/
/*below is the including part*/
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "wave_alloc.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

#include "wave_komplex.h"

#define SF_PI (3.14159265358979323846264338328)
/*above is the including part*/

static bool cmplx;
static int n1, n2, n3, nk;
static float wwt;

static float ***ff=NULL;
static np_complex ***cc=NULL;

static kiss_fftr_cfg cfg, icfg;
static kiss_fft_cfg cfg1, icfg1, cfg2, icfg2, cfg3, icfg3;
static kiss_fft_cpx ***tmp, *ctrace2, *ctrace3;
static np_complex *trace2, *trace3;

int fft3_init(bool cmplx1        /* if complex transform */,
	      int pad1           /* padding on the first axis */,
	      int nx,   int ny,   int nz   /* axis 1,2,3; input data size */, 
	      int *nx2, int *ny2, int *nz2 /* axis 1,2,3; padded data size */)
/*< initialize >*/
{
    int i2, i3;
    
    cmplx = cmplx1;

    /* axis 1 */
    if (cmplx) {
	nk = n1 = kiss_fft_next_fast_size(nx*pad1);
	cfg1  = kiss_fft_alloc(n1,0,NULL,NULL);
	icfg1 = kiss_fft_alloc(n1,1,NULL,NULL);
    } else {
	nk = kiss_fft_next_fast_size(pad1*(nx+1)/2)+1;
	n1 = 2*(nk-1);
	cfg  = kiss_fftr_alloc(n1,0,NULL,NULL);
	icfg = kiss_fftr_alloc(n1,1,NULL,NULL);
    }

    /* axis 2 */
    n2 = kiss_fft_next_fast_size(ny);
    cfg2  = kiss_fft_alloc(n2,0,NULL,NULL);
    icfg2 = kiss_fft_alloc(n2,1,NULL,NULL);

    trace2 = np_complexalloc(n2);
    ctrace2 = (kiss_fft_cpx *) trace2;

    /* axis 3 */
    n3 = kiss_fft_next_fast_size(nz);
    cfg3  = kiss_fft_alloc(n3,0,NULL,NULL);
    icfg3 = kiss_fft_alloc(n3,1,NULL,NULL);

    trace3 = np_complexalloc(n3);
    ctrace3 = (kiss_fft_cpx *) trace3;

    /* --- */

    tmp = (kiss_fft_cpx***) np_alloc (n3,sizeof(kiss_fft_cpx**));
    tmp[0] = (kiss_fft_cpx**) np_alloc (n2*n3,sizeof(kiss_fft_cpx*));
    tmp[0][0] = (kiss_fft_cpx*) np_alloc (nk*n2*n3,sizeof(kiss_fft_cpx));

    for (i2=1; i2 < n2*n3; i2++) {
	tmp[0][i2] = tmp[0][0]+i2*nk;
    }

    for (i3=1; i3 < n3; i3++) {
	tmp[i3] = tmp[0]+i3*n2;
    }


    if (cmplx) {
	cc = np_complexalloc3(n1,n2,n3);
    } else {
	ff = np_floatalloc3(n1,n2,n3);
    }

    *nx2 = n1;
    *ny2 = n2;
    *nz2 = n3;

    wwt =  1.0/(n3*n2*n1);

    return (nk*n2*n3);
}

void fft3(float *inp      /* [n1*n2*n3] */, 
	  np_complex *out /* [nk*n2*n3] */)
/*< 3-D FFT >*/
{
    int i1, i2, i3;
    float f; 
    
    /* FFT centering */    
    for (i3=0; i3<n3; i3++) {
	for (i2=0; i2<n2; i2++) {
	    for (i1=0; i1<n1; i1++) {
		f = inp[(i3*n2+i2)*n1+i1];
		if (cmplx) {
		    cc[i3][i2][i1] = np_cmplx((((i3%2==0)==(i2%2==0))==(i1%2==0))? f:-f,0.);
		} else {
		    ff[i3][i2][i1] = ((i3%2==0)==(i2%2==0))? f:-f;
		}
	    }
	}
    }

    /* FFT over first axis */
    for (i3=0; i3 < n3; i3++) {
	for (i2=0; i2 < n2; i2++) {
	    if (cmplx) {
		kiss_fft_stride(cfg1,(kiss_fft_cpx *) cc[i3][i2],tmp[i3][i2],1);
	    } else {
		kiss_fftr (cfg,ff[i3][i2],tmp[i3][i2]);
	    }
	}
    }

    /* FFT over second axis */
    for (i3=0; i3 < n3; i3++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(cfg2,tmp[i3][0]+i1,ctrace2,nk);
	    for (i2=0; i2 < n2; i2++) {
		tmp[i3][i2][i1]=ctrace2[i2];
	    }
	}
    }

    /* FFT over third axis */
    for (i2=0; i2 < n2; i2++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(cfg3,tmp[0][0]+i2*nk+i1,ctrace3,nk*n2);
	    for (i3=0; i3<n3; i3++) {
		out[(i3*n2+i2)*nk+i1] = trace3[i3];
	    }
	}
    } 
}

void ifft3(float *out      /* [n1*n2*n3] */, 
	   np_complex *inp /* [nk*n2*n3] */)
/*< 3-D inverse FFT >*/
{
    int i1, i2, i3;

    /* IFFT over third axis */
    for (i2=0; i2 < n2; i2++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(icfg3,(kiss_fft_cpx *) (inp+i2*nk+i1),ctrace3,nk*n2);
	    for (i3=0; i3<n3; i3++) {
		tmp[i3][i2][i1] = ctrace3[i3];
	    }
	}
    }
    
    /* IFFT over second axis */
    for (i3=0; i3 < n3; i3++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(icfg2,tmp[i3][0]+i1,ctrace2,nk);		
	    for (i2=0; i2<n2; i2++) {
		tmp[i3][i2][i1] = ctrace2[i2];
	    }
	}
    }

    /* IFFT over first axis */
    for (i3=0; i3 < n3; i3++) {
	for (i2=0; i2 < n2; i2++) {
	    if (cmplx) {
		kiss_fft_stride(icfg1,tmp[i3][i2],(kiss_fft_cpx *) cc[i3][i2],1);		
	    } else {
		kiss_fftri(icfg,tmp[i3][i2],ff[i3][i2]);
	    }
	}
    }

    /* FFT centering and normalization */
    for (i3=0; i3<n3; i3++) {
	for (i2=0; i2<n2; i2++) {
	    for (i1=0; i1<n1; i1++) {
		if (cmplx) {
		    out[(i3*n2+i2)*n1+i1] = ((((i3%2==0)==(i2%2==0))==(i1%2==0))? wwt:-wwt)*crealf(cc[i3][i2][i1]);
		} else {
		    out[(i3*n2+i2)*n1+i1] = (((i3%2==0)==(i2%2==0))? wwt: - wwt)*ff[i3][i2][i1];
		}
	    }
	}
    }
}
