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

#define np_PI (3.14159265358979323846264338328)
/*above is the including part*/

static bool cmplx;
static int n1, n2, nk;
static float wwt;

static float **ff=NULL;
static np_complex **cc=NULL,*dd=NULL;

static kiss_fftr_cfg cfg=NULL, icfg=NULL;
static kiss_fft_cfg cfg1=NULL, icfg1=NULL, cfg2=NULL, icfg2=NULL;
static kiss_fft_cpx **tmp=NULL, *ctrace2=NULL;
static np_complex *trace2=NULL;

int fft2_init(bool cmplx1        /* if complex transform */,
	      int pad1           /* padding on the first axis */,
	      int nx,   int ny   /* input data size */, 
	      int *nx2, int *ny2 /* padded data size */)
/*< initialize >*/
{

	int i2;
	
	cmplx = cmplx1;
	
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
		
    n2 = kiss_fft_next_fast_size(ny);

    if (cmplx) {
	cc = np_complexalloc2(n1,n2);
    } else {
	ff = np_floatalloc2(n1,n2);
    }
    dd = np_complexalloc(nk*n2);
	

    cfg2  = kiss_fft_alloc(n2,0,NULL,NULL);
    icfg2 = kiss_fft_alloc(n2,1,NULL,NULL);
 	
    tmp =    (kiss_fft_cpx **) np_alloc(n2,sizeof(*tmp));
    tmp[0] = (kiss_fft_cpx *)  np_alloc(nk*n2,sizeof(kiss_fft_cpx));
    for (i2=0; i2 < n2; i2++) {
	tmp[i2] = tmp[0]+i2*nk;
    }
	
    trace2 = np_complexalloc(n2);
    ctrace2 = (kiss_fft_cpx *) trace2;

    *nx2 = n1;
    *ny2 = n2;
	
    wwt =  1.0/(n1*n2);
	
    return (nk*n2);
}

void fft2(float *inp      /* [n1*n2] */, 
	  np_complex *out /* [nk*n2] */)
/*< 2-D FFT >*/
{
    int i1, i2;

    /* FFT centering */
    for (i2=0; i2<n2; i2++) {
	for (i1=0; i1<n1; i1++) {
	    if (cmplx) {
		cc[i2][i1] = np_cmplx(((i2%2==0)==(i1%2==0))? inp[i2*n1+i1]:-inp[i2*n1+i1],0.);
	    } else {
		ff[i2][i1] = (i2%2)? -inp[i2*n1+i1]:inp[i2*n1+i1];
	    }
	}
    }
    
    for (i2=0; i2 < n2; i2++) {
	if (cmplx) {
	    kiss_fft_stride(cfg1,(kiss_fft_cpx *) cc[i2],tmp[i2],1);
	} else {
	    kiss_fftr (cfg,ff[i2],tmp[i2]);
	}
    }
	
    for (i1=0; i1 < nk; i1++) {
	kiss_fft_stride(cfg2,tmp[0]+i1,ctrace2,nk);
	for (i2=0; i2<n2; i2++) {
	    out[i2*nk+i1] = trace2[i2];
	}
    }
}

void ifft2_allocate(np_complex *inp /* [nk*n2] */)
/*< allocate inverse transform >*/
{
  /* for backward compatibility */
}

void ifft2(float *out      /* [n1*n2] */, 
	   np_complex *inp /* [nk*n2] */)
/*< 2-D inverse FFT >*/
{
    int i1, i2;

    for (i1=0; i1 < nk; i1++) {
	kiss_fft_stride(icfg2,(kiss_fft_cpx *) (inp+i1),ctrace2,nk);
		
	for (i2=0; i2<n2; i2++) {
	    tmp[i2][i1] = ctrace2[i2];
	}
    }
    for (i2=0; i2 < n2; i2++) {
	if (cmplx) {
	    kiss_fft_stride(icfg1,tmp[i2],(kiss_fft_cpx *) cc[i2],1);
	} else {
	    kiss_fftri(icfg,tmp[i2],ff[i2]);
	}
    }
    
    /* FFT centering and normalization */
    for (i2=0; i2<n2; i2++) {
	for (i1=0; i1<n1; i1++) {
	    if (cmplx) {
		out[i2*n1+i1] = (((i2%2==0)==(i1%2==0))? wwt:-wwt) * crealf(cc[i2][i1]);
	    } else {
		out[i2*n1+i1] = (i2%2? -wwt: wwt)*ff[i2][i1];
	    }
	}
    }
}

void fft2_finalize()
/*< clean up fftw >*/
{
/* make sure everything is back to its pristine state */

    if (NULL != cfg) { free(cfg); cfg=NULL; }
    if (NULL != icfg) { free(icfg); icfg=NULL; }
    if (NULL != cfg1) { free(cfg1); cfg1=NULL; }
    if (NULL != icfg1) { free(icfg1); icfg1=NULL; }
    if (NULL != cfg2) { free(cfg2); cfg2=NULL; }
    if (NULL != icfg2) { free(icfg2); icfg2=NULL; }
    if (NULL != tmp) { free(*tmp); free(tmp); tmp=NULL; }
    if (NULL != trace2) { free(trace2); trace2=NULL; }
    if (cmplx) {
      if (NULL != cc) { free(*cc); free(cc); cc=NULL; }
    } else {
      if (NULL != ff) { free(*ff); free(ff); ff=NULL; }
    }
    if (NULL != dd) { free(dd); dd=NULL; }
}