/*below is the including part*/
// #include <math.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include "wave_alloc.h"
// #include "wave_ricker.h"
// #include "wave_abs.h"

#ifndef _psp_h
#define _psp_h

/** Part IV: pseudo-spectral wave extrapolation ********/
typedef struct Psmpar {
  /*survey parameters*/
  int   nx, ny, nz;
  float dx, dy, dz;
  int   ns;
  int   *spx, *spy, *spz;
  int   gpz, gpx, gpy, gplx, gply;
  int   gpz_v, gpx_v, gpl_v;
  int   jsnap;
  /*fft related*/
  bool  cmplx;
  int   pad1;
  /*absorbing boundary*/
  bool abc;
  int nbt, nbb, nblx, nbrx, nbly, nbry;
  float ct,cb,clx,crx,cly,cry;
  /*source parameters*/
  int src; /*source type*/
  int nt;
  float dt,*f0,*t0,*A;
  /*misc*/
  bool verb, ps;
  float vref;
} * psmpar; /*psm parameters*/
/*^*/

int psm(float **wvfld, float ***dat, float **dat_v, float *img, float *vel, psmpar par, bool tri);
/*< pseudo-spectral method >*/

int psm2d(float **wvfld, float **dat, float **dat_v, float *img, float *vel, pspar par, bool tri);
/*< pseudo-spectral method (2D) >*/

#endif