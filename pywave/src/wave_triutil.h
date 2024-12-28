/* Time reversal imaging of passive seismic data linear operator */
#ifndef _triutil_h

#include <stdio.h>

#define NOP 4 /* derivative operator half-size */
#define C0 -205.0f/72.0f
#define C1 +8.0f/5.0f
#define C2 -1.0f/5.0f
#define C3 +8.0f/315.0f
#define C4 -1.0f/560.0f
#define Lap(a,ix,iz,sx,sz,v)  ( ( C4*(a[ix+4][iz  ] + a[ix-4][iz  ]) +      \
                                  C3*(a[ix+3][iz  ] + a[ix-3][iz  ]) +      \
                                  C2*(a[ix+2][iz  ] + a[ix-2][iz  ]) +      \
                                  C1*(a[ix+1][iz  ] + a[ix-1][iz  ]) +      \
                                  C0*(a[ix  ][iz  ]) )*sx            +      \
                                ( C4*(a[ix  ][iz+4] + a[ix  ][iz-4]) +      \
                                  C3*(a[ix  ][iz+3] + a[ix  ][iz-3]) +      \
                                  C2*(a[ix  ][iz+2] + a[ix  ][iz-2]) +      \
                                  C1*(a[ix  ][iz+1] + a[ix  ][iz-1]) +      \
                                  C0*(a[ix  ][iz  ]) )*sz )*v[ix][iz]
#define LapT(a,ix,iz,sx,sz,v) ( ( C4*(a[ix+4][iz  ]*v[ix+4][iz  ] + a[ix-4][iz  ]*v[ix-4][iz  ]) +      \
                                  C3*(a[ix+3][iz  ]*v[ix+3][iz  ] + a[ix-3][iz  ]*v[ix-3][iz  ]) +      \
                                  C2*(a[ix+2][iz  ]*v[ix+2][iz  ] + a[ix-2][iz  ]*v[ix-2][iz  ]) +      \
                                  C1*(a[ix+1][iz  ]*v[ix+1][iz  ] + a[ix-1][iz  ]*v[ix-1][iz  ]) +      \
                                  C0*(a[ix  ][iz  ]*v[ix  ][iz  ]) )*sx                          +      \
                                ( C4*(a[ix  ][iz+4]*v[ix  ][iz+4] + a[ix  ][iz-4]*v[ix  ][iz-4]) +      \
                                  C3*(a[ix  ][iz+3]*v[ix  ][iz+3] + a[ix  ][iz-3]*v[ix  ][iz-3]) +      \
                                  C2*(a[ix  ][iz+2]*v[ix  ][iz+2] + a[ix  ][iz-2]*v[ix  ][iz-2]) +      \
                                  C1*(a[ix  ][iz+1]*v[ix  ][iz+1] + a[ix  ][iz-1]*v[ix  ][iz-1]) +      \
                                  C0*(a[ix  ][iz  ]*v[ix  ][iz  ]) )*sz )
/*^*/

typedef struct tri2 *tri2d;
/*^*/

struct tri2{
    bool verb,abc;
    int  nt, nx, nz, nb, depth, nxpad, nzpad, nzxpad;
    float dt2, idz2, idx2, cb;
};
/*^*/



/***************************************************************/
tri2d tri2d_make(bool verb, bool abc,
                int nt, int nx, int nz, int nb, int depth,
                float dt, float dx, float dz, float cb);
/*< initialize tri2d utilities >*/

/***************************************************************/
// static tri2d tr;
// static float **vvpad, **u0, **u1, **u2, **tmp;

void timerev_init(bool verb, bool abc,
                  int nt, int nx, int nz, int nb, int depth,
                  float dt, float dx, float dz, float cb,
                  float **vv);
/*< initialize >*/

void timerev_lop(bool adj, bool add, int nm, int nd, float *mod, float *dat);
/*< time reversal imaging linear operator >*/

void timerev_close();
/*< finalize >*/

/***************************************************************/
void ctimerev(int ngrp, float ***ww, float **dd);
/*< correlative time reversal imaging condition >*/

/***************************************************************/
/* absorbing boundary */
// static float *decay=NULL;

void abc_init(tri2d tri);
/*< initialization >*/
   

void abc_close(void);
/*< free memory allocation>*/

void abc_apply(float *a /*2-D matrix*/,
               tri2d tri);
/*< boundary decay>*/

void abc_cal(int nb  /* absorbing layer length*/, 
             float c /* decaying parameter*/,
             float* w /* output weight[nb] */);
/*< calculate absorbing coefficients >*/

/***************************************************************/
/* sliding window normalization */
void threshold(bool step, int n, float hard, float *dat);
/*< in-place hard thresholding >*/

void absval(int n, float *dat);
/*< in-place absolute value >*/

void autopow(int n, float p, float *dat);
/*< in-place auto-correlation with abs >*/

float maxval(int n, float *dat);
/*< maximum absolute value and variance (optional) >*/

void scale(float a, int n, float *dat);
/*< scale an array >*/

void swnorm(bool verb, bool sw, int nz, int nx, int nt, int size, float perc, float *dat);
/*< local (sliding-window) normalization >*/

/*********************************************************/
/* smoothing */
void smooth(int n1, int n2, int n3,
            int rect1, int rect2, int rect3,
            int nrep,
	    float *dat);
/*< Generate reflectivity map with smoothing >*/

#endif