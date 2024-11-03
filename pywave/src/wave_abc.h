/*below is the including part*/
// #include <math.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include "wave_alloc.h"

#ifndef _abs_h
#define _abs_h

/** Part II: Absorbing boundary condition ********/
/*Note: more powerful and efficient ABC can be incorporated*/
// static int nx, ny, nz, nx2, ny2, nz2, nbt, nbb, nblx, nbrx, nbly, nbry;
// static float ct, cb, clx, crx, cly, cry;
// static float *wt, *wb, *wlx, *wrx, *wly, *wry;

void vel_expand(float *vel, 				/* input velocity */
				float *vel2,				/* output velocity */
				int nz,   int nx,   int ny, /* size of input velocity */
				int nbt,  int nbb, 			/* ABC size in z  */
				int nblx, int nbrx,			/* ABC size in x  */
				int nbly, int nbry			/* ABC size in y  */);
/*< expand velocity model for ABC, revised on June 2022 YC>*/

void abc_cal(int abc /* decaying type*/,
             int nb  /* absorbing layer length*/, 
             float c /* decaying parameter*/,
             float* w /* output weight[nb] */);
/*< find absorbing coefficients >*/

void abc_init(int n1,  int n2, int n3,    /*model size*/
	      int n12, int n22, int n32,   /*padded model size*/
	      int nb1, int nb2,    /*top, bottom*/
	      int nb3, int nb4,   /*left x, right x*/
	      int nb5, int nb6,   /*left y, right y*/
	      float c1, float c2, /*top, bottom*/
	      float c3, float c4, /*left x, right x*/
	      float c5, float c6 /*left y, right y*/);
/*< initialization >*/
   

void abc_close(void);
/*< free memory allocation>*/

void abc_apply(float *a /*2-D matrix*/);
/*< boundary decay>*/

#endif