/*below is the including part*/
// #include <math.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include "wave_alloc.h"
// #include "wave_ricker.h"
// #include "wave_abs.h"

#ifndef _fwiutil_h
#define _fwiutil_h

typedef struct np_source{
	float fhi;
	float flo;
	int rectx;
	int rectz;
} *np_sou;
/*^*/

typedef struct np_acquisition{
	// model dimension
	int nx;
	int nz;
	float dx;
	float dz;
	float x0;
	float z0;
	// wavelet dimension
	int nt;
	float dt;
	float t0;
	// absorbing boundary condition
	int nb;
	float coef;
	float *bc;
	// padding
	int padnx;
	int padnz;
	float padx0;
	float padz0;
	// acquisition type
	int acqui_type;
	// shot
	int ns;
	float ds;
	float s0;
	int sz;
	int ds_v;
	int s0_v;
	// receiver
	int nr;
	float dr;
	float r0;
	int rz;
	int dr_v;
	int *r0_v;
	int *r02;
	int *nr2;
	// reference frequency
	float f0;
	// wavefield storing interval
	int interval;
} *np_acqui;
/*^*/

typedef struct np_1darray{
	float *vv;
	float *qq;
	float *tau;
	float *taus;
	float *ww;
} *np_vec;
/*^*/

typedef struct np_fwipar{
	bool onlygrad;
	int grad_type;
	int misfit_type;
	int opt_type;
	// data residual weighting
	float wt1;
	float wt2;
	float woff1;
	float woff2;
        bool oreo;
	// water layer depth
	int waterz;
	int waterzb;
	// gradient smoothing parameters
	int rectx;
	int rectz;
} *np_fwi;
/*^*/

typedef struct np_optimization {
	int niter;
	float conv_error;
	int npair;
	int nls;
	int igrad;
	int ipair;
	int ils;
        int repeat;
	float c1;
	float c2;
	float factor;
	float alpha;
	float f0;
	float fk;
	float gk_norm;
	float **sk, **yk;
        /* bound constraints */
        float v1;
        float v2;
} *np_optim;
/*^*/

typedef struct np_passive{
    bool inv;
    bool onlysrc;
    bool onlyvel;
    bool sw;
    bool ctr;
    bool prec;
    bool hidesrc;
    int niter;
    int ngrp;
    int size;
    int rectz;
    int rectx;
    int rectt;
    int repeat;
    float perc;
    float hard;
} *np_pas;
/*^*/

void preparation(float *vv, float *qq, float *ww, np_acqui acpar, np_sou soupar, np_vec array);
/*< read data, initialize variables and prepare acquisition geometry >*/

void pad2d(float *vec, float **array, int nz, int nx, int nb);
/*< convert a vector to an array >*/

void source_map(int sx, int sz, int rectx, int rectz, int padnx, int padnz, int padnzx, float *rr);
/*< generate source map >*/

void laplace(float **p1, float **term, int padnx, int padnz, float dx2, float dz2);
/*< laplace operator >*/

void apply_sponge(float **p, float *bc, int padnx, int padnz, int nb);
/*< apply absorbing boundary condition >*/

void residual_weighting(float **ww, int nt, int nr, int wtn1, int wtn2, int woffn1, int woffn2, bool oreo);
/*< data residual weighting >*/

void gradient_smooth2(int rectx, int rectz, int nx, int nz, int waterz, float scaling, float *grad);
/*< smooth gradient, zero bathymetry layer and normalization >*/

void gradient_smooth2b(int rectx, int rectz, int nx, int nz, int waterz, int waterzb, float scaling, float *grad);
/*< smooth gradient, zero bathymetry layer and normalization >*/

void l2norm(int n, float *a, float *norm);
/*< L2 norm of a vector >*/

void reverse(int n, float *a, float *b);
/*< reverse the sign of the vector >*/

void copy(int n, float *a, float *b);
/*< copy vector >*/

void dot_product(int n, float *a, float *b, float *product);
/*< dot product of two vectors >*/

#endif