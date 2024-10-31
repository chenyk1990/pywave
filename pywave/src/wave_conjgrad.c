#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "npre_alloc.h"
#include "npre_conjgrad.h"
#include "npre_komplex.h"

#ifndef KISS_FFT_H
#include "npre_kissfft.h"
#endif

static int np, nx, nr, nd;
static np_complex *r, *d, *sp, *sx, *sr, *gp, *gx, *gr;
static float eps, tol;
static bool verb, hasp0;

void np_adjnull (bool adj /* adjoint flag */, 
		 bool add /* addition flag */, 
		 int nx   /* size of x */, 
		 int ny   /* size of y */, 
		 float* x, 
		 float* y) 
/*< Zeros out the output (unless add is true). 
  Useful first step for any linear operator. >*/
{
    int i;
    
    if(add) return;
    
    if(adj) {
	for (i = 0; i < nx; i++) {
	    x[i] = 0.;
	}
    } else {
	for (i = 0; i < ny; i++) {
	    y[i] = 0.;
	}
    }
}

void np_cadjnull (bool adj /* adjoint flag */, 
		  bool add /* addition flag */, 
		  int nx   /* size of x */, 
		  int ny   /* size of y */, 
		  np_complex* x, 
		  np_complex* y) 
/*< adjnull version for complex data. >*/
{
    int i;
    
    if(add) return;
    
    if(adj) {
	for (i = 0; i < nx; i++) {
	    x[i] = np_cmplx(0.0,0.0);
	}
    } else {
	for (i = 0; i < ny; i++) {
	    y[i] = np_cmplx(0.0,0.0);
	}
    }
}


static double norm (int n, const np_complex* x) 
/* double-precision L2 norm of a complex number */
{
    double prod, xi, yi;
    int i;

    prod = 0.;
    for (i = 0; i < n; i++) {
	xi = (double) crealf(x[i]);
	yi = (double) cimagf(x[i]);
	prod += xi*xi + yi*yi;
    }
    return prod;
}

void np_cconjgrad_init(int np1     /* preconditioned size */, 
		       int nx1     /* model size */, 
		       int nd1     /* data size */, 
		       int nr1     /* residual size */, 
		       float eps1  /* scaling */,
		       float tol1  /* tolerance */, 
		       bool verb1  /* verbosity flag */, 
		       bool hasp01 /* if has initial model */) 
/*< solver constructor >*/
{
    np = np1; 
    nx = nx1;
    nr = nr1;
    nd = nd1;
    eps = eps1*eps1;
    tol = tol1;
    verb = verb1;
    hasp0 = hasp01;

    r = np_complexalloc(nr);  
    d = np_complexalloc(nd); 
    sp = np_complexalloc(np);
    gp = np_complexalloc(np);
    sx = np_complexalloc(nx);
    gx = np_complexalloc(nx);
    sr = np_complexalloc(nr);
    gr = np_complexalloc(nr);
}

void np_cconjgrad_close(void) 
/*< Free allocated space >*/
{
    free (r);
    free (d);
    free (sp);
    free (gp);
    free (sx);
    free (gx);
    free (sr);
    free (gr);
}

void np_cconjgrad(np_coperator prec     /* data preconditioning */, 
		  np_coperator oper     /* linear operator */, 
		  np_coperator shape    /* shaping operator */, 
		  np_complex* p         /* preconditioned model */, 
		  np_complex* x         /* estimated model */, 
		  const np_complex* dat /* data */, 
		  int niter             /* number of iterations */)
/*< Conjugate gradient solver with shaping >*/
{
    double gn, gnp, alpha, beta, g0, dg, r0, b0;
    int i, iter;
    
    if (NULL != prec) {
	for (i=0; i < nd; i++) {
	    d[i] = np_cneg(dat[i]);
	}
	prec(false,false,nd,nr,d,r);
    } else {
	for (i=0; i < nr; i++) {
	    r[i] = np_cneg(dat[i]);
	}
    }
    
    if (hasp0) { /* initial p */
	shape(false,false,np,nx,p,x);
	if (NULL != prec) {
	    oper(false,false,nx,nd,x,d);
	    prec(false,true,nd,nr,d,r);
	} else {
	    oper(false,true,nx,nr,x,r);
	}
    } else {
	for (i=0; i < np; i++) {
	    p[i] = np_cmplx(0.0,0.0);
	}
	for (i=0; i < nx; i++) {
	    x[i] = np_cmplx(0.0,0.0);
	}
    } 
    
    dg = g0 = b0 = gnp = 0.;
    r0 = verb? norm(nr,r): 0.;

    for (iter=0; iter < niter; iter++) {
	for (i=0; i < np; i++) {
	    gp[i] = np_crmul(p[i],eps);
	}
	for (i=0; i < nx; i++) {
	    gx[i] = np_crmul(x[i],-eps);
	}

	if (NULL != prec) {
	    prec(true,false,nd,nr,d,r);
	    oper(true,true,nx,nd,gx,d);
	} else {
	    oper(true,true,nx,nr,gx,r);
	}

	shape(true,true,np,nx,gp,gx);
	shape(false,false,np,nx,gp,gx);

	if (NULL != prec) {
	    oper(false,false,nx,nd,gx,d);
	    prec(false,false,nd,nr,d,gr);
	} else {
	    oper(false,false,nx,nr,gx,gr);
	}

	gn = norm(np,gp);

	if (iter==0) {
	    g0 = gn;
	    b0 = fabs(gn + eps*(norm(nr,gr)-norm(nx,gx)));

	    for (i=0; i < np; i++) {
		sp[i] = gp[i];
	    }
	    for (i=0; i < nx; i++) {
		sx[i] = gx[i];
	    }
	    for (i=0; i < nr; i++) {
		sr[i] = gr[i];
	    }
	} else {
	    alpha = gn / gnp;
	    dg = gn / g0;

	    if (alpha < tol || dg < tol) {
		if (verb) 
		    printf(
			"convergence in %d iterations, alpha=%g, gd=%g\n",
			iter,alpha,dg);
		break;
	    }

	    for (i=0; i < np; i++) {
		sp[i] = np_cadd(gp[i],np_crmul(sp[i],alpha));
	    }
	    for (i=0; i < nx; i++) {
		sx[i] = np_cadd(gx[i],np_crmul(sx[i],alpha));
	    }
	    for (i=0; i < nr; i++) {
		sr[i] = np_cadd(gr[i],np_crmul(sr[i],alpha));
	    }
	}

	beta = norm(nr,sr) + eps*(norm(np,sp) - norm(nx,sx));

	/*
	if (beta/b0 < tol) {
	    if (verb) 
		printf("convergence in %d iterations, beta=%g",iter,beta);
	    break;
	}
	*/
	
	if (verb) printf("iteration %d res: %f grad: %f\n",
			     iter,norm(nr,r)/r0,dg);

	alpha = - gn / beta;

	for (i=0; i < np; i++) {
	    p[i] = np_cadd(p[i],np_crmul(sp[i],alpha));
	}

	for (i=0; i < nx; i++) {
	    x[i] = np_cadd(x[i],np_crmul(sx[i],alpha));
	}

	for (i=0; i < nr; i++) {
	    r[i] = np_cadd(r[i],np_crmul(sr[i],alpha));
	}

	gnp = gn;
    }
}
