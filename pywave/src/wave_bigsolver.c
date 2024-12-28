/* Solver functions for iterative least-squares optimization. */

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include "wave_bigsolver.h"
#include "wave_chain.h"
#include "wave_alloc.h"
// #include "error.h"
#include "wave_komplex.h"
#include "wave_blas.h"
#include "wave_blasc.h"

static const float TOLERANCE=1.e-12;

void np_solver_prec (np_operator oper   /* linear operator */, 
		     np_solverstep solv /* stepping function */, 
		     np_operator prec   /* preconditioning operator */, 
		     int nprec          /* size of p */, 
		     int nx             /* size of x */, 
		     int ny             /* size of dat */, 
		     float* x           /* estimated model */, 
		     const float* dat   /* data */, 
		     int niter          /* number of iterations */, 
		     double eps          /* regularization parameter */, 
		     ...                /* variable number of arguments */) 
/*< Generic preconditioned linear solver.
 ---
 Solves
 oper{x} =~ dat
 eps p   =~ 0
 where x = prec{p}
 ---
 The last parameter in the call to this function should be "end".
 Example: 
 ---
 np_solver_prec (oper_lop,np_cgstep,prec_lop,
 np,nx,ny,x,y,100,1.0,"x0",x0,"end");
 ---
 Parameters in ...:
 ... 
 "wt":     float*:         weight      
 "wght":   np_weight wght: weighting function
 "x0":     float*:         initial model
 "nloper": np_operator:    nonlinear operator  
 "mwt":    float*:         model weight
 "verb":   bool:           verbosity flag
 "known":  bool*:          known model mask
 "nmem":   int:            iteration memory
 "nfreq":  int:            periodic restart
 "xmov":   float**:        model iteration
 "rmov":   float**:        residual iteration
 "err":    float*:         final error
 "res":    float*:         final residual
 "xp":     float*:         preconditioned model
 >*/
{
    va_list args;
    char* par;
    float* wt = NULL;
    np_weight wght = NULL;
    float* x0 = NULL;
    np_operator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* xp = NULL;
    float* wht = NULL;
    float *p, *g, *rr, *gg, *tp = NULL, *td = NULL;
    int i, iter;
    double dprr, dppd, dppm, dpgm, dprr0=1., dpgm0=1.;
    bool forget = false;

    va_start (args, eps);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_operator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else if (0 == strcmp (par,"xp"))      
	{                    xp = va_arg (args, float*);}
	else 
	{ 
// 	np_error("%s: unknown parameter %s",__FILE__,par);
	}
    }
    va_end (args);
  
    p = np_floatalloc (ny+nprec);
    g = np_floatalloc (ny+nprec);
    rr = np_floatalloc (ny);
    gg = np_floatalloc (ny);
    for (i=0; i < ny; i++) {
	rr[i] = -dat[i];
	p[i+nprec] = 0.0;
    }

    if (wt != NULL || wght != NULL) {
	td = np_floatalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    if (mwt != NULL) tp = np_floatalloc (nprec);

    if (x0 != NULL) {
	for (i=0; i < nprec; i++) {
	    p[i] = x0[i]; 
	}
	if (nloper != NULL) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		np_chain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_chain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		np_chain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_chain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	}
    } else {
	for (i=0; i < nprec; i++) {
	    p[i] = 0.0; 
	}
    }

    for (iter = 0; iter < niter; iter++) {
	if (nmem >= 0) {
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = eps*p[i+nprec] + wht[i]*rr[i];
		td[i] = rr[i]*wht[i];
	    } 
	    np_chain (oper, prec, true, false, nprec, ny, nx, g, td, x); 
	} else {
	    np_chain (oper, prec, true, false, nprec, ny, nx, g, rr, x);
	}
	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		g[i] *= mwt[i];
	    }
	}
	for (i=0; i < ny; i++) {
	    g[i+nprec] = eps*rr[i];
	}
	if (known != NULL) {
	    for (i=0; i < nprec; i++) {
		if (known[i]) {
		    g[i] = 0.0;
		} 
	    }
	}

	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		tp[i] = g[i]*mwt[i];
	    }
	    np_chain (oper, prec, false, false, nprec, ny, nx, tp, gg, x);
	} else {
	    np_chain (oper, prec, false, false, nprec, ny, nx,  g, gg, x);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] *= wht[i];
	    }
	}
	cblas_saxpy(ny,eps,g+nprec,1,gg,1);

	if (forget && nfreq != 0) {  /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq);
	} 
	
	if (iter == 0) {
	    dprr0 = cblas_snrm2 (ny, rr, 1);
	    dpgm0 = cblas_snrm2 (nprec, g, 1);
	    dprr = 1.;
	    dpgm = 1.;
	} else {
	    dprr = cblas_snrm2 (ny, rr, 1)/dprr0;
	    dpgm = cblas_snrm2 (nprec, g, 1)/dpgm0;
	}
	dppd = cblas_snrm2 (ny, p+nprec, 1);
	dppm = cblas_snrm2 (nprec, p, 1);

	if (verb) 
	    printf("iteration %d res %g prec dat %g prec mod %g grad %g\n", 
		       iter, dprr, dppd, dppm, dpgm);
	
	if (dprr < TOLERANCE || dpgm < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }

	    break;
	}

	solv (forget, nprec+ny, ny, p, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = eps*p[i+nprec] - dat[i];
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		np_chain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_chain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i];
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		np_chain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_chain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }	
	} else if (xmov != NULL || iter == niter-1) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }
	}
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  p[i+nprec] * eps;
	    }
	}
	if (err != NULL) err[iter] = cblas_snrm2(ny, rr, 1);
    } /* iter */

    if (xp != NULL) {
	for (i=0; i < nprec; i++) {
	    xp[i] = p[i];
	}
    }
    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }

    for (; iter < niter; iter++) {
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  p[i+nprec] * eps;
	    }
	}    
	if (err != NULL) err[iter] = cblas_snrm2(ny, rr, 1);
    }  

    free (p);
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }

    if (mwt != NULL) {
	free (tp);
    }

}

void np_csolver_prec (np_coperator oper   /* linear operator */, 
		      np_csolverstep solv /* stepping function */, 
		      np_coperator prec   /* preconditioning operator */, 
		      int nprec          /* size of p */, 
		      int nx             /* size of x */, 
		      int ny             /* size of dat */, 
		      np_complex* x           /* estimated model */, 
		      const np_complex* dat   /* data */, 
		      int niter          /* number of iterations */, 
		      double eps          /* regularization parameter */, 
		     ...                /* variable number of arguments */) 
/*< Generic preconditioned linear solver.
 ---
 Solves
 oper{x} =~ dat
 eps p   =~ 0
 where x = prec{p}
 ---
 The last parameter in the call to this function should be "end".
 Example: 
 ---
 np_csolver_prec (oper_lop,np_cgstep,prec_lop,
 np,nx,ny,x,y,100,1.0,"x0",x0,"end");
 ---
 Parameters in ...:
 ... 
 "wt":     float*:         weight      
 "wght":   np_cweight wght: weighting function
 "x0":     np_complex*:         initial model
 "nloper": np_coperator:   nonlinear operator  
 "mwt":    float*:         model weight
 "verb":   bool:           verbosity flag
 "known":  bool*:          known model mask
 "nmem":   int:            iteration memory
 "nfreq":  int:            periodic restart
 "xmov":   np_complex**:        model iteration
 "rmov":   np_complex**:        residual iteration
 "err":    np_float*:         final error
 "res":    np_complex*:         final residual
 "xp":     np_complex*:         preconditioned model
 >*/
{
    va_list args;
    char* par;
    float* wt = NULL;
    np_cweight wght = NULL;
    np_complex* x0 = NULL;
    np_coperator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    np_complex** xmov = NULL;
    np_complex** rmov = NULL;
    float* err = NULL;
    np_complex* res = NULL;
    np_complex* xp = NULL;
    float* wht = NULL;
    np_complex *p, *g, *rr, *gg, *tp = NULL, *td = NULL;
    int i, iter;
    double dprr, dppd, dppm, dpgm, dprr0=1., dpgm0=1.;
    bool forget = false;

    va_start (args, eps);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_cweight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, np_complex*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_coperator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, np_complex**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, np_complex**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, np_complex*);}
	else if (0 == strcmp (par,"xp"))      
	{                    xp = va_arg (args, np_complex*);}
	else 
	{ 
// 	np_error("%s: unknown parameter %s",__FILE__,par);
	}
    }
    va_end (args);
  
    p = np_complexalloc (ny+nprec);
    g = np_complexalloc (ny+nprec);
    rr = np_complexalloc (ny);
    gg = np_complexalloc (ny);
    for (i=0; i < ny; i++) {
	rr[i] = np_cneg(dat[i]);

	p[i+nprec] = np_cmplx(0.0,0.0);
    }

    if (wt != NULL || wght != NULL) {
	td = np_complexalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    if (mwt != NULL) tp = np_complexalloc (nprec);

    if (x0 != NULL) {
	for (i=0; i < nprec; i++) {
	    p[i] = x0[i]; 
	}
	if (nloper != NULL) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		np_cchain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_cchain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		np_cchain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_cchain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	}
    } else {
	for (i=0; i < nprec; i++) {
	    p[i] = np_cmplx(0.0,0.0); 
	}
    }

    for (iter = 0; iter < niter; iter++) {
	if (nmem >= 0) {
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cadd(np_crmul(p[i+nprec],eps),
				np_crmul(rr[i],wht[i]));
		td[i] = np_crmul(rr[i],wht[i]);
	    } 
	    np_cchain (oper, prec, true, false, nprec, ny, nx, g, td, x); 
	} else {
	    np_cchain (oper, prec, true, false, nprec, ny, nx, g, rr, x);
	}
	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		g[i] = np_crmul(g[i],mwt[i]);
	    }
	}
	for (i=0; i < ny; i++) {
	    g[i+nprec] = np_crmul(rr[i],eps);
	}
	if (known != NULL) {
	    for (i=0; i < nprec; i++) {
		if (known[i]) {
		    g[i] = np_cmplx(0.0,0.0);
		} 
	    }
	}

	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		tp[i] = np_crmul(g[i],mwt[i]);
	    }
	    np_cchain (oper, prec, false, false, nprec, ny, nx, tp, gg, x);
	} else {
	    np_cchain (oper, prec, false, false, nprec, ny, nx,  g, gg, x);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] = np_crmul(gg[i],wht[i]);
	    }
	}
        /*	cblas_saxpy(ny,eps,g+nprec,1,gg,1); */

	for (i=0; i < ny; i++) {
	    gg[i] = np_cadd(gg[i],np_crmul(g[i+nprec],eps));
	}

	if (forget && nfreq != 0) {  /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq);
	} 
	
	if (iter == 0) {
	    dprr0 = cblas_scnrm2 (ny, rr, 1);
	    dpgm0 = cblas_scnrm2 (nprec, g, 1);
	    dprr = 1.;
	    dpgm = 1.;
	} else {
	    dprr = cblas_scnrm2 (ny, rr, 1)/dprr0;
	    dpgm = cblas_scnrm2 (nprec, g, 1)/dpgm0;
	}
	dppd = cblas_scnrm2 (ny, p+nprec, 1);
	dppm = cblas_scnrm2 (nprec, p, 1);

	if (verb) 
	    printf("iteration %d res %g prec dat %g prec mod %g grad %g\n", 
		       iter, dprr, dppd, dppm, dpgm);
	
	if (dprr < TOLERANCE || dpgm < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }

	    break;
	}

	solv (forget, nprec+ny, ny, p, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cadd(np_crmul(p[i+nprec],eps),np_cneg(dat[i]));
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		np_cchain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_cchain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cneg(dat[i]);
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		np_cchain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		np_cchain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }	
	} else if (xmov != NULL || iter == niter-1) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = np_crmul(p[i],mwt[i]);
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }
	}
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  np_crmul(p[i+nprec],eps);
	    }
	}
	if (err != NULL) err[iter] = cblas_scnrm2(ny, rr, 1);
    } /* iter */

    if (xp != NULL) {
	for (i=0; i < nprec; i++) {
	    xp[i] = p[i];
	}
    }
    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }

    for (; iter < niter; iter++) {
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  np_crmul(p[i+nprec],eps);
	    }
	}    
	if (err != NULL) err[iter] = cblas_scnrm2(ny, rr, 1);
    }  

    free (p);
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }

    if (mwt != NULL) {
	free (tp);
    }

}

void np_solver_reg (np_operator oper   /* linear operator */, 
		    np_solverstep solv /* stepping function */,
		    np_operator reg    /* regularization operator */, 
		    int nreg           /* size of reg{x} */, 
		    int nx             /* size of x */, 
		    int ny             /* size of dat */, 
		    float* x           /* estimated model */, 
		    const float* dat   /* data */, 
		    int niter          /* number of iterations */, 
		    double eps          /* regularization parameter */, 
		    ...                /* variable number of arguments */) 
/*< Generic regularized linear solver.
  ---
  Solves
  oper{x}    =~ dat
  eps reg{x} =~ 0
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  np_solver_reg (oper_lop,np_cgstep,reg_lop,
  np,nx,ny,x,y,100,1.0,"x0",x0,"end");
  ---
  Parameters in ...:
  
  "wt":     float*:         weight      
  "wght":   np_weight wght: weighting function
  "x0":     float*:         initial model
  "nloper": np_operator:    nonlinear operator  
  "nlreg":  np_operator:    nonlinear regularization operator
  "verb":   bool:           verbosity flag
  "known":  bool*:          known model mask
  "nmem":   int:            iteration memory
  "nfreq":  int:            periodic restart
  "xmov":   float**:        model iteration
  "rmov":   float**:        residual iteration
  "err":    float*:         final error
  "res":    float*:         final residual
  "resm":   float*:         final model residual
  >*/
{

    va_list args;
    char* par;
    float* wt = NULL;
    np_weight wght = NULL;
    float* x0 = NULL;
    np_operator nloper = NULL;
    np_operator nlreg = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* resm = NULL;
    float* wht = NULL;
    float *g, *rr, *gg, *tr, *td = NULL;
    float dpr, dpg, dpr0, dpg0;
    int i, iter;
    bool forget = false;

    va_start (args, eps);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_operator);}
	else if (0 == strcmp (par,"nlreg"))      
	{                    nlreg = va_arg (args, np_operator);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else if (0 == strcmp (par,"resm"))      
	{                    resm = va_arg (args, float*);}
	else 
	{ 
// 	np_error ("%s: unknown parameter %s",__FILE__,par);
	}
    }
    va_end (args);
 
    g =  np_floatalloc (nx);
    tr = np_floatalloc (nreg);
    rr = np_floatalloc (ny+nreg);
    gg = np_floatalloc (ny+nreg);

    if (wt != NULL || wght != NULL) {
	td = np_floatalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    for (i=0; i < ny; i++) {
	rr[i] = - dat[i];
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, ny, x, rr); 
	} else {
	    oper (false, true, nx, ny, x, rr); 
	}
	if (nlreg != NULL) {
	    nlreg  (false, false, nx, nreg, x, rr+ny);
	} else {
	    reg  (false, false, nx, nreg, x, rr+ny);            
	}
	cblas_sscal(nreg,eps,rr+ny,1);
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = 0.0;
	} 
	for (i=0; i < nreg; i++) {
	    rr[i+ny] = 0.0;
	}
    }

    dpr0 = cblas_snrm2(ny, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	for (i=0; i < nreg; i++) {
	    tr[i] = rr[i+ny]*eps;
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] *= wht[i];
		td[i] = rr[i]*wht[i];
	    }
      
	    np_array (oper, reg, true, false, nx, ny, nreg, g, td, tr);
	} else {
	    np_array (oper, reg, true, false, nx, ny, nreg, g, rr, tr);
	} 
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = 0.0;
	    }
	} 
	np_array (oper, reg, false, false, nx, ny, nreg, g, gg, gg+ny);

	cblas_sscal(nreg,eps,gg+ny,1);

	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] *= wht[i];
	    }
	}
 
	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq);
	}

	if (iter == 0) {
	    dpg0  = cblas_snrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = cblas_snrm2 (ny, rr, 1)/dpr0;	    
	    dpg = cblas_snrm2 (nx, g , 1)/dpg0;
	} 

	if (verb) 
	    printf ("iteration %d res dat %f res mod %f mod %f grad %f\n",
			iter, dpr, cblas_snrm2 (nreg, rr+ny, 1), cblas_snrm2 (nx, x, 1), dpg);

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);
	    break;
	}

	solv (forget, nx, ny+nreg, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }
	    nloper (false, true, nx, ny, x, rr);
	}
	if (nlreg != NULL) {
	    nlreg  (false, false, nx, nreg, x, rr+ny); 

	    cblas_sscal(nreg,eps,rr+ny,1);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }
	    oper (false, true, nx, ny, x, rr);
	}  
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = cblas_snrm2(ny, rr, 1);
	}
    }

    if (resm != NULL) {
	for (i=0; i < nreg; i++) {
	    resm[i] = rr[i+ny];
	}
    }
    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }

    free (tr);
    free (g); 
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }
}

void np_csolver_reg (np_coperator oper   /* linear operator */, 
		     np_csolverstep solv /* stepping function */,
		     np_coperator reg    /* regularization operator */, 
		     int nreg            /* size of reg{x} */, 
		     int nx              /* size of x */, 
		     int ny              /* size of dat */, 
		     np_complex* x           /* estimated model */, 
		     const np_complex* dat   /* data */, 
		     int niter          /* number of iterations */, 
		     double eps          /* regularization parameter */, 
		    ...                /* variable number of arguments */) 
/*< Generic regularized linear solver.
  ---
  Solves
  oper{x}    =~ dat
  eps reg{x} =~ 0
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  np_solver_reg (oper_lop,np_cgstep,reg_lop,
  np,nx,ny,x,y,100,1.0,"x0",x0,"end");
  ---
  Parameters in ...:
  
  "x0":     np_complex*:         initial model
  "nloper": np_operator:    nonlinear operator  
  "nlreg":  np_operator:    nonlinear regularization operator
  "verb":   bool:           verbosity flag
  "known":  bool*:          known model mask
  "nmem":   int:            iteration memory
  "nfreq":  int:            periodic restart
  "xmov":   np_complex**:        model iteration
  "rmov":   np_complex**:        residual iteration
  "err":    float*:         final error
  "res":    np_complex*:         final residual
  "resm":   np_complex*:         final model residual
  >*/
{

    va_list args;
    char* par;
    np_complex* x0 = NULL;
    np_coperator nloper = NULL;
    np_coperator nlreg = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    np_complex** xmov = NULL;
    np_complex** rmov = NULL;
    float* err = NULL;
    np_complex* res = NULL;
    np_complex* resm = NULL;
    np_complex *g, *rr, *gg, *tr;
    float dpr, dpg, dpr0, dpg0;
    int i, iter;
    bool forget = false;

    va_start (args, eps);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, np_complex*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_coperator);}
	else if (0 == strcmp (par,"nlreg"))      
	{                    nlreg = va_arg (args, np_coperator);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, np_complex**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, np_complex**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, np_complex*);}
	else if (0 == strcmp (par,"resm"))      
	{                    resm = va_arg (args, np_complex*);}
	else 
	{ 
// 	np_error ("%s: unknown parameter %s",__FILE__,par);
	}
    }
    va_end (args);
 
    g =  np_complexalloc (nx);
    tr = np_complexalloc (nreg);
    rr = np_complexalloc (ny+nreg);
    gg = np_complexalloc (ny+nreg);

    for (i=0; i < ny; i++) {
	rr[i] = np_cneg(dat[i]);
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, ny, x, rr); 
	} else {
	    oper (false, true, nx, ny, x, rr); 
	}
	if (nlreg != NULL) {
	    nlreg  (false, false, nx, nreg, x, rr+ny);
	} else {
	    reg  (false, false, nx, nreg, x, rr+ny);            
	}
	cblas_csscal(nreg,eps,rr+ny,1);
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = np_cmplx(0.0,0.0);
	} 
	for (i=0; i < nreg; i++) {
	    rr[i+ny] = np_cmplx(0.0,0.0);
	}
    }

    dpr0 = cblas_scnrm2(ny, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	for (i=0; i < nreg; i++) {
	    tr[i] = np_crmul(rr[i+ny],eps);
	}
	np_carray (oper, reg, true, false, nx, ny, nreg, g, rr, tr);
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = np_cmplx(0.0,0.0);
	    }
	} 
	np_carray (oper, reg, false, false, nx, ny, nreg, g, gg, gg+ny);

	cblas_csscal(nreg,eps,gg+ny,1);

	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq);
	}

	if (iter == 0) {
	    dpg0  = cblas_scnrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = cblas_scnrm2 (ny, rr, 1)/dpr0;	    
	    dpg = cblas_scnrm2 (nx, g , 1)/dpg0;
	} 

	if (verb) 
	    printf ("iteration %d res dat %f res mod %f mod %f grad %f\n",
			iter, dpr, cblas_scnrm2 (nreg, rr+ny, 1), cblas_scnrm2 (nx, x, 1), dpg);

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);
	    break;
	}

	solv (forget, nx, ny+nreg, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cneg(dat[i]);
	    }
	    nloper (false, true, nx, ny, x, rr);
	}
	if (nlreg != NULL) {
	    nlreg  (false, false, nx, nreg, x, rr+ny); 

	    cblas_csscal(nreg,eps,rr+ny,1);
	}
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = cblas_scnrm2(ny, rr, 1);
	}
    }

    if (resm != NULL) {
	for (i=0; i < nreg; i++) {
	    resm[i] = rr[i+ny];
	}
    }
    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }

    free (tr);
    free (g); 
    free (rr);
    free (gg);
}


void np_solver (np_operator oper   /* linear operator */, 
		np_solverstep solv /* stepping function */, 
		int nx             /* size of x */, 
		int ny             /* size of dat */, 
		float* x           /* estimated model */, 
		const float* dat   /* data */, 
		int niter          /* number of iterations */, 
		...                /* variable number of arguments */)
/*< Generic linear solver.
  ---
  Solves
  oper{x}    =~ dat
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  np_solver (oper_lop,np_cgstep,nx,ny,x,y,100,"x0",x0,"end");
  ---
  Parameters in ...:
  ---
  "wt":     float*:         weight      
  "wght":   np_weight wght: weighting function
  "x0":     float*:         initial model
  "nloper": np_operator:    nonlinear operator
  "mwt":    float*:         model weight
  "verb":   bool:           verbosity flag
  "known":  bool*:          known model mask
  "nmem":   int:            iteration memory
  "nfreq":  int:            periodic restart
  "xmov":   float**:        model iteration
  "rmov":   float**:        residual iteration
  "err":    float*:         final error
  "res":    float*:         final residual
  >*/ 
{

    va_list args;
    char* par;
    float* wt = NULL;
    np_weight wght = NULL;
    float* x0 = NULL;
    np_operator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* wht = NULL;
    float *g, *rr, *gg, *td = NULL, *g2 = NULL;
    float dpr, dpg, dpr0, dpg0;
    int i, iter; 
    bool forget = false;

    va_start (args, niter);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_operator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else 
	{ 
// 	np_error("solver: unknown argument %s",par);
	}
    }
    va_end (args);
 
    g =  np_floatalloc (nx);
    rr = np_floatalloc (ny);
    gg = np_floatalloc (ny);

    if (wt != NULL || wght != NULL) {
	td = np_floatalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    if (mwt != NULL) {
	g2 = np_floatalloc (nx);
    }

    for (i=0; i < ny; i++) {
	rr[i] = - dat[i];
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 	
	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, ny, x, rr);
	} else {
	    oper (false, true, nx, ny, x, rr);
            
	}
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = 0.0;
	} 
    }

    dpr0 = cblas_snrm2(ny, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] *= wht[i];
		td[i] = rr[i]*wht[i];
	    }
      
	    oper (true, false, nx, ny, g, td);
	} else {
	    oper (true, false, nx, ny, g, rr);
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g[i] *= mwt[i];
	    }
	}
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = 0.0;
	    }
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g2[i] = g[i]*mwt[i];
	    }
	    oper (false, false, nx, ny, g2, gg);
	} else {
	    oper (false, false, nx, ny, g, gg);
	}

	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] *= wht[i];
	    }
	}
 
	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq); 
	}


	if (iter == 0) {
	    dpg0  = cblas_snrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = cblas_snrm2 (ny, rr, 1)/dpr0;
	    dpg = cblas_snrm2 (nx, g , 1)/dpg0;
	}    

	if (verb) 
	    printf ("iteration %d res %f mod %f grad %f\n",
			iter+1, dpr, cblas_snrm2 (nx, x, 1), dpg);
	

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    break;
	}

	solv (forget, nx, ny, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    nloper (false, true, nx, ny, x, rr);
	} else if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }
	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    oper (false, true, nx, ny, x, rr);
	}  else if (mwt != NULL && (xmov != NULL || iter == niter-1)) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	}

	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = cblas_snrm2(ny, rr, 1);
	}
    }

    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }
  
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }

    if (mwt != NULL) {
	free (g2);
    }
}

void np_left_solver (np_operator oper   /* linear operator */, 
		     np_solverstep solv /* stepping function */, 
		     int nx             /* size of x and dat */, 
		     float* x           /* estimated model */, 
		     const float* dat   /* data */, 
		     int niter          /* number of iterations */, 
		     ...                /* variable number of arguments */)
/*< Generic linear solver for non-symmetric operators.
  ---
  Solves
  oper{x}    =~ dat
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  np_left_solver (oper_lop,np_cdstep,nx,ny,x,y,100,"x0",x0,"end");
  ---
  Parameters in ...:
  ---
  "wt":     float*:         weight      
  "wght":   np_weight wght: weighting function
  "x0":     float*:         initial model
  "nloper": np_operator:    nonlinear operator
  "mwt":    float*:         model weight
  "verb":   bool:           verbosity flag
  "known":  bool*:          known model mask
  "nmem":   int:            iteration memory
  "nfreq":  int:            periodic restart
  "xmov":   float**:        model iteration
  "rmov":   float**:        residual iteration
  "err":    float*:         final error
  "res":    float*:         final residual
  >*/ 
{

    va_list args;
    char* par;
    float* wt = NULL;
    np_weight wght = NULL;
    float* x0 = NULL;
    np_operator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* wht = NULL;
    float *g, *rr, *gg;
    float dpr, dpg, dpr0, dpg0;
    int i, iter; 
    bool forget = false;

    va_start (args, niter);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_operator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else 
	{ 
// 	np_error("solver: unknown argument %s",par);
	}
    }
    va_end (args);
 
    g =  np_floatalloc (nx);
    rr = np_floatalloc (nx);
    gg = np_floatalloc (nx);

    if (wt != NULL || wght != NULL) {
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (nx);
	    for (i=0; i < nx; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    for (i=0; i < nx; i++) {
	rr[i] = - dat[i];
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 	
	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, nx, x, rr);
	} else {
	    oper (false, true, nx, nx, x, rr);
            
	}
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = 0.0;
	} 
    }

    dpr0 = cblas_snrm2(nx, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (nx, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < nx; i++) {
		rr[i] *= wht[i];
		g[i] = rr[i]*wht[i];
	    }
	} else {
	    for (i=0; i < nx; i++) {
		g[i] = rr[i];
	    }
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g[i] *= mwt[i];
	    }
	}
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = 0.0;
	    }
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g[i] *= mwt[i];
	    }
	}
	
	oper (false, false, nx, nx, g, gg);

	if (wht != NULL) {
	    for (i=0; i < nx; i++) {
		gg[i] *= wht[i];
	    }
	}
 
	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq); 
	}


	if (iter == 0) {
	    dpg0  = cblas_snrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = cblas_snrm2 (nx, rr, 1)/dpr0;
	    dpg = cblas_snrm2 (nx, g , 1)/dpg0;
	}    

	if (verb) 
	    printf ("iteration %d res %f mod %f grad %f\n",
			iter+1, dpr, cblas_snrm2 (nx, x, 1), dpg);
	

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    break;
	}

	solv (forget, nx, nx, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < nx; i++) {
		rr[i] = -dat[i]; 
	    }

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    nloper (false, true, nx, nx, x, rr);
	} else if (wht != NULL) {
	    for (i=0; i < nx; i++) {
		rr[i] = -dat[i]; 
	    }
	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    oper (false, true, nx, nx, x, rr);
	}  else if (mwt != NULL && (xmov != NULL || iter == niter-1)) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	}

	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < nx; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = cblas_snrm2(nx, rr, 1);
	}
    }

    if (res != NULL) {
	for (i=0; i < nx; i++) {
	    res[i] = rr[i];
	}
    }
  
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL && wt == NULL) free (wht);
}


void np_csolver (np_coperator oper        /* linear operator */, 
		 np_csolverstep solv      /* stepping function */, 
		 int nx                   /* size of x */, 
		 int ny                   /* size of dat */, 
		 np_complex* x            /* estimated model */, 
		 const np_complex* dat    /* data */, 
		 int niter                /* number of iterations */, 
		 ...                      /* variable number of arguments */) 
/*< Generic linear solver for complex data.
  ---
  Solves
  oper{x}    =~ dat
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  np_csolver (oper_lop,np_cgstep,nx,ny,x,y,100,"x0",x0,"end");
  ---
  Parameters in ...:
  ---
  "wt":     float*:          weight      
  "wght":   np_cweight wght: weighting function
  "x0":     np_complex*:  initial model
  "nloper": np_coperator:    nonlinear operator  
  "verb":   bool:            verbosity flag
  "known":  bool*:           known model mask
  "nmem":   int:             iteration memory
  "nfreq":  int:             periodic restart
  "xmov":   np_complex**: model iteration
  "rmov":   np_complex**: residual iteration
  "err":    float*:  final error
  "res":    np_complex*:  final residual
  >*/ 
{

    va_list args;
    char* par;
    float * wt = NULL;
    np_cweight wght = NULL;
    np_complex * x0 = NULL;
    np_coperator nloper = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    np_complex ** xmov = NULL;
    np_complex ** rmov = NULL;
    float * err = NULL;
    np_complex * res = NULL;
    float * wht = NULL;
    np_complex *g, *rr, *gg, *td = NULL;
    float dpr, dpg, dpr0, dpg0;
    int i, iter; 
    bool forget = false;

    va_start (args, niter);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, np_cweight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, np_complex *);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, np_coperator);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, np_complex **);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, np_complex **);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float *);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, np_complex *);}
	else 
	{ 
// 	np_error("solver: unknown argument %s",par);
	}
    }
    va_end (args);
 
    g =  np_complexalloc (nx);
    rr = np_complexalloc (ny);
    gg = np_complexalloc (ny);

    if (wt != NULL || wght != NULL) {
	td = np_complexalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = np_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    for (i=0; i < ny; i++) {
	rr[i] = np_cneg(dat[i]);
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, ny, x, rr); 
	} else {
	    oper (false, true, nx, ny, x, rr); 
	}
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = np_cmplx(0.0,0.0);
	} 
    }

    dpr0 = cblas_scnrm2(ny, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_crmul(rr[i],wht[i]);
		td[i] = np_crmul(rr[i],wht[i]);
	    }
      
	    oper (true, false, nx, ny, g, td);
	} else {
	    oper (true, false, nx, ny, g, rr);
	} 
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = np_cmplx(0.0,0.0);
	    }
	} 
	oper (false, false, nx, ny, g, gg);
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] = np_crmul(gg[i],wht[i]);
	    }
	}
 
	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq); 
	}

	if (iter == 0) {
	    dpg0  = cblas_scnrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = cblas_scnrm2 (ny, rr, 1)/dpr0;
	    dpg = cblas_scnrm2 (nx, g , 1)/dpg0;
	}    

	if (verb) {
	    printf ("iteration %d res %f mod %f grad %f\n",
			iter+1, dpr, cblas_scnrm2 (nx, x, 1), dpg);
	}

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);
	    break;
	}

	solv (forget, nx, ny, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cneg(dat[i]);
	    }
	    nloper (false, true, nx, ny, x, rr);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = np_cneg(dat[i]);
	    }
	    oper (false, true, nx, ny, x, rr);
	}  
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = cblas_scnrm2(ny, rr, 1);
	}
    }

    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }
  
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }
}

