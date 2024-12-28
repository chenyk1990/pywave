/* Smooth division with several components and complex numbers */
// #include <rsf.h>
// #include "cdivn.h"

/*below is the including part*/
#include <math.h>
#include <stdio.h>
// #include <complex.h>
#include <stdbool.h>
#include "wave_alloc.h"
#include "wave_conjgrad.h"
#include "wave_komplex.h"
#include "wave_cdivn.h"

#include "wave_ctriangle.h"
#include "wave_ctrianglen.h"
#include "wave_cntriangle.h"
#include "wave_cntrianglen.h"
#include "wave_decart.h"

#define SF_MAX_DIM 9

/*below is from cdivn.c*/
static np_complex *p, **w;
static int n1, n2, nw;
static np_coperator oper;

void cmultidivn_init(int nw      /* number of components */, 
		       int ndim          /* number of dimensions */, 
		       int n             /* data size */, 
		       int *ndat         /* data dimensions [ndim] */, 
		       int *nbox         /* smoothing radius [ndim] */,
		       np_complex* den   /* denominator [nw*nd] */,
		       bool verb         /* verbosity flag */)
/*< initialize >*/
{
    int n2;

    n2 = n*nw;
    
    np_ctrianglen_init(ndim, nbox, ndat);
    crepeat_init(n,nw,np_ctrianglen_lop);

    np_cconjgrad_init(n2, n2, n, n, 1., 1.e-6, verb, false);
    p = np_complexalloc (n2);
    cweight2_init(nw,n,den);
}

void cmultidivn_close (void)
/*< free allocated storage >*/
{
    np_ctrianglen_close();
    np_cconjgrad_close();
    cweight2_close();
    free (p);
}

void cmultidivn (np_complex* num  /* numerator */, 
		 np_complex* rat  /* ratio */, 
		 int niter        /* number of iterations */)
/*< smoothly divide num/rat >*/
{
    np_cconjgrad(NULL,cweight2_lop,crepeat_lop,p,rat,num,niter);
}

void crepeat_init(int m1            /* trace length */, 
		  int m2            /* number of traces */, 
		  np_coperator oper1 /* operator */)
/*< initialize >*/
{
    n1 = m1;
    n2 = m2;
    oper = oper1;
}

void crepeat_lop (bool adj, bool add, int nx, int ny, np_complex *xx, np_complex *yy)
/*< combined linear operator >*/
{
    int i2;       
    
    if (nx != ny || nx != n1*n2) 
// 	np_error("%s: Wrong size (nx=%d ny=%d n1=%d n2=%d)",
// 		 __FILE__,nx,ny,n1,n2);

    np_cadjnull (adj, add, nx, ny, xx, yy);

    for (i2=0; i2 < n2; i2++) {
	oper(adj,true,n1,n1,xx+i2*n1,yy+i2*n1);
    }
}

void cweight2_init(int nw1        /* number of components */, 
		   int n          /* model size */, 
		   np_complex *ww /* weight [nw*n] */)
/*< initialize >*/
{
    int iw;

    nw = nw1;
    w = (np_complex**) np_alloc(nw,sizeof(np_complex*));

    for (iw=0; iw < nw; iw++) {
	w[iw] = ww+iw*n;
    }
}

void cweight2_close(void)
/*< free allocated storage >*/
{
    free(w);
}

void cweight2_lop (bool adj, bool add, int nx, int ny, np_complex* xx, np_complex* yy)
/*< linear operator >*/
{
    int i, iw;

//     if (nw*ny != nx) np_error("%s: size mismatch: %d*%d != %d",
// 			      __FILE__,nw,ny,nx);

    np_cadjnull (adj, add, nx, ny, xx, yy);
  
    for (iw=0; iw < nw; iw++) {
	for (i=0; i < ny; i++) {
	    if (adj) {
		xx[i+iw*ny] = np_cadd(xx[i+iw*ny],
				      np_cmul(yy[i],conjf(w[iw][i])));
	    } else {
		yy[i] = np_cadd(yy[i],np_cmul(xx[i+iw*ny],w[iw][i]));
	    }
	}
    }
}

/*direct weight*/
static np_complex* ww;

void cweight_init(np_complex *w1)
/*< initialize >*/
{
    ww = w1;
}

void cweight_lop (bool adj, bool add, int nx, int ny, 
		  np_complex* xx, np_complex* yy)
/*< linear operator >*/
{
    int i;

//     if (ny!=nx) np_error("%s: size mismatch: %d != %d",__FILE__,ny,nx);

    np_cadjnull (adj, add, nx, ny, xx, yy);
  
    for (i=0; i < nx; i++) {
	if (adj) {
	    xx[i] = np_cadd(xx[i],np_cmul(yy[i],conjf(ww[i])));
	} else {
	    yy[i] = np_cadd(yy[i],np_cmul(xx[i],ww[i]));
	}
    }
}
/*above is from cdivn.c*/


/*below is from cdivn_fs.c*/
static int *n, s[SF_MAX_DIM], nd, dim; 
static np_ctriangle tr0; /*triangle smoother in frequency*/
static np_ctriangle *tr;
static np_complex *tmp;

// static np_complex *p;
static int nf; /*nf is the dimension of frequency*/

void cmultidivn_fs_init(int nw            /* number of components */, 
		       int ndim          /* number of dimensions */, 
		       int n             /* data size */, 
		       int *ndat         /* data dimensions [ndim] */, 
			   int nbox0  /* triangle radius in frequency */, 
			   int **nbox /* triangle radius [ndim-1] */, 
		       np_complex* den   /* denominator [nw*nd] */,
		       bool verb         /* verbosity flag */)
/*< initialize cmultidivn with frequency-dependent smoothing >*/
{
    int n2;

    nf = ndat[0];
    n2 = n*nw;
    smooth_fs_init(ndim, nbox0, nbox, ndat);
    crepeat_init(n,nw,smooth_fs_lop);
    np_cconjgrad_init(n2, n2, n, n, 1., 1.e-6, verb, false);
    p = np_complexalloc (n2);
    cweight2_init(nw,n,den);    
}


void cmultidivn_fs_close (void)
/*< free allocated storage >*/
{
	smooth_fs_close();
    np_cconjgrad_close();
    cweight2_close();
    free (p);
}

void cmultidivn_fs (np_complex* num  /* numerator */, 
		 np_complex* rat  /* ratio */, 
		 int niter        /* number of iterations */)
/*< smoothly divide num/rat >*/
{
    np_cconjgrad(NULL,cweight2_lop,crepeat_lop,p,rat,num,niter);
}


void smooth_fs_init (int ndim  /* number of dimensions */, 
			int nbox0  /* triangle radius in frequency */, 
			int **nbox /* triangle radius [ndim-1] */, 
			int *ndat /* data dimensions [ndim] */)
/*< initialize >*/
{
    int i, j, k;

    dim = ndim;
    n = np_intalloc(dim);
    
    tr = (np_ctriangle*) np_alloc(dim*ndat[0],sizeof(np_ctriangle));
    nd = 1;
    for (i=0; i < dim; i++) {
	s[i] = nd;
	n[i] = ndat[i];
	nd *= ndat[i];
    }
	/*for frequency*/
    tr0 = (nbox0 > 1)? np_ctriangle_init (nbox0,ndat[0],false): NULL; /*smoother for frequency*/
	/*for x and y*/
    for (i=1; i < dim; i++) {
	    for (j=0; j < nd/n[i]; j++) {
	    k=j%nf;
	    tr[i*nf+k] = (nbox[i-1][k] > 1)? np_ctriangle_init (nbox[i-1][k],ndat[i],false): NULL;
	    }
    }
    tmp = np_complexalloc (nd);
}


void smooth_fs_lop (bool adj, bool add, int nx, int ny, np_complex* x, np_complex* y)
/*< linear operator >*/
{
    int i, j, k, i0;

    if (nx != ny || nx != nd) 
// 	np_error("%s: Wrong data dimensions: nx=%d, ny=%d, nd=%d",
// 		 __FILE__,nx,ny,nd);

    np_cadjnull (adj,add,nx,ny,x,y);
  
    if (adj) {
	for (i=0; i < nd; i++) {
	    tmp[i] = y[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    tmp[i] = x[i];
	}
    }

	/*for frequency*/
	if (NULL != tr0) {
	    for (j=0; j < nd/n[0]; j++) {
		i0 = np_first_index (0,j,dim,n,s);
		np_csmooth (tr0, i0, s[0], false, tmp);
	    }
	}

    /*for x and y*/
    for (i=1; i < dim; i++) {
	    for (j=0; j < nd/n[i]; j++) {
	    k=j%nf;
	    if (NULL != tr[i*nf+k]) {
		i0 = np_first_index (i,j,dim,n,s);
		np_csmooth (tr[i*nf+k], i0, s[i], false, tmp);
	    }
	}
    }
	
    if (adj) {
	for (i=0; i < nd; i++) {
#ifdef NP_HAS_COMPLEX_H
	    x[i] += tmp[i];
#else
	    x[i] = np_cadd(x[i],tmp[i]);
#endif
	}
    } else {
	for (i=0; i < nd; i++) {
#ifdef NP_HAS_COMPLEX_H
	    y[i] += tmp[i];
#else
	    y[i] = np_cadd(y[i],tmp[i]);
#endif
	}
    }    
}

void smooth_fs_close (void)
/*< free allocated storage >*/
{
/*	int i;
	for(i=0;i<nf*dim;i++)
	{
		if (NULL != tr[i]) 
			np_ctriangle_close(tr[i]);
	}*/ 	/*what's wrong with it, only for 2D denoising ???*/
	free(tr);
	if (NULL != tr0) np_ctriangle_close(tr0);
	free(n);
	free(tmp);
}
/*above is from cdivn_fs.c*/


/*below is from cdivn_rnar.c*/

// static int *n, s[SF_MAX_DIM], nd, dim;
// static np_ctriangle *tr;
// static np_complex *tmp;
// static np_complex *p;

void cmultidivn_rnar_init(int nw            /* number of components */, 
		       int ndim          /* number of dimensions */, 
		       int n             /* data size */, 
		       int *ndat         /* data dimensions [ndim] */, 
			   int *nbox /* triangle radius [ndim-1] */, 
		       np_complex* den   /* denominator [nw*nd] */,
		       bool verb         /* verbosity flag */)
/*< initialize cmultidivn with frequency-dependent smoothing >*/
{
    int n2;

    n2 = n*nw;
    smooth_rnar_init(ndim, nbox, ndat);
    crepeat_init(n,nw,smooth_rnar_lop);
    np_cconjgrad_init(n2, n2, n, n, 1., 1.e-6, verb, false);
    p = np_complexalloc (n2);
    cweight2_init(nw,n,den);
}

void cmultidivn_rnar_close (void)
/*< free allocated storage >*/
{
	smooth_rnar_close();
    np_cconjgrad_close();
    cweight2_close();
    free (p);
}

void cmultidivn_rnar (np_complex* num  /* numerator */, 
		 np_complex* rat  /* ratio */, 
		 int niter        /* number of iterations */)
/*< smoothly divide num/rat >*/
{
    np_cconjgrad(NULL,cweight2_lop,crepeat_lop,p,rat,num,niter);
}


void smooth_rnar_init (int ndim  /* number of dimensions */, 
			int *nbox /* triangle radius [ndim-1] */, 
			int *ndat /* data dimensions [ndim] */)
/*< initialize >*/
{
    int i;

    dim = ndim;
    n = np_intalloc(dim);
    
    tr = (np_ctriangle*) np_alloc(dim,sizeof(np_ctriangle));
    nd = 1;
    for (i=0; i < dim; i++) {
	tr[i] = (nbox[i] > 1)? np_ctriangle_init (nbox[i],ndat[i],false): NULL;
	s[i] = nd;
	n[i] = ndat[i];
	nd *= ndat[i];
    }

    tmp = np_complexalloc (nd);
}

void smooth_rnar_close(void)
/*< free allocated storage >*/
{
    int i;

	free(n);
    free (tmp);

    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) np_ctriangle_close (tr[i]);
    }

    free(tr);
}


void smooth_rnar_lop (bool adj, bool add, int nx, int ny, np_complex* x, np_complex* y)
/*< linear operator >*/
{
    int i, j, i0;

    if (nx != ny || nx != nd) 
// 	np_error("%s: Wrong data dimensions: nx=%d, ny=%d, nd=%d",
// 		 __FILE__,nx,ny,nd);

    np_cadjnull (adj,add,nx,ny,x,y);
  
    if (adj) {
	for (i=0; i < nd; i++) {
	    tmp[i] = y[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    tmp[i] = x[i];
	}
    }

    /*only for x and y when rect[0]=1 (rect1=1 in command line)*/
    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) {
	    for (j=0; j < nd/n[i]; j++) {
		i0 = np_first_index (i,j,dim,n,s);
		np_csmooth (tr[i], i0, s[i], false, tmp);
	    }
	}
    }
	
    if (adj) {
	for (i=0; i < nd; i++) {
	    x[i] = np_cadd(x[i],tmp[i]);
	}
    } else {
	for (i=0; i < nd; i++) {
	    y[i] = np_cadd(y[i],tmp[i]);
	}
    }    
}


/*above is from cdivn_rnar.c*/


/*below is from cdivnn.c*/
// static int n2;
// static np_complex *p;

void cmultidivnn_init(int nw            /* number of components */, 
		       int ndim          /* number of dimensions */, 
		       int n             /* data size */, 
		       int *ndat         /* data dimensions [ndim] */, 
		       int *nbox         /* smoothing radius [ndim] */,
		       float **rct /* triangle lengths [ndim][nd] */,
               int **sft /* triangle shifts [ndim][nd] */,
		       np_complex* den   /* denominator [nw*nd] */,
		       bool verb         /* verbosity flag */)
/*< initialize cmultidivn with frequency-dependent smoothing >*/
{
    int n2;

    n2 = n*nw;
    
    cntrianglen_init(ndim, nbox, ndat, rct, sft, 1);
    crepeat_init(n,nw,cntrianglen_lop);

    np_cconjgrad_init(n2, n2, n, n, 1., 1.e-6, verb, false);
    p = np_complexalloc (n2);
    cweight2_init(nw,n,den);
}

void cmultidivnn_close (void)
/*< free allocated storage >*/
{
	cntrianglen_close();
    np_cconjgrad_close();
    cweight2_close();
    free (p);
}

void cmultidivnn (np_complex* num  /* numerator */, 
		 np_complex* rat  /* ratio */, 
		 int niter        /* number of iterations */)
/*< smoothly divide num/rat >*/
{
    np_cconjgrad(NULL,cweight2_lop,crepeat_lop,p,rat,num,niter);
}



/**Following is single division*/

static int niter, nn;
// static np_complex *p; /*defined above*/

void cdivnn_init(int ndim   /* number of dimensions */, 
	       int nd     /* data size */, 
	       int *ndat  /* data dimensions [ndim] */, 
	       int *nbox  /* smoothing radius [ndim] */, 
		   float **rct /* triangle lengths [ndim][nd] */,
           int **sft /* triangle shifts [ndim][nd] */,
	       int niter1 /* number of iterations */,
	       bool verb  /* verbosity */) 
/*< initialize >*/
{
    niter = niter1;
    nn = nd;
	cntrianglen_init(ndim, nbox, ndat, rct, sft, 1);
//     np_ctrianglen_init(ndim, nbox, ndat);
    np_cconjgrad_init(nd, nd, nd, nd, 1., 1.e-6, verb, false);
    p = np_complexalloc (nd);
}

void cdivnn_close (void)
/*< free allocated storage >*/
{
    cntrianglen_close();
    np_cconjgrad_close();
    free (p);
}

void cdivnn (np_complex* num, np_complex* den,  np_complex* rat)
/*< smoothly divide rat=num/den >*/
{
    cweight_init(den);
    np_cconjgrad(NULL, cweight_lop,cntrianglen_lop,p,rat,num,niter); 
//     np_cconjgrad(NULL, cweight_lop,np_ctrianglen_lop,p,rat,num,niter); 

}


void cdivnne (np_complex* num, np_complex* den,  np_complex* rat)
/*< smoothly divide rat=num/den with preconditioning >*/
{
    int id;
    float a, norm=0.; 

    for (id=0; id < nn; id++) {
	a = cabsf(den[id]);
	norm += a*a;
    }
    norm = sqrtf(nn/norm);

    for (id=0; id < nn; id++) {

	num[id] = np_crmul(num[id],norm);
	den[id] = np_crmul(den[id],norm);

    }
    
    cweight_init(den);
    np_cconjgrad(NULL, cweight_lop,cntrianglen_lop,p,rat,num,niter); 
}
/*above is from cdivnn.c*/


/*below is from cdivns.c*/
// static np_complex *p, **w;
// static int n1, n2, nw, nf;
static int nf;
// static np_coperator oper;

void cmultidivns_init(int nw            /* number of components */, 
		       int ndim          /* number of dimensions */, 
		       int n             /* data size */, 
		       int *ndat         /* data dimensions [ndim] */, 
		       int *nbox         /* smoothing radius [ndim] */,
		       np_complex* den   /* denominator [nw*nd] */,
		       bool verb         /* verbosity flag */)
/*< initialize >*/
{
    int n2;

    n2 = n*nw;
    nf=ndat[0]; /*frequency points*/
    np_ctrianglen_init(ndim, nbox, ndat);
    crepeats_init(nf,nw,np_ctrianglen_lop);

    np_cconjgrad_init(nw*nf,nw*nf, n, n, 1., 1.e-6, verb, false);
    p = np_complexalloc (n2);
    cweight2s_init(nw,n,den);
}

void cmultidivns_close (void)
/*< free allocated storage >*/
{
    np_ctrianglen_close();
    np_cconjgrad_close();
    cweight2s_close();
    free (p);
}

void cmultidivns (np_complex* num  /* numerator */, 
		 np_complex* rat  /* ratio */, 
		 int niter        /* number of iterations */)
/*< smoothly divide num/rat >*/
{
    np_cconjgrad(NULL,cweight2s_lop,crepeats_lop,p,rat,num,niter);
}


void crepeats_init(int m1            /* trace length */, 
		  int m2            /* number of traces */, 
		  np_coperator oper1 /* operator */)
/*< initialize >*/
{
    n1 = m1;
    n2 = m2;
    oper = oper1;
}

void crepeats_lop (bool adj, bool add, int nx, int ny, np_complex *xx, np_complex *yy)
/*< combined linear operator >*/
{
    int i2;       
    
    if (nx != ny || nx != n1*n2) 
// 	np_error("%s: Wrong size (nx=%d ny=%d n1=%d n2=%d)",
// 		 __FILE__,nx,ny,n1,n2);

    np_cadjnull (adj, add, nx, ny, xx, yy);

    for (i2=0; i2 < n2; i2++) {
	oper(adj,true,n1,n1,xx+i2*n1,yy+i2*n1);
    }
}

void cweight2s_init(int nw1        /* number of components */, 
		   int n          /* model size */, 
		   np_complex *ww /* weight [nw*n] */)
/*< initialize >*/
{
    int iw;

    nw = nw1;
    w = (np_complex**) np_alloc(nw,sizeof(np_complex*));

    for (iw=0; iw < nw; iw++) {
	w[iw] = ww+iw*n;
    }
}

void cweight2s_close(void)
/*< free allocated storage >*/
{
    free(w);
}

void cweight2s_lop (bool adj, bool add, int nx, int ny, np_complex* xx, np_complex* yy)
/*< linear operator >*/
{
    int i,iff, iw;
	int n22; /*n22=n2*n3;ny=n2*n3*nf;nx=nf*nw*/
	n22=ny/nf;


    np_cadjnull (adj, add, nx, ny, xx, yy);
  
    for (iw=0; iw < nw; iw++) {
    for (iff=0;iff<nf;iff++){
	for (i=0; i < n22; i++) {
	    if (adj) {
		xx[iw*nf+iff] = np_cadd(xx[iw*nf+iff],
				      np_cmul(yy[i*nf+iff],conjf(w[iw][i*nf+iff])));
	    } else {

		yy[i*nf+iff] = np_cadd(yy[i*nf+iff],np_cmul(xx[iw*nf+iff],w[iw][i*nf+iff]));
	    }
	}
	}
    }
}
/*up is from cdivns.c*/

