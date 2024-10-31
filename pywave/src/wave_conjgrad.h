#ifndef _conjgrad_h
#define _conjgrad_h

#include "npre_dtype.h"
#include <stdbool.h>

// #include "_bool.h"
// #include "c99.h"
// #include "_solver.h"

/*below is from _solver.h */
typedef void (*np_operator)(bool,bool,int,int,float*,float*);
typedef void (*np_solverstep)(bool,int,int,float*,
			   const float*,float*,const float*);
typedef void (*np_weight)(int,const float*,float*);
/*^*/

typedef void (*np_coperator)(bool,bool,int,int,np_complex*,np_complex*);
typedef void (*np_csolverstep)(bool,int,int,np_complex*,
			       const np_complex*,np_complex*,
			       const np_complex*);
typedef void (*np_cweight)(int,const np_complex*,float*);
/*above is from _solver.h */

void np_adjnull (bool adj /* adjoint flag */, 
		 bool add /* addition flag */, 
		 int nx   /* size of x */, 
		 int ny   /* size of y */, 
		 float* x, 
		 float* y);
/*< Zeros out the output (unless add is true). 
  Useful first step for any linear operator. >*/
  
void np_cadjnull (bool adj /* adjoint flag */, 
		  bool add /* addition flag */, 
		  int nx   /* size of x */, 
		  int ny   /* size of y */, 
		  np_complex* x, 
		  np_complex* y);
/*< adjnull version for complex data. >*/


void np_cconjgrad_init(int np1     /* preconditioned size */, 
		       int nx1     /* model size */, 
		       int nd1     /* data size */, 
		       int nr1     /* residual size */, 
		       float eps1  /* scaling */,
		       float tol1  /* tolerance */, 
		       bool verb1  /* verbosity flag */, 
		       bool hasp01 /* if has initial model */);
/*< solver constructor >*/


void np_cconjgrad_close(void);
/*< Free allocated space >*/


void np_cconjgrad(np_coperator prec     /* data preconditioning */, 
		  np_coperator oper     /* linear operator */, 
		  np_coperator shape    /* shaping operator */, 
		  np_complex* p         /* preconditioned model */, 
		  np_complex* x         /* estimated model */, 
		  const np_complex* dat /* data */, 
		  int niter             /* number of iterations */);
/*< Conjugate gradient solver with shaping >*/



#endif