
#ifndef bigsolver_h
#define bigsolver_h

#include<stdbool.h>

#include "wave_komplex.h"

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
/*^*/


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
		     ...                /* variable number of arguments */);
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
		     ...                /* variable number of arguments */);
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
		    ...                /* variable number of arguments */);
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
		    ...                /* variable number of arguments */);
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


void np_solver (np_operator oper   /* linear operator */, 
		np_solverstep solv /* stepping function */, 
		int nx             /* size of x */, 
		int ny             /* size of dat */, 
		float* x           /* estimated model */, 
		const float* dat   /* data */, 
		int niter          /* number of iterations */, 
		...                /* variable number of arguments */);
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


void np_left_solver (np_operator oper   /* linear operator */, 
		     np_solverstep solv /* stepping function */, 
		     int nx             /* size of x and dat */, 
		     float* x           /* estimated model */, 
		     const float* dat   /* data */, 
		     int niter          /* number of iterations */, 
		     ...                /* variable number of arguments */);
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


void np_csolver (np_coperator oper        /* linear operator */, 
		 np_csolverstep solv      /* stepping function */, 
		 int nx                   /* size of x */, 
		 int ny                   /* size of dat */, 
		 np_complex* x            /* estimated model */, 
		 const np_complex* dat    /* data */, 
		 int niter                /* number of iterations */, 
		 ...                      /* variable number of arguments */);
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

#endif
