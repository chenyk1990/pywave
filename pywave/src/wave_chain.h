#ifndef chain_h
#define chain_h

#include<stdbool.h>

#include "wave_komplex.h"
#include "wave_bigsolver.h"
// #include "_bool.h"
// #include "_solver.h"

void np_chain( np_operator oper1     /* outer operator */, 
	       np_operator oper2     /* inner operator */, 
	       bool adj              /* adjoint flag */, 
	       bool add              /* addition flag */, 
	       int nm                /* model size */, 
	       int nd                /* data size */, 
	       int nt                /* intermediate size */, 
	       /*@out@*/ float* mod  /* [nm] model */, 
	       /*@out@*/ float* dat  /* [nd] data */, 
	       float* tmp            /* [nt] intermediate */);
/*< Chains two operators, computing oper1{oper2{mod}} 
  or its adjoint. The tmp array is used for temporary storage. >*/


void np_cchain( np_coperator oper1         /* outer operator */, 
		np_coperator oper2         /* inner operator */, 
		bool adj                   /* adjoint flag */, 
		bool add                   /* addition flag */, 
		int nm                     /* model size */, 
		int nd                     /* data size */, 
		int nt                     /* intermediate size */, 
		/*@out@*/ np_complex* mod  /* [nm] model */, 
		/*@out@*/ np_complex* dat  /* [nd] data */, 
		np_complex* tmp            /* [nt] intermediate */);
/*< Chains two complex operators, computing oper1{oper2{mod}} 
  or its adjoint. The tmp array is used for temporary storage. >*/


void np_array( np_operator oper1     /* top operator */, 
	       np_operator oper2     /* bottom operator */, 
	       bool adj              /* adjoint flag */, 
	       bool add              /* addition flag */, 
	       int nm                /* model size */, 
	       int nd1               /* top data size */, 
	       int nd2               /* bottom data size */, 
	       /*@out@*/ float* mod  /* [nm] model */, 
	       /*@out@*/ float* dat1 /* [nd1] top data */, 
	       /*@out@*/ float* dat2 /* [nd2] bottom data */);
/*< Constructs an array of two operators, 
  computing {oper1{mod},oper2{mod}} or its adjoint. >*/


void np_carray( np_coperator oper1     /* top operator */, 
	       np_coperator oper2     /* bottom operator */, 
	       bool adj              /* adjoint flag */, 
	       bool add              /* addition flag */, 
	       int nm                /* model size */, 
	       int nd1               /* top data size */, 
	       int nd2               /* bottom data size */, 
	       /*@out@*/ np_complex* mod  /* [nm] model */, 
	       /*@out@*/ np_complex* dat1 /* [nd1] top data */, 
	       /*@out@*/ np_complex* dat2 /* [nd2] bottom data */);
/*< Constructs an array of two complex operators, 
  computing {oper1{mod},oper2{mod}} or its adjoint. >*/


void np_normal (np_operator oper /* operator */, 
		bool add         /* addition flag */, 
		int nm           /* model size */, 
		int nd           /* data size */, 
		float *mod       /* [nd] model */, 
		float *dat       /* [nd] data */, 
		float *tmp       /* [nm] intermediate */);
/*< Applies a normal operator (self-adjoint) >*/


void np_chain3 (np_operator oper1 /* outer operator */, 
		np_operator oper2 /* middle operator */, 
		np_operator oper3 /* inner operator */, 
		bool adj          /* adjoint flag */, 
		bool add          /* addition flag */, 
		int nm            /* model size */, 
		int nt1           /* inner intermediate size */, 
		int nt2           /* outer intermediate size */, 
		int nd            /* data size */, 
		float* mod        /* [nm] model */, 
		float* dat        /* [nd] data */, 
		float* tmp1       /* [nt1] inner intermediate */, 
		float* tmp2       /* [nt2] outer intermediate */);
/*< Chains three operators, computing oper1{oper2{poer3{{mod}}} or its adjoint.
  The tmp1 and tmp2 arrays are used for temporary storage. >*/

#endif
