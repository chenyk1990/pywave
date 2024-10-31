/* N-D triangle smoothing as a linear operator for complex numbers */

// #include "ctrianglen.h"
// #include "ctriangle.h"
// #include "alloc.h"
// #include "file.h"
// #include "error.h"
// #include "adjnull.h"
// #include "decart.h"
// 
// #include "_bool.h"
// #include "komplex.h"
/*^*/

#include "wave_conjgrad.h"
#include "wave_komplex.h"
#include "wave_alloc.h"
#include "wave_dtype.h"
#include "wave_triangle.h"
#include "wave_trianglen.h"
#include "wave_decart.h"


#define SF_MAX_DIM 9

static int *n, s[SF_MAX_DIM], nd, dim;
static np_ctriangle *tr;
static np_complex *tmp;

void np_ctrianglen_init (int ndim  /* number of dimensions */, 
			int *nbox /* triangle radius [ndim] */, 
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

void np_ctrianglen_lop (bool adj, bool add, int nx, int ny, np_complex* x, np_complex* y)
/*< linear operator >*/
{
    int i, j, i0;

//     if (nx != ny || nx != nd) 
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

void np_ctrianglen_close(void)
/*< free allocated storage >*/
{
    int i;

    free (tmp);

    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) np_ctriangle_close (tr[i]);
    }

    free(tr);
    free(n);
}

