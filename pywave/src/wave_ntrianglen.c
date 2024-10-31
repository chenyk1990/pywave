/* N-D non-stationary triangle smoothing as a linear operator 
for both real (future?) and complex values
*/


#include <stdbool.h>
#include "npre_conjgrad.h"
#include "npre_dtype.h"
#include "npre_komplex.h"
#include "npre_alloc.h"
#include "npre_triangle.h"
#include "npre_trianglen.h"
#include "npre_ntriangle.h"
#include "npre_ntrianglen.h"
#include "npre_decart.h"

// #ifndef ntriangle_h
// struct CNtriangle {
//     np_complex *tmp;
//     int np, nb, nx;
// };
// typedef struct CNtriangle *cntriangle;
// /* abstract data type */
// /*^*/
// #endif

// struct CNtriangle {
//     np_complex *tmp;
//     int np, nb, nx;
// };
// 
// typedef struct CNtriangle *cntriangle;
/* abstract data type */

#define SF_MAX_DIM 9

static int *n, s[SF_MAX_DIM], nd, dim, **tsft, nrep;
static cntriangle *tr;
static np_complex *tmp;
static float **tlen;

void cntrianglen_init (int ndim  /* number of dimensions */, 
		      int *nbox /* triangle radius [ndim] */, 
		      int *ndat /* data dimensions [ndim] */,
		      float **len /* triangle lengths [ndim][nd] */,
                      int **sft /* triangle shifts [ndim][nd] */,
		      int repeat /* repeated smoothing */)
/*< initialize >*/
{
    int i;

    n = ndat;
    dim = ndim;

    tr = (cntriangle*) np_alloc(dim,sizeof(cntriangle));
    
    nd = 1;
    for (i=0; i < dim; i++) {
	tr[i] = (nbox[i] > 1)? cntriangle_init (nbox[i],ndat[i]): NULL;
	s[i] = nd;
	nd *= ndat[i];
    }
    tlen = len; 
    tsft = sft;

    tmp = np_complexalloc(nd);
    nrep = repeat;
}

void cntrianglen_lop (bool adj, bool add, int nx, int ny, np_complex* x, np_complex* y)
/*< linear operator >*/
{
    int i, j, i0, irep;

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

		for (irep=0; irep < nrep; irep++) {
		    cnsmooth (tr[i], i0, s[i], false, tlen[i], tsft[i], tmp);
		}
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

void cntrianglen_close(void)
/*< free allocated storage >*/
{
    int i;

    free (tmp);

    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) cntriangle_close (tr[i]);
    }

    free(tr);
}