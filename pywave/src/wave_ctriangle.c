/* Triangle smoothing for complex numbers */
/*^*/
#include <stdbool.h>
#include "wave_dtype.h"
#include "wave_komplex.h"
#include "wave_alloc.h"
#include "wave_ctriangle.h"


static void fold (int o, int d, int nx, int nb, int np, 
		 np_complex *x, const np_complex* tmp);
static void doubint (int nx, np_complex *x, bool der);
static void triple (int o, int d, int nx, int nb, const np_complex* x, np_complex* tmp, bool box, float wt);

np_ctriangle np_ctriangle_init (int nbox /* triangle length */, 
				int ndat /* data length */,
				bool box /* if box instead of triangle */)
/*< initialize >*/
{
    np_ctriangle tr;

    tr = (np_ctriangle) np_alloc(1,sizeof(*tr));

    tr->nx = ndat;
    tr->nb = nbox;
    tr->box = box;
    tr->np = ndat + 2*nbox;

    if (box) {
	tr->wt = 1.0/(2*nbox-1);
    } else {
	tr->wt = 1.0/(nbox*nbox);
    }
    
    tr->tmp = np_complexalloc(tr->np);

    return tr;
}

static void fold (int o, int d, int nx, int nb, int np, 
		   np_complex *x, const np_complex* tmp)
{
    int i, j;

    /* copy middle */
    for (i=0; i < nx; i++) 
	x[o+i*d] = tmp[i+nb];

    /* reflections from the right side */
    for (j=nb+nx; j < np; j += nx) {
	for (i=0; i < nx && i < np-j; i++) {
	    x[o+(nx-1-i)*d] = np_cadd(x[o+(nx-1-i)*d],tmp[j+i]);
	}
	j += nx;
	for (i=0; i < nx && i < np-j; i++) {
	    x[o+i*d] = np_cadd(x[o+i*d],tmp[j+i]);
	}
    }
    
    /* reflections from the left side */
    for (j=nb; j >= 0; j -= nx) {
	for (i=0; i < nx && i < j; i++) {
	    x[o+i*d] = np_cadd(x[o+i*d],tmp[j-1-i]);
	}
	j -= nx;
	for (i=0; i < nx && i < j; i++) {
	    x[o+(nx-1-i)*d] = np_cadd(x[o+(nx-1-i)*d],tmp[j-1-i]);
	}
    }
}
    
static void doubint (int nx, np_complex *xx, bool der)
{
    int i;
    np_complex t;


    /* integrate forward */
    t=np_cmplx(0.,0.);
    for (i=0; i < nx; i++) {
	t = np_cadd(t,xx[i]);
	xx[i] = t;
    }

    if (der) return;

    /* integrate backward */
    t = np_cmplx(0.,0.);
    for (i=nx-1; i >= 0; i--) {
	t = np_cadd(t,xx[i]);
	xx[i] = t;
    }
}

static void triple (int o, int d, int nx, int nb, const np_complex* x, np_complex* tmp, bool box, float wt)
{
    int i;
    np_complex xi;

    for (i=0; i < nx + 2*nb; i++) {
	tmp[i] = np_cmplx(0.,0.);
    }

    if (box) {
	for (i=0; i < nx; i++) {
	    xi = np_crmul(x[o+i*d],wt);

	    tmp[i+1]    = np_cadd(tmp[i+1],xi);
	    tmp[i+2*nb] = np_cadd(tmp[i+2*nb],np_cneg(xi));
	}
    } else {
	for (i=0; i < nx; i++) {
	    xi = np_crmul(x[o+i*d],wt);

	    tmp[i]      = np_cadd(tmp[i],np_cneg(xi));
	    tmp[i+nb]   = np_cadd(tmp[i+nb],np_crmul(xi,2.));
	    tmp[i+2*nb] = np_cadd(tmp[i+2*nb],np_cneg(xi));
	}
    }
}

void np_csmooth (np_ctriangle tr    /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 np_complex *x   /* data (smoothed in place) */)
/*< apply adjoint triangle smoothing >*/
{
    triple (o,d,tr->nx,tr->nb,x,tr->tmp,tr->box,tr->wt);
    doubint (tr->np,tr->tmp,(bool) (tr->box || der));
    fold (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
}

void  np_ctriangle_close(np_ctriangle tr)
/*< free allocated storage >*/
{
    free (tr->tmp);
    free (tr);
}




