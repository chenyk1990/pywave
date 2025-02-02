/*below is the including part*/
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "wave_alloc.h"
#include "wave_ricker.h"
#include "wave_abc.h"
#include "wave_komplex.h"
#include "wave_psp.h"
#include "wave_fdm.h"
#include "wave_freqfilt.h"
#define np_PI (3.14159265358979323846264338328)

#include "wave_fwi.h"
#include "wave_fwiutil.h"
#include "wave_fwigradient.h"
#include "wave_fwilbfgs.h"
#include "wave_memcpy.h"
/*^*/

void forward_modeling_a(float ***data, np_sou soupar, np_acqui acpar, np_vec array, bool verb)
/*< acoustic forward modeling >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx2, dz2, dt2;
	float **vv, **dd;
	float **p0, **p1, **p2, **term, **tmparray, *rr;
	
	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx2=acpar->dx*acpar->dx;
	dz2=acpar->dz*acpar->dz;
	dt2=acpar->dt*acpar->dt;

	vv = np_floatalloc2(padnz, padnx);
	dd=np_floatalloc2(nt, nr);

	p0=np_floatalloc2(padnz, padnx);
	p1=np_floatalloc2(padnz, padnx);
	p2=np_floatalloc2(padnz, padnx);
	term=np_floatalloc2(padnz, padnx);
	rr=np_floatalloc(padnzx);

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);

    for(is=0; is<acpar->ns; is++){
		printf("###### is=%d ######\n", is+1);

		memset(dd[0], 0., nr*nt*sizeof(float));
		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
// 			if(verb) printf("Modeling is=%d; it=%d; \n", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				dd[acpar->r02[is]+ir][it]=p1[rx][rz];
			}

			/* laplacian operator */
			laplace(p1, term, padnx, padnz, dx2, dz2);
			
			/* load source */
			for(ix=0; ix<padnx; ix++){
				for(iz=0; iz<padnz; iz++){
					term[ix][iz] += rr[ix*padnz+iz]*array->ww[it];
				}
			}

			/* update */
			for(ix=0; ix<padnx; ix++){
				for(iz=0; iz<padnz; iz++){
					p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
				}
			}
			
			/* swap wavefield pointer of different time steps */
			tmparray=p0; p0=p1; p1=p2; p2=tmparray;

			/* boundary condition */
			apply_sponge(p0, acpar->bc, padnx, padnz, nb);
			apply_sponge(p1, acpar->bc, padnx, padnz, nb);
		} // end of time loop

		mcp(data[is][0],dd[0],0,0,nr*nt); /*destination*/
	}// end of shot loop
	
	/* release allocated memory */
	free(*p0); free(p0); free(*p1); free(p1);
	free(*p2); free(p2); free(*vv); free(vv);
	free(*dd); free(dd);
	free(rr); free(*term); free(term);
}

void forward_modeling(float ***data, np_sou soupar, np_acqui acpar, np_vec array, bool verb)
/*< visco-acoustic forward modeling >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx2, dz2, dt2, idt;
	float **vv, **tau, **taus, **dd;
	float **p0, **p1, **p2, **r1, **r2, **term, **tmparray, *rr;

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx2=acpar->dx*acpar->dx;
	dz2=acpar->dz*acpar->dz;
	dt2=acpar->dt*acpar->dt;
	idt=1./acpar->dt;

	vv = np_floatalloc2(padnz, padnx);
	tau= np_floatalloc2(padnz, padnx);
	taus=np_floatalloc2(padnz, padnx);
	dd=np_floatalloc2(nt, nr);

	p0=np_floatalloc2(padnz, padnx);
	p1=np_floatalloc2(padnz, padnx);
	p2=np_floatalloc2(padnz, padnx);
	r1=np_floatalloc2(padnz, padnx);
	r2=np_floatalloc2(padnz, padnx);
	term=np_floatalloc2(padnz, padnx);
	rr=np_floatalloc(padnzx);

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);
	
    for(is=0; is<acpar->ns; is++){
		printf("###### is=%d ######", is+1);

		memset(dd[0], 0., nr*nt*sizeof(float));
		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(r1[0], 0., padnzx*sizeof(float));
		memset(r2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
// 			if(verb) printf("Modeling is=%d; it=%d;\n", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				dd[acpar->r02[is]+ir][it]=p1[rx][rz];
			}

			/* laplacian operator */
			laplace(p1, term, padnx, padnz, dx2, dz2);

			/* calculate r, load source and update wavefield */
			for(ix=0; ix<padnx; ix++){
				for(iz=0; iz<padnz; iz++){
					r2[ix][iz]=
						(tau[ix][iz]/taus[ix][iz]*term[ix][iz]
						 + (idt-0.5/taus[ix][iz])*r1[ix][iz])
						/(idt+0.5/taus[ix][iz]);
					term[ix][iz]=term[ix][iz]*(1.+tau[ix][iz]) - (r2[ix][iz]+r1[ix][iz])*0.5 + rr[ix*padnz+iz]*array->ww[it];
					p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
				}
			}
			
			/* swap wavefield pointer of different time steps */
			tmparray=p0; p0=p1; p1=p2; p2=tmparray;
			tmparray=r1; r1=r2; r2=tmparray;

			/* boundary condition */
			apply_sponge(p0, acpar->bc, padnx, padnz, nb);
			apply_sponge(p1, acpar->bc, padnx, padnz, nb);
			apply_sponge(r1, acpar->bc, padnx, padnz, nb);
		} // end of time loop

		mcp(data[is][0],dd[0],0,0,nr*nt); /*destination*/
	}// end of shot loop
	
	/* release allocated memory */
	free(*p0); free(p0); free(*p1); free(p1);
	free(*p2); free(p2); free(*r1); free(r1);
	free(*r2); free(r2); free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*dd); free(dd); free(rr); 
	free(*term); free(term);
}
