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
#include "wave_fwiutil.h"
#include "wave_triangle.h"
#include "wave_decart.h"

#define np_PI (3.14159265358979323846264338328)

typedef void (*np_gradient)(float*,float*,float*);
/*^*/

const float c0=-205./72, c1=8./5, c2=-1./5, c3=8./315, c4=-1./560;

void preparation(float *vv, float *qq, float *ww, np_acqui acpar, np_sou soupar, np_vec array)
/*< read data, initialize variables and prepare acquisition geometry >*/
{
	int i, nb, nzx;
	float sx, xend, rbegin, rend;
	float *taue, tmp;

	int nplo=3, nphi=3, nt;
	float eps=0.0001;
// 	np_butter blo=NULL, bhi=NULL;

	/* absorbing boundary coefficients */
	nb=acpar->nb;
	acpar->bc=np_floatalloc(nb);
	for(i=0; i<nb; i++){
		tmp=acpar->coef*(nb-i);
		acpar->bc[i]=expf(-tmp*tmp);
	}

	/* padding variables */
	acpar->padnx=acpar->nx+2*nb;
	acpar->padnz=acpar->nz+2*nb;
	acpar->padx0=acpar->x0-nb*acpar->dx;
	acpar->padz0=acpar->z0-nb*acpar->dz;

	/* acquisition parameters */
	acpar->ds_v=acpar->ds/acpar->dx+0.5;
	acpar->s0_v=acpar->s0/acpar->dx+0.5+nb;
	acpar->sz += nb;

	acpar->dr_v=acpar->dr/acpar->dx+0.5;
	acpar->r0_v=np_intalloc(acpar->ns);
	acpar->r02=np_intalloc(acpar->ns);
	acpar->nr2=np_intalloc(acpar->ns);
	acpar->rz += nb;
	xend=acpar->x0+(acpar->nx-1)*acpar->dx;
	if(acpar->acqui_type==1){
		for(i=0; i<acpar->ns; i++){
			acpar->r0_v[i]=(acpar->r0-acpar->x0)/acpar->dx+0.5+nb;
			acpar->r02[i]=0;
			acpar->nr2[i]=acpar->nr;
		}
	}else{
		for(i=0; i<acpar->ns; i++){
			sx=acpar->s0+acpar->ds*i;
			rbegin=(sx+acpar->r0 <acpar->x0)? acpar->x0 : sx+acpar->r0;
			rend=sx+acpar->r0 +(acpar->nr-1)*acpar->dr;
			rend=(rend < xend)? rend : xend;
			acpar->r0_v[i]=rbegin/acpar->dx+0.5+nb;
			acpar->r02[i]=(rbegin-sx-acpar->r0)/acpar->dx+0.5;
			acpar->nr2[i]=(rend-rbegin)/acpar->dx+1.5;
		}
	}

	/* read model parameters */
	nzx=acpar->nz*acpar->nx;
	nt=acpar->nt;
// 	array->vv=np_floatalloc(nzx);
// 	array->qq=np_floatalloc(nzx);
	array->tau=np_floatalloc(nzx);
	array->taus=np_floatalloc(nzx);
// 	array->ww=np_floatalloc(nt);
	taue=np_floatalloc(nzx);

// 	np_floatread(array->vv, nzx, vv);
// 	np_floatread(array->qq, nzx, qq);
// 	np_floatread(array->ww, nt, ww);
	
	array->vv=vv;
	array->qq=qq;
	array->ww=ww;
	
	/* calculate tau */
	for(i=0; i<nzx; i++){
		tmp=sqrtf(array->qq[i]*array->qq[i]+1);
		taue[i]=(tmp+1)/(2.*np_PI*acpar->f0*array->qq[i]);
		array->taus[i]=(tmp-1)/(2.*np_PI*acpar->f0*array->qq[i]);
		array->tau[i]=taue[i]/array->taus[i]-1;
	}

	/* bandpass the wavelet */
	soupar->flo *= acpar->dt;
	soupar->fhi *= acpar->dt;
// 	if(soupar->flo > eps) blo=np_butter_init(false, soupar->flo, nplo);
// 	if(soupar->fhi < 0.5-eps) bhi=np_butter_init(true, soupar->fhi, nphi);

// 	if(NULL != blo){
// 		np_butter_apply(blo, nt, array->ww);
// 		np_reverse(nt, array->ww);
// 		np_butter_apply(blo, nt, array->ww);
// 		np_reverse(nt, array->ww);
// 		np_butter_close(blo);
// 	}
// 	if(NULL != bhi){
// 		np_butter_apply(bhi, nt, array->ww);
// 		np_reverse(nt, array->ww);
// 		np_butter_apply(bhi, nt, array->ww);
// 		np_reverse(nt, array->ww);
// 		np_butter_close(bhi);
// 	}
	
	free(taue);
}

void pad2d(float *vec, float **array, int nz, int nx, int nb)
/*< convert a vector to an array >*/
{
	int ix, iz;
	
	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			array[ix+nb][iz+nb]=vec[ix*nz+iz];
		}
	}

    for (ix=nb; ix<nx+nb; ix++){
		for (iz=0; iz<nb; iz++){
			array[ix][iz]=array[ix][nb];
			array[ix][iz+nz+nb]=array[ix][nz+nb-1];
		}
	}

	for (ix=0; ix<nb; ix++){
		for (iz=0; iz<nz+2*nb; iz++){
			array[ix][iz]=array[nb][iz];
			array[ix+nx+nb][iz]=array[nx+nb-1][iz];
		}
	}
}

void source_map(int sx, int sz, int rectx, int rectz, int padnx, int padnz, int padnzx, float *rr)
/*< generate source map >*/
{
	int i, j, i0;
	int n[2], s[2], rect[2];
	bool diff[2], box[2];
	np_triangle tr;

	n[0]=padnz; n[1]=padnx;
	s[0]=1; s[1]=padnz;
	rect[0]=rectz; rect[1]=rectx;
	diff[0]=false; diff[1]=false;
	box[0]=false; box[1]=false;

	for (i=0; i<padnzx; i++)
		rr[i]=0.;
	j=sx*padnz+sz;
	rr[j]=1.;

	for (i=0; i<2; i++){
		if(rect[i] <=1) continue;
		tr=np_triangle_init(rect[i], n[i], box[i]);
		for(j=0; j<padnzx/n[i]; j++){
			i0=np_first_index(i,j,2,n,s);
			np_smooth2(tr,i0,s[i],diff[i],rr);
		}
		np_triangle_close(tr);
	}
}

void laplace(float **p1, float **term, int padnx, int padnz, float dx2, float dz2)
/*< laplace operator >*/
{
	int ix, iz;

	for (ix=4; ix<padnx-4; ix++){
		for (iz=4; iz<padnz-4; iz++){
			term[ix][iz] = 
				(c0*p1[ix][iz]
				+c1*(p1[ix+1][iz]+p1[ix-1][iz])
				+c2*(p1[ix+2][iz]+p1[ix-2][iz])
				+c3*(p1[ix+3][iz]+p1[ix-3][iz])
				+c4*(p1[ix+4][iz]+p1[ix-4][iz]))/dx2 
				+(c0*p1[ix][iz]
				+c1*(p1[ix][iz+1]+p1[ix][iz-1])
				+c2*(p1[ix][iz+2]+p1[ix][iz-2])
				+c3*(p1[ix][iz+3]+p1[ix][iz-3])
				+c4*(p1[ix][iz+4]+p1[ix][iz-4]))/dz2;
		}
	}
}

void apply_sponge(float **p, float *bc, int padnx, int padnz, int nb)
/*< apply absorbing boundary condition >*/
{
	int ix, iz;

	for (ix=0; ix<padnx; ix++){
		for(iz=0; iz<nb; iz++){	// top ABC
			p[ix][iz]=bc[iz]*p[ix][iz];
		}
		for(iz=padnz-nb; iz<padnz; iz++){ // bottom ABC			
			p[ix][iz]=bc[padnz-iz-1]*p[ix][iz];
		} 
	}

	for (iz=0; iz<padnz; iz++){
		for(ix=0; ix<nb; ix++){ // left ABC			
			p[ix][iz]=bc[ix]*p[ix][iz];
		}
		for(ix=padnx-nb; ix<padnx; ix++){ // right ABC			
			p[ix][iz]=bc[padnx-ix-1]*p[ix][iz];
		}
	}
}

void residual_weighting(float **ww, int nt, int nr, int wtn1, int wtn2, int woffn1, int woffn2, bool oreo)
/*< data residual weighting >*/
{
	int it, ir;
	float w[10];

	for(it=0; it<10; it++){
		w[it]=sinf(0.5*np_PI*(it+1)/11.);
	}

        if (oreo) {

            for(ir=0; ir<nr; ir++){
                for(it=0; it<nt; it++){
                    ww[ir][it]=1.;
                }
            }

            for(ir=woffn1; ir<=woffn2; ir++){
                for(it=wtn1; it<=wtn2; it++){
                    ww[ir][it]=0.;
                }
            }

            for(ir=0; ir<10; ir++){
                for(it=wtn1; it<=wtn2; it++){
                    if(woffn1-ir>0 ) ww[woffn1-ir][it] *= w[ir];
                    if(woffn2+ir<nr) ww[woffn2+ir][it] *= w[ir];
                }
            }

            for(it=0; it<10; it++){
                for(ir=0; ir<woffn1; ir++){
                    ww[ir][wtn1+it] *= w[it];
                    ww[ir][wtn2-it] *= w[it];
                }
                for(ir=woffn2+1; ir<nr; ir++){
                    ww[ir][wtn1+it] *= w[it];
                    ww[ir][wtn2-it] *= w[it];
                }
            }

        } else {

            for(ir=0; ir<nr; ir++){
                for(it=0; it<nt; it++){
                    ww[ir][it]=0.;
                }
            }

            for(ir=woffn1; ir<=woffn2; ir++){
                for(it=wtn1; it<=wtn2; it++){
                    ww[ir][it]=1.;
                }
            }

            for(ir=0; ir<10; ir++){
                for(it=wtn1; it<=wtn2; it++){
                    ww[woffn1+ir][it] *= w[ir];
                    ww[woffn2-ir][it] *= w[ir];
                }
            }

            for(it=0; it<10; it++){
                for(ir=woffn1; ir<=woffn2; ir++){
                    ww[ir][wtn1+it] *= w[it];
                    ww[ir][wtn2-it] *= w[it];
                }
            }

        }
}

void gradient_smooth2(int rectx, int rectz, int nx, int nz, int waterz, float scaling, float *grad)
/*< smooth gradient, zero bathymetry layer and normalization >*/
{
	int i, j, i0, nzx;
	int n[2], s[2], rect[2];
	bool diff[2], box[2];
	np_triangle tr;

	nzx=nz*nx;
	n[0]=nz; n[1]=nx;
	s[0]=1; s[1]=nz;
	rect[0]=rectz; rect[1]=rectx;
	diff[0]=false; diff[1]=false;
	box[0]=false; box[1]=false;

	for (i=0; i<2; i++){
		if(rect[i] <=1) continue;
		tr=np_triangle_init(rect[i], n[i], box[i]);
		for(j=0; j<nzx/n[i]; j++){
			i0=np_first_index(i,j,2,n,s);
			np_smooth2(tr,i0,s[i],diff[i],grad);
		}
		np_triangle_close(tr);
	}

        if (waterz>=0) {
            for(i=0; i<waterz; i++)
                for(j=0; j<nx; j++)
                    grad[i+j*nz]=0.;
        } else {
            for(i=nz+waterz; i<nz; i++)
                for(j=0; j<nx; j++)
                    grad[i+j*nz]=0.;
        }

	for(i=0; i<nzx; i++)
		grad[i] *= scaling;
}

void gradient_smooth2b(int rectx, int rectz, int nx, int nz, int waterz, int waterzb, float scaling, float *grad)
/*< smooth gradient, zero bathymetry layer and normalization >*/
{
	int i, j, i0, nzx;
	int n[2], s[2], rect[2];
	bool diff[2], box[2];
	np_triangle tr;

	nzx=nz*nx;
	n[0]=nz; n[1]=nx;
	s[0]=1; s[1]=nz;
	rect[0]=rectz; rect[1]=rectx;
	diff[0]=false; diff[1]=false;
	box[0]=false; box[1]=false;

	for (i=0; i<2; i++){
		if(rect[i] <=1) continue;
		tr=np_triangle_init(rect[i], n[i], box[i]);
		for(j=0; j<nzx/n[i]; j++){
			i0=np_first_index(i,j,2,n,s);
			np_smooth2(tr,i0,s[i],diff[i],grad);
		}
		np_triangle_close(tr);
	}

        for(i=0; i<waterz; i++)
            for(j=0; j<nx; j++)
                grad[i+j*nz]=0.;

        for(i=nz-waterzb; i<nz; i++)
            for(j=0; j<nx; j++)
                grad[i+j*nz]=0.;

	for(i=0; i<nzx; i++)
		grad[i] *= scaling;
}

void l2norm(int n, float *a, float *norm)
/*< L2 norm of a vector >*/
{
	int i;
	*norm=0.;
	for(i=0; i<n; i++){
		*norm += a[i]*a[i];
	}
	*norm=sqrtf(*norm);
}

void reverse(int n, float *a, float *b)
/*< reverse the sign of the vector >*/
{
	int i;
	for (i=0; i<n; i++)
		b[i]=-a[i];
}

void copy(int n, float *a, float *b)
/*< copy vector >*/
{
	int i;
	for(i=0; i<n; i++)
		b[i]=a[i];
}

void dot_product(int n, float *a, float *b, float *product)
/*< dot product of two vectors >*/
{
	int i;
	*product=0.;
	for (i=0; i<n; i++)
		*product += a[i]*b[i];
}

// void print_iteration(FILE *fp, int iter, np_optim opt)
// /*< print out iteration information >*/
// {
// 	if(iter%10==0){
// 		fprintf(fp,"*********************************************\n");
// 		fprintf(fp,"Maximum Iteration: %d\n", opt->niter);
// 		fprintf(fp,"Convergence Error: %3.2e\n", opt->conv_error);
// 		fprintf(fp,"*********************************************\n");
// 		fprintf(fp,"Niter  Misfit  Rel_Misfit  Grad_Norm  Alpha   Num_Pair  Num_LS  Total_Grad\n");
// 	}
// 	fprintf(fp,"%3d   %3.2e  %3.2e   %3.2e  %3.2e  %3d       %3d      %4d\n", iter, opt->fk, opt->fk/opt->f0, opt->gk_norm, opt->alpha, opt->ipair, opt->ils, opt->igrad);
// }
// 
// np_pas passive_init(np_acqui acpar)
// /*< read data, initialize variables and prepare acquisition geometry >*/
// {
//     np_pas paspar;
// 
//     paspar = (np_pas) np_alloc(1,sizeof(*paspar));
// 
//     if(!np_getbool("inv", &paspar->inv)) paspar->inv=false; /* inversion flag */
//     if(!np_getbool("onlysrc", &paspar->onlysrc)) paspar->onlysrc=false;  /* only invert for source (vel known), active when inv=y */
//     if(!np_getbool("onlyvel", &paspar->onlyvel)) paspar->onlyvel=false;  /* only invert for vel (source known), active when inv=y */
//     if(!np_getbool("sw", &paspar->sw)) paspar->sw=false;  /* sliding window normalization */
//     if(!np_getbool("ctr", &paspar->ctr)) paspar->ctr=false; /* cross-correlation time-reversal imaging */
//     if(!np_getbool("precsrc", &paspar->prec)) paspar->prec=false; /* source inversion preconditioning */
//     if(!np_getbool("hidesrc", &paspar->hidesrc)) paspar->hidesrc=false; /* hide source footprint in fwi */
//     if(!np_getint("nitersrc", &paspar->niter)) paspar->niter=1;   /* num of iter'ns for source inversion */
//     if(!np_getint("ngrp", &paspar->ngrp))   paspar->ngrp=1; /* number of sub-groups of receivers */
//     if(!np_getint("size", &paspar->size))   paspar->size=0; /* sliding window radius */
//     if(!np_getint("rectzsrc", &paspar->rectz)) paspar->rectz=1; /* source smoothing in z before masking */
//     if(!np_getint("rectxsrc", &paspar->rectx)) paspar->rectx=1; /* source smoothing in x before masking */
//     if(!np_getint("recttsrc", &paspar->rectt)) paspar->rectt=50; /* source smoothing in t before masking */
//     if(!np_getint("repeatsrc", &paspar->repeat)) paspar->repeat=1; /* source smoothing repeatation times */
//     if(!np_getfloat("perc", &paspar->perc)) paspar->perc=1.0f; /* padding percentatge for swnorm */
//     if(!np_getfloat("hard", &paspar->hard)) paspar->hard=0.1f; /* hard thresholding for masking */
// 
//     if(paspar->onlyvel && paspar->onlysrc) np_error("Error: onlyvel and onlysrc cannot both be true!");
//     if(paspar->inv) {
//         if(paspar->onlyvel) np_warning("inverting for velocity only ...");
//         else if(paspar->onlysrc) np_warning("inverting for source only ...");
//         else np_warning("inverting for both velocity and source ...");
//     } else np_warning("forward modeling ...");
// 
//     return paspar;
// }
