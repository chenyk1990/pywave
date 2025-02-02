#include "wave_bigsolver.h"
#include "wave_triutil.h"
#include "wave_fwiutil.h"
#include "wave_cgstep.h"
#include "wave_alloc.h"
#include "wave_butter.h"

static bool verb, first;
static int nz, nx, nzx, padnz, padnx, padnzx, nb, nt;
static int ns, ds_v, s0_v, sz, nr, dr_v, rz;
static int *nr2, *r02, *r0_v;
static int rectx, rectz, grectx, grectz, interval, wnt;
static int waterz, waterzb, wtn1, wtn2, woffn1, woffn2;

static float dt, idt, dt2, dx2, dz2, wdt, wdt2, scaling;
static float wt1, wt2, woff1, woff2;
static float ***dd, **vv, **tau, **taus, *ww, *bc, **weight;


void gradient_init(float ***data, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, bool verb1)
/*< initialize >*/
{
	int is;

	verb=verb1;
	first=true; // only at the first iteration, need to calculate the gradient scaling parameter

	nz=acpar->nz;
	nx=acpar->nx;
	nzx=nz*nx;
	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nb=acpar->nb;
	nt=acpar->nt;

	ns=acpar->ns;
	ds_v=acpar->ds_v;
	s0_v=acpar->s0_v;
	sz=acpar->sz;
	nr=acpar->nr;
	dr_v=acpar->dr_v;
	nr2=acpar->nr2;
	r02=acpar->r02;
	r0_v=acpar->r0_v;
	rz=acpar->rz;

	rectx=soupar->rectx;
	rectz=soupar->rectz;
	grectx=fwipar->rectx;
	grectz=fwipar->rectz;
	interval=acpar->interval;
	wnt=(nt-1)/interval+1;

	dt=acpar->dt;
	idt=1./dt;
	dt2=dt*dt;
	wdt=dt*interval;
	wdt2=wdt*wdt;
	dx2=acpar->dx*acpar->dx;
	dz2=acpar->dz*acpar->dz;

	wt1=fwipar->wt1;
	wt2=fwipar->wt2;
	woff1=fwipar->woff1;
	woff2=fwipar->woff2;
	waterz=fwipar->waterz;

	ww=array->ww;
	bc=acpar->bc;
	
	dd=data;
	
	/* data residual weights */
	wtn1=(wt1-acpar->t0)/dt+0.5;
	wtn2=(wt2-acpar->t0)/dt+0.5;
	woffn1=(woff1-acpar->r0)/acpar->dr+0.5;
	woffn2=(woff2-acpar->r0)/acpar->dr+0.5;
	weight=np_floatalloc2(nt, nr);
	residual_weighting(weight, nt, nr, wtn1, wtn2, woffn1, woffn2, fwipar->oreo);

	/* padding and convert vector to 2-d array */
	vv = np_floatalloc2(padnz, padnx);
	tau= np_floatalloc2(padnz, padnx);
	taus=np_floatalloc2(padnz, padnx);
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	return;
}

void gradient_av(float *x, float *fcost, float *grad)
/*< acoustic velocity gradient >*/
{
	int ix, iz, is, ir, it, wit;
	int sx, rx;

	float temp, dmax;
	float **p0, **p1, **p2, **term, **tmparray, *rr, ***wave, **pp;
	float *sendbuf, *recvbuf;

	/* initialize fcost */
	*fcost=0.;
	/* update velocity */
	pad2d(x, vv, nz, nx, nb);
	/* initialize gradient */
	memset(grad, 0., nzx*sizeof(float));

	/* memory allocation */
	p0=np_floatalloc2(padnz, padnx);
	p1=np_floatalloc2(padnz, padnx);
	p2=np_floatalloc2(padnz, padnx);
	term=np_floatalloc2(padnz, padnx);
	rr=np_floatalloc(padnzx);
	wave=np_floatalloc3(nz, nx, wnt);
	pp=np_floatalloc2(nt, nr);

	for(is=0; is<ns; is++){
		printf("###### is=%d ######\n", is+1);

		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(pp[0], 0., nr*nt*sizeof(float));
		
		sx=s0_v+is*ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		wit=0;
		/* forward propagation */
		for(it=0; it<nt; it++){
// 			if(verb) printf("Forward propagation is=%d; it=%d;\n", is+1, it);

			/* output predicted data */
			for(ir=0; ir<nr2[is]; ir++){
				rx=r0_v[is]+ir*dr_v;
				pp[r02[is]+ir][it]=p1[rx][rz];
			}

			/* save wavefield */
			if(it%interval==0){
				for(ix=0; ix<nx; ix++)
					for(iz=0; iz<nz; iz++)
						wave[wit][ix][iz]=p1[ix+nb][iz+nb];
				wit++;
			}

			/* laplacian operator */
			laplace(p1, term, padnx, padnz, dx2, dz2);
			
			/* load source */
			for(ix=0; ix<padnx; ix++){
				for(iz=0; iz<padnz; iz++){
					term[ix][iz] += rr[ix*padnz+iz]*ww[it];
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
			apply_sponge(p0, bc, padnx, padnz, nb);
			apply_sponge(p1, bc, padnx, padnz, nb);
		} // end of time loop

		/* check */
		if(wit != wnt) printf("Incorrect number of wavefield snapshots\n");
		wit--;
		
		/* calculate data residual and data misfit */
		for(ir=0; ir<nr; ir++){
			for(it=0; it<nt; it++){
				pp[ir][it]=dd[is][ir][it]-pp[ir][it];
				*fcost += 0.5*pp[ir][it]*pp[ir][it];
			}
		}
		
		/* window the data residual */
		for(ir=0; ir<nr; ir++){
			for(it=0; it<nt; it++){
				pp[ir][it] *= weight[ir][it];
			}
		}
// 		is++;

		/* initialization */
		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
                memset(term[0], 0., padnzx*sizeof(float));
		
		/* backward propagation */
		for(it=nt-1; it>=0; it--){
// 			if(verb) printf("Backward propagation is=%d; it=%d;\n", is+1, it);
			
			/* laplacian operator */
			laplace(p1, term, padnx, padnz, dx2, dz2);
			
			/* load data residual*/
			for(ir=0; ir<nr2[is]; ir++){
				rx=r0_v[is]+ir*dr_v;
				term[rx][rz] += pp[r02[is]+ir][it];
			}

			/* update */
			for(ix=0; ix<padnx; ix++){
				for(iz=0; iz<padnz; iz++){
					p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
				}
			}
			
			/* calculate gradient  */
			if(it%interval==0){
				if(wit != wnt-1 && wit != 0){ // avoid the first and last time step
					for(ix=0; ix<nx; ix++){
						for(iz=0; iz<nz; iz++){
							temp=vv[ix+nb][iz+nb];
							temp=temp*temp*temp;
							temp=-2./temp;
							grad[ix*nz+iz] += (wave[wit+1][ix][iz]-2.*wave[wit][ix][iz]+wave[wit-1][ix][iz])/wdt2*p1[ix+nb][iz+nb]*temp;
						}
					}
				}
				wit--;
			}
			
			/* swap wavefield pointer of different time steps */
			tmparray=p0; p0=p1; p1=p2; p2=tmparray;

			/* boundary condition */
			apply_sponge(p0, bc, padnx, padnz, nb);
			apply_sponge(p1, bc, padnx, padnz, nb);
		} // end of time loop
	}// end of shot loop

	/* scaling gradient */
	if(first){
		dmax=0.;
		for(ix=0; ix<nzx; ix++)
			if(fabsf(grad[ix])>dmax)
				dmax=fabsf(grad[ix]);
		scaling=0.1/dmax;
		first=false;
	}

	/* smooth gradient */
	gradient_smooth2(grectx, grectz, nx, nz, waterz, scaling, grad);

	/* free allocated memory */
	free(*p0); free(p0); free(*p1); free(p1);
	free(*p2); free(p2); free(*pp); free(pp);
	free(**wave); free(*wave); free(wave);
	free(rr); free(*term); free(term);
}

void gradient_v(float *x, float *fcost, float *grad)
/*< velocity gradient >*/
{
	int ix, iz, is, ir, it, wit;
	int sx, rx;

	float temp, dmax;
	float **p0, **p1, **p2, **r1, **r2, **term, **tmp, **tmparray, *rr, ***wave, **pp;
	float *sendbuf, *recvbuf;

	/* initialize fcost */
	*fcost=0.;
	/* update velocity */
	pad2d(x, vv, nz, nx, nb);
	/* initialize gradient */
	memset(grad, 0., nzx*sizeof(float));

	/* memory allocation */
	p0=np_floatalloc2(padnz, padnx);
	p1=np_floatalloc2(padnz, padnx);
	p2=np_floatalloc2(padnz, padnx);
	r1=np_floatalloc2(padnz, padnx);
	r2=np_floatalloc2(padnz, padnx);
	tmp=np_floatalloc2(padnz, padnx);
	term=np_floatalloc2(padnz, padnx);
	rr=np_floatalloc(padnzx);
	wave=np_floatalloc3(nz, nx, wnt);
	pp=np_floatalloc2(nt, nr);

	for(is=0; is<ns; is++){
		printf("###### is=%d ######\n", is+1);

		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(r1[0], 0., padnzx*sizeof(float));
		memset(r2[0], 0., padnzx*sizeof(float));
		memset(pp[0], 0., nr*nt*sizeof(float));
		memset(term[0], 0., padnzx*sizeof(float));
		memset(tmp[0], 0., padnzx*sizeof(float));
		
		sx=s0_v+is*ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		wit=0;
		/* forward propagation */
		for(it=0; it<nt; it++){
// 			if(verb) printf("Forward propagation is=%d; it=%d;", is+1, it);

			/* output predicted data */
			for(ir=0; ir<nr2[is]; ir++){
				rx=r0_v[is]+ir*dr_v;
				pp[r02[is]+ir][it]=p1[rx][rz];
			}

			/* save wavefield */
			if(it%interval==0){
				for(ix=0; ix<nx; ix++)
					for(iz=0; iz<nz; iz++)
						wave[wit][ix][iz]=p1[ix+nb][iz+nb];
				wit++;
			}

			/* laplacian operator */
			laplace(p1, term, padnx, padnz, dx2, dz2);
			
			/* calculate r, load source and update wavefield */
			for(ix=4; ix<padnx-4; ix++){
				for(iz=4; iz<padnz-4; iz++){
					r2[ix][iz]=
						(tau[ix][iz]/taus[ix][iz]*term[ix][iz]
						 + (idt-0.5/taus[ix][iz])*r1[ix][iz])
						/(idt+0.5/taus[ix][iz]);
					term[ix][iz]=term[ix][iz]*(1.+tau[ix][iz]) - (r2[ix][iz]+r1[ix][iz])*0.5 + rr[ix*padnz+iz]*ww[it];
					p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
				}
			}
			
			/* swap wavefield pointer of different time steps */
			tmparray=p0; p0=p1; p1=p2; p2=tmparray;
			tmparray=r1; r1=r2; r2=tmparray;

			/* boundary condition */
			apply_sponge(p0, bc, padnx, padnz, nb);
			apply_sponge(p1, bc, padnx, padnz, nb);
			apply_sponge(r1, bc, padnx, padnz, nb);
		} // end of time loop

		/* check */
		if(wit != wnt) printf("Incorrect number of wavefield snapshots\n");
		wit--;
		
		/* calculate data residual and data misfit */
		for(ir=0; ir<nr; ir++){
			for(it=0; it<nt; it++){
				pp[ir][it]=dd[is][ir][it]-pp[ir][it];
				*fcost += 0.5*pp[ir][it]*pp[ir][it];
				pp[ir][it] *= weight[ir][it];
			}
		}
// 		is++;

		/* initialization */
		memset(p0[0], 0., padnzx*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(r1[0], 0., padnzx*sizeof(float));
		memset(r2[0], 0., padnzx*sizeof(float));
		memset(term[0], 0., padnzx*sizeof(float));
		memset(tmp[0], 0., padnzx*sizeof(float));
		
		/* backward propagation */
		for(it=nt-1; it>=0; it--){
// 			if(verb) printf("Backward propagation is=%d; it=%d;", is+1, it);

			/* calculate and load r term */
			for(ix=4; ix<padnx-4; ix++){
				for(iz=4; iz<padnz-4; iz++){
					r2[ix][iz]=
						(-tau[ix][iz]/taus[ix][iz]*p1[ix][iz]
						 + (-idt+0.5/taus[ix][iz])*r1[ix][iz])
						/(-idt-0.5/taus[ix][iz]);
					tmp[ix][iz]=p1[ix][iz]*(1.+tau[ix][iz]) - 0.5*(r2[ix][iz]+r1[ix][iz]);
				}
			}

			/* laplacian operator */
			laplace(tmp, term, padnx, padnz, dx2, dz2);
			
			/* load data residual*/
			for(ir=0; ir<nr2[is]; ir++){
				rx=r0_v[is]+ir*dr_v;
				term[rx][rz] += pp[r02[is]+ir][it];
			}

			/* update */
			for(ix=4; ix<padnx-4; ix++){
				for(iz=4; iz<padnz-4; iz++){
					p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
				}
			}
			
			/* calculate gradient  */
			if(it%interval==0){
				if(wit != wnt-1 && wit != 0){ // avoid the first and last time step
					for(ix=0; ix<nx; ix++){
						for(iz=0; iz<nz; iz++){
							temp=vv[ix+nb][iz+nb];
							temp=temp*temp*temp;
							temp=-2./temp;
							grad[ix*nz+iz] += (wave[wit+1][ix][iz]-2.*wave[wit][ix][iz]+wave[wit-1][ix][iz])/wdt2*p1[ix+nb][iz+nb]*temp;
						}
					}
				}
				wit--;
			}
			
			/* swap wavefield pointer of different time steps */
			tmparray=p0; p0=p1; p1=p2; p2=tmparray;
			tmparray=r1; r1=r2; r2=tmparray;

			/* boundary condition */
			apply_sponge(p0, bc, padnx, padnz, nb);
			apply_sponge(p1, bc, padnx, padnz, nb);
			apply_sponge(r1, bc, padnx, padnz, nb);
		} // end of time loop
	}// end of shot loop

	/* scaling gradient */
	if(first){
		dmax=0.;
		for(ix=0; ix<nzx; ix++)
			if(fabsf(grad[ix])>dmax)
				dmax=fabsf(grad[ix]);
		scaling=0.1/dmax;
		first=false;
	}

	/* smooth gradient */
	gradient_smooth2(grectx, grectz, nx, nz, waterz, scaling, grad);

	/* free allocated memory */
	free(*p0); free(p0); free(*p1); free(p1);
	free(*p2); free(p2); free(*pp); free(pp);
	free(*r1); free(r1); free(*r2); free(r2);
	free(**wave); free(*wave); free(wave);
	free(rr); free(*term); free(term);
	free(*tmp); free(tmp);
}

void lstri_op(float **dd, float **dwt, float ***ww, float ***mwt, np_acqui acpar, np_vec array, np_pas paspar, bool verb)
/*< ls TRI operator >*/
{
    float **vv1;
    int ix,iz,it;


    if (paspar->inv) {
        for         (it=0; it<acpar->nt; it++)
            for     (ix=0; ix<acpar->nx; ix++)
                for (iz=0; iz<acpar->nz; iz++)
                    ww[it][ix][iz] = 0.0f;
        if (NULL!=mwt) {
            for         (it=0; it<acpar->nt; it++)
                for     (ix=0; ix<acpar->nx; ix++)
                    for (iz=0; iz<acpar->nz; iz++)
                        mwt[it][ix][iz] = 1.0f;
        }
    } else {
        for     (ix=0; ix<acpar->nx; ix++)
            for (it=0; it<acpar->nt; it++)
                dd[ix][it] = 0.0f;
    }
	printf("before timerev0\n");
    /* map 1d to 2d */
    vv1 = (float**) np_alloc (acpar->nx,sizeof(float*)); 
    vv1[0] = array->vv;
    for (ix=1; ix<acpar->nx; ix++) vv1[ix] = vv1[0]+ix*acpar->nz;

	printf("before timerev\n");
    timerev_init(verb, true, acpar->nt, acpar->nx, acpar->nz, acpar->nb, acpar->rz-acpar->nb, acpar->dt, acpar->dx, acpar->dz, acpar->coef, vv1);

	printf("before timerev2\n");
    /* calculate model weighting using correlative imaging condition */
//     if (paspar->inv && paspar->prec) { 
//         if (paspar->ctr) {
//             ctimerev(paspar->ngrp,mwt,dd);
//             absval(acpar->nz*acpar->nx*acpar->nt,mwt[0][0]);
//         } else {
//             timerev_lop(true, false, acpar->nz*acpar->nx*acpar->nt, acpar->nt*acpar->nx, mwt[0][0], dd[0]);
//             autopow(acpar->nz*acpar->nx*acpar->nt,(float)paspar->ngrp,mwt[0][0]);
//         }
//         /* smoothing */
//         smooth(acpar->nz, acpar->nx, acpar->nt, paspar->rectz, paspar->rectx, paspar->rectt, paspar->repeat, mwt[0][0]);
//         /* local normalizaiton */
//         swnorm(verb, paspar->sw, acpar->nz, acpar->nx, acpar->nt, paspar->size, paspar->perc, mwt[0][0]);
//         /* hard thresholding */
//         if (paspar->hard>0) threshold(false, acpar->nz*acpar->nx*acpar->nt, paspar->hard, mwt[0][0]);
//     }

    /* apply time-reversal imaging linear operator */
    if (paspar->inv) {
    	printf("In FWIgradient, niter=%d\n",paspar->niter);
//     	paspar->niter=10;
        if (NULL!=dwt) np_solver(timerev_lop,np_cgstep,acpar->nz*acpar->nx*acpar->nt,acpar->nt*acpar->nx,ww[0][0],dd[0],paspar->niter,"mwt",mwt[0][0],"wt",dwt[0],"verb",verb,"end");
        else np_solver(timerev_lop,np_cgstep,acpar->nz*acpar->nx*acpar->nt,acpar->nt*acpar->nx,ww[0][0],dd[0],paspar->niter,"mwt",mwt[0][0],"verb",verb,"end");
    } else {
    	printf("before timerev3\n");
        timerev_lop(false, false, acpar->nz*acpar->nx*acpar->nt, acpar->nt*acpar->nx, ww[0][0], dd[0]);
    	printf("before timerev4\n");
    }
    
    /* close */
    timerev_close();
    free(vv1);

}

static float ****ww3;
static float ****gwt;
static np_butter blo=NULL, bhi=NULL;
// 
// void lstri_op(float **dd, float **dwt, float ***ww, float ***mwt, np_acqui acpar, np_vec array, np_pas paspar, bool verb)
// /* for passive source and fwi */
void gradient_pas_init(float ***data, float ****src, float ***mwt, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_pas paspar, bool verb1)
/*< initialize >*/
{
	float **dwt=NULL,***wwt;
	int it,ix,iz,is,rdn;
// 	char filename[20]="tempbin",srdn[100000];
//         FILE *temp;

	
	verb=verb1;
	first=true; // only at the first iteration, need to calculate the gradient scaling parameter

	nz=acpar->nz;
	nx=acpar->nx;
	nzx=nz*nx;
	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nb=acpar->nb;
	nt=acpar->nt;

	ns=acpar->ns;
	nr=acpar->nr;
	dr_v=acpar->dr_v;
	r0_v=acpar->r0_v;
	rz=acpar->rz;

	grectx=fwipar->rectx;
	grectz=fwipar->rectz;
	interval=acpar->interval;
	wnt=(nt-1)/interval+1;

	dt=acpar->dt;
	idt=1./dt;
	dt2=dt*dt;
	wdt=dt*interval;
	wdt2=wdt*wdt;
	dx2=acpar->dx*acpar->dx;
	dz2=acpar->dz*acpar->dz;

	wt1=fwipar->wt1;
	wt2=fwipar->wt2;
	woff1=fwipar->woff1;
	woff2=fwipar->woff2;
	waterz=fwipar->waterz;
	waterzb=fwipar->waterzb;

	bc=acpar->bc;

        /* allocate data/source/weight */
//         dd  = np_floatalloc3(nt, nx, ns);
//         ww3 = np_floatalloc4(nz, nx, nt, ns);
		dd=data;
		ww3=src;
        gwt = np_floatalloc4(nz, nx, nt, ns);
//         wwt = np_floatalloc3(nz, nx, nt); /* temporary output var */
        if (!paspar->onlyvel) {
            mwt = np_floatalloc3(nz, nx, nt); /* src model weight */
            /*
            dwt = np_floatalloc2(acpar->nt, acpar->nx);

            wtn1=(fwipar->wt1-acpar->t0)/acpar->dt+0.5;
            wtn2=(fwipar->wt2-acpar->t0)/acpar->dt+0.5;
            woffn1=(fwipar->woff1-acpar->r0)/acpar->dr+0.5;
            woffn2=(fwipar->woff2-acpar->r0)/acpar->dr+0.5;
            residual_weighting(dwt, acpar->nt, acpar->nx, wtn1, wtn2, woffn1, woffn2, !fwipar->oreo);
            */
        } else {
            mwt=NULL;
            dwt=NULL;
        }

        /* read data/source */
        for(is=0; is<ns; is++){
                /* read data */
//                 np_seek(Fdat, is*nt*nx*sizeof(float), SEEK_SET);
//                 np_floatread(dd[is][0], nt*nx, Fdat);
//                 dd=data[is];
                if (paspar->onlyvel) {
                    /* read source */
//                     np_seek(Fsrc, is*nz*nx*nt*sizeof(float), SEEK_SET);
//                     np_floatread(ww3[is][0][0], nz*nx*nt, Fsrc);
// 					 ww3=src;
				printf("in FWI gradient, only vel inversion");
                } else {
                    /* linear inversion of source */
                    lstri_op(data[is], dwt, src[is], mwt, acpar, array, paspar, verb); /*ww3 is updated iteratively*/
                    /*this operator is source-free, i.e., source-by-source, one source a time*/
                    
                    /* write source */
//                     fseeko(temp, is*nz*nx*nt*sizeof(float), SEEK_SET);
//                     fwrite(ww3[is][0][0], sizeof(float), nz*nx*nt, temp);
//                     if (NULL!=Fmwt && is==0) np_floatwrite(mwt[0][0], nz*nx*nt, Fmwt);
                    
// 					lstri_op(float **dd, float **dwt, float ***ww, float ***mwt, np_acqui acpar, np_vec array, np_pas paspar, bool verb)
                    /*this operator is source-free, i.e., source-by-source, one source a time*/
                }

                /* calculate gradient mask */
                if (!paspar->onlyvel && paspar->prec && paspar->hidesrc) {
                    for         (it=0; it<nt; it++)
                        for     (ix=0; ix<nx; ix++)
                            for (iz=0; iz<nz; iz++)
                                gwt[is][it][ix][iz] = mwt[it][ix][iz];
                    threshold(true, nz*nx*nt, paspar->hard, gwt[is][0][0]);
                    for         (it=0; it<nt; it++)
                        for     (ix=0; ix<nx; ix++)
                            for (iz=0; iz<nz; iz++)
                                gwt[is][it][ix][iz] = 1.-gwt[is][it][ix][iz];
                } else {
                    for         (it=0; it<nt; it++)
                        for     (ix=0; ix<nx; ix++)
                            for (iz=0; iz<nz; iz++)
                                gwt[is][it][ix][iz] = 1.;
                }
        }

	/* data residual weights */
	wtn1=(wt1-acpar->t0)/dt+0.5;
	wtn2=(wt2-acpar->t0)/dt+0.5;
	woffn1=(woff1-acpar->r0)/acpar->dr+0.5;
	woffn2=(woff2-acpar->r0)/acpar->dr+0.5;
	weight=np_floatalloc2(nt, nr);
	residual_weighting(weight, nt, nr, wtn1, wtn2, woffn1, woffn2, fwipar->oreo);

	/* padding and convert vector to 2-d array */
	vv = np_floatalloc2(padnz, padnx);
	tau= np_floatalloc2(padnz, padnx);
	taus=np_floatalloc2(padnz, padnx);
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

        /* multiscale gradient */
	if(soupar->flo > 0.0001) blo=np_butter_init(false, soupar->flo, 3);
	if(soupar->fhi < 0.5-0.0001) bhi=np_butter_init(true, soupar->fhi, 3);

//         free(**wwt); free(*wwt); free(wwt);
//         if (NULL!=mwt) { free(**mwt); free(*mwt); free(mwt); }
	return;
}
	
//JS and YC
static int counter=0;
void gradient_pas_av(float *x, float *fcost, float *grad)
/*< acoustic velocity gradient >*/
{
	int ix, iz, is, ir, it, wit;
	int rx;

	float temp, dmax;
	float **p0, **p1, **p2, **term, **tmparray, ***wave, **pp;

	/* initialize fcost */
	*fcost=0.;
	/* update velocity */
	pad2d(x, vv, nz, nx, nb);
	/* initialize gradient */
	memset(grad, 0., nzx*sizeof(float));

	/* memory allocation */
	p0=np_floatalloc2(padnz, padnx);
	p1=np_floatalloc2(padnz, padnx);
	p2=np_floatalloc2(padnz, padnx);
	term=np_floatalloc2(padnz, padnx);
	wave=np_floatalloc3(nz, nx, wnt);
	pp=np_floatalloc2(nt, nr);

        is=0;
        for(is=0; is<ns; is++){
            printf("###### is=%d ######\n", is+1);

            memset(p0[0], 0., padnzx*sizeof(float));
            memset(p1[0], 0., padnzx*sizeof(float));
            memset(p2[0], 0., padnzx*sizeof(float));
            memset(pp[0], 0., nr*nt*sizeof(float));

            wit=0;
            /* forward propagation */
            for(it=0; it<nt; it++){
//                 if(verb) printf("Forward propagation it=%d;\n", it);

                /* output predicted data */
                for(ir=0; ir<nr; ir++){
                    rx=r0_v[0]+ir*dr_v;
                    pp[ir][it]=p1[rx][rz];
                }

                /* save wavefield */
                if(it%interval==0){
                    for(ix=0; ix<nx; ix++)
                        for(iz=0; iz<nz; iz++)
                            wave[wit][ix][iz]=p1[ix+nb][iz+nb];
                    wit++;
                }

                /*
                //JS
                if(is==0 && counter==3 && it%50==0) np_floatwrite(p1[0],padnzx,Fwfl1);
                */

                /* laplacian operator */
                laplace(p1, term, padnx, padnz, dx2, dz2);

                /* load source */
                for(ix=0; ix<nx; ix++){
                    for(iz=0; iz<nz; iz++){
                        term[ix+nb][iz+nb] += ww3[is][it][ix][iz];
                    }
                }

                /* update */
                for(ix=4; ix<padnx-4; ix++){
                    for(iz=4; iz<padnz-4; iz++){
                        p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
                    }
                }

                /* swap wavefield pointer of different time steps */
                tmparray=p0; p0=p1; p1=p2; p2=tmparray;

                /* boundary condition */
                apply_sponge(p0, bc, padnx, padnz, nb);
                apply_sponge(p1, bc, padnx, padnz, nb);
            } // end of time loop

            /* check */
            if(wit != wnt) printf("ERROR: Incorrect number of wavefield snapshots \n ");
            wit--;

            /* calculate data residual and data misfit */
            for(ir=0; ir<nr; ir++){
                for(it=0; it<nt; it++){
                    pp[ir][it]=dd[is][ir][it]-pp[ir][it];
                    *fcost += 0.5*pp[ir][it]*pp[ir][it];
                }
            }

            /* window the data residual */
            for(ir=0; ir<nr; ir++){
                /* multiscale */
                if(NULL != blo){
                    np_butter_apply(blo, nt, pp[ir]);
                    np_reverse(nt, pp[ir]);
                    np_butter_apply(blo, nt, pp[ir]);
                    np_reverse(nt, pp[ir]);
                }
                if(NULL != bhi){
                    np_butter_apply(bhi, nt, pp[ir]);
                    np_reverse(nt, pp[ir]);
                    np_butter_apply(bhi, nt, pp[ir]);
                    np_reverse(nt, pp[ir]);
                }
                for(it=0; it<nt; it++){
                    pp[ir][it] *= weight[ir][it];
                }
            }

            /*
            // JS
            if(is==0 && counter==3) np_floatwrite(pp[0], nr*nt, Fres);
            */

            /* initialization */
            memset(p0[0], 0., padnzx*sizeof(float));
            memset(p1[0], 0., padnzx*sizeof(float));
            memset(p2[0], 0., padnzx*sizeof(float));
            memset(term[0], 0., padnzx*sizeof(float));

            /* backward propagation */
            for(it=nt-1; it>=0; it--){
//                 if(verb) printf("Backward propagation it=%d;\n", it);

                /* laplacian operator */
                laplace(p1, term, padnx, padnz, dx2, dz2);

                /* load data residual*/
                for(ir=0; ir<nr; ir++){
                    rx=r0_v[0]+ir*dr_v;
                    term[rx][rz] += pp[ir][it];
                }

                /* update */
                for(ix=4; ix<padnx-4; ix++){
                    for(iz=4; iz<padnz-4; iz++){
                        p2[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*vv[ix][iz]*dt2*term[ix][iz];
                    }
                }

                /*
                // JS
                if(is==0 && counter==3 && it%50==0) np_floatwrite(p1[0],padnzx,Fwfl2);
                */

                /* calculate gradient  */
                if(it%interval==0){
                    if(wit != wnt-1 && wit != 0){ // avoid the first and last time step
                        for(ix=0; ix<nx; ix++){
                            for(iz=0; iz<nz; iz++){
                                temp=vv[ix+nb][iz+nb];
                                temp=temp*temp*temp;
                                temp=-2./temp;
                                grad[ix*nz+iz] += gwt[is][it][ix][iz]*(wave[wit+1][ix][iz]-2.*wave[wit][ix][iz]+wave[wit-1][ix][iz])/wdt2*p1[ix+nb][iz+nb]*temp;
                            }
                        }
                    }
                    wit--;
                }

                /* swap wavefield pointer of different time steps */
                tmparray=p0; p0=p1; p1=p2; p2=tmparray;

                /* boundary condition */
                apply_sponge(p0, bc, padnx, padnz, nb);
                apply_sponge(p1, bc, padnx, padnz, nb);
            } // end of time loop

//             is++;

        } // end of shot loop

	/* misfit reduction */
// 	*fcost is the summation

        /* scaling gradient */
        if(first){
            dmax=0.;
            for(ix=0; ix<nzx; ix++)
                if(fabsf(grad[ix])>dmax)
                    dmax=fabsf(grad[ix]);
            scaling=0.1/dmax;
            first=false;
        }

	/* smooth gradient */
	gradient_smooth2b(grectx, grectz, nx, nz, waterz, waterzb, scaling, grad);

	/* free allocated memory */
	free(*p0); free(p0); free(*p1); free(p1);
	free(*p2); free(p2); free(*pp); free(pp);
	free(**wave); free(*wave); free(wave);
	free(*term); free(term);
}

