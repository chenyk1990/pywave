/* Time reversal imaging of passive seismic data linear operator */
#include "wave_triutil.h"
#include "wave_alloc.h"
#include "wave_triangle.h"
#include "wave_trianglen.h"
#include "wave_decart.h"
#include "wave_conjgrad.h"

#include <stdio.h>
#include <stdbool.h>

#define np_PI (3.14159265358979323846264338328)

/***************************************************************/
tri2d tri2d_make(bool verb, bool abc,
                int nt, int nx, int nz, int nb, int depth,
                float dt, float dx, float dz, float cb)
/*< initialize tri2d utilities >*/
{   
    tri2d tri;

    tri = (tri2d) np_alloc(1,sizeof(*tri));

    tri->verb  = verb;
    tri->abc   = abc;
    tri->nt    = nt;
    tri->nx    = nx;
    tri->nz    = nz;
    tri->nb    = nb;
    tri->depth = depth;
    tri->dt2   = dt*dt;
    tri->idx2  = 1.0f/(dx*dx);
    tri->idz2  = 1.0f/(dz*dz);
    tri->cb    = cb; 

    tri->nxpad = nx+2*nb;
    tri->nzpad = nz+2*nb;
    tri->nzxpad= tri->nzpad*tri->nxpad;
    tri->depth = depth+nb;

    return tri;
}

/***************************************************************/
static tri2d tr;
static float **vvpad, **u0, **u1, **u2, **tmp;

void timerev_init(bool verb, bool abc,
                  int nt, int nx, int nz, int nb, int depth,
                  float dt, float dx, float dz, float cb,
                  float **vv)
/*< initialize >*/
{
    int ix, iz;

    tr = tri2d_make(verb, abc, nt, nx, nz, nb, depth, dt, dx, dz, cb);

    /* set Laplacian coefficients */
    vvpad = np_floatalloc2(tr->nzpad, tr->nxpad);
    u0    = np_floatalloc2(tr->nzpad, tr->nxpad);
    u1    = np_floatalloc2(tr->nzpad, tr->nxpad);
    u2    = np_floatalloc2(tr->nzpad, tr->nxpad);

    /* pad boundary */
    for     (ix=0; ix<tr->nx; ix++)
        for (iz=0; iz<tr->nz; iz++)
            vvpad[ix+tr->nb][iz+tr->nb] = vv[ix][iz]*vv[ix][iz]*tr->dt2;
    for     (ix=0; ix<tr->nxpad; ix++){
        for (iz=0; iz<tr->nb;    iz++){
            vvpad[ix][          iz  ] = vvpad[ix][          tr->nb  ];
            vvpad[ix][tr->nzpad-iz-1] = vvpad[ix][tr->nzpad-tr->nb-1];
        }
    }
    for     (ix=0; ix<tr->nb;    ix++){
        for (iz=0; iz<tr->nzpad; iz++){
            vvpad[          ix  ][iz]=vvpad[          tr->nb  ][iz];
            vvpad[tr->nxpad-ix-1][iz]=vvpad[tr->nxpad-tr->nb-1][iz];
        }
    }

    /* absorbing boundary condition */
    pfwiabc_init(tr);

}

void timerev_lop(bool adj, bool add, int nm, int nd, float *mod, float *dat)
/*< time reversal imaging linear operator >*/
{
    int ix, iz, it;
    float **dd, ***ww;

    if (nm!=tr->nz*tr->nx*tr->nt || nd!=tr->nt*tr->nx) 
    {
    printf("timerev_lop  wrong dimensions\n");
    }
//     np_error("%s: wrong dimensions",__FILE__);
    np_adjnull(adj, add, nm, nd, mod, dat);
            for     (ix=0; ix<tr->nxpad; ix++)
                for (iz=0; iz<tr->nzpad; iz++)
                {
                    u0[ix][iz] = 0.0f;
                    u1[ix][iz] = 0.0f;
                    u2[ix][iz] = 0.0f;
                }
 
    /* map 1d to 2d */
    dd = (float**) np_alloc (tr->nx,sizeof(float*)); 
    dd[0] = dat;
    for (ix=1; ix<tr->nx; ix++) dd[ix] = dd[0]+ix*tr->nt;

    /* map 1d to 3d */
    ww = (float***) np_alloc (tr->nt,sizeof(float**));
    ww[0] = (float**) np_alloc (tr->nx*tr->nt,sizeof(float*));
    ww[0][0] = mod;
    for (ix=1; ix<tr->nx*tr->nt; ix++) ww[0][ix] = ww[0][0]+ix*tr->nz; 
    for (it=1; it<tr->nt; it++) ww[it] = ww[0]+it*tr->nx;
	
	printf("Timerev_lop\n");
	
    if (adj) { /* migration */
        
        for (it=tr->nt-1; it>-1; it--){
           
            /* 4 - apply abc */
            if (tr->abc) pfwiabc_apply(u1[0],tr);
            if (tr->abc) pfwiabc_apply(u0[0],tr);

            /* 3 - image source */
            for     (ix=0; ix<tr->nx; ix++)
                for (iz=0; iz<tr->nz; iz++)
                    ww[it][ix][iz] += u1[ix+tr->nb][iz+tr->nb]*vvpad[ix+tr->nb][iz+tr->nb];

            /* 2 - time stepping */
            for     (ix=NOP; ix<tr->nxpad-NOP; ix++){
                for (iz=NOP; iz<tr->nzpad-NOP; iz++){
                    u2[ix][iz] = LapT(u1,ix,iz,tr->idx2,tr->idz2,vvpad) + 2.0f*u1[ix][iz] - u0[ix][iz];
                }
            }
            /* rotate pointers */
            tmp=u0; u0=u1; u1=u2; u2=tmp;

            /* 1 - inject data */
            for (ix=tr->nb; ix<tr->nb+tr->nx; ix++)
                u1[ix][tr->depth] += dd[ix-tr->nb][it];
 
        } /* it loop */

    } else { /* modeling */
    		printf("Timerev_lop modeling\n");
    	for (it=0; it<tr->nt; it++){

             /* 1 - record data */
            for (ix=tr->nb; ix<tr->nb+tr->nx; ix++)
                dd[ix-tr->nb][it] += u1[ix][tr->depth];
           
            /* 2 - time stepping */
            for     (ix=NOP; ix<tr->nxpad-NOP; ix++){
                for (iz=NOP; iz<tr->nzpad-NOP; iz++){
                    u2[ix][iz] = Lap(u1,ix,iz,tr->idx2,tr->idz2,vvpad) + 2.0f*u1[ix][iz] - u0[ix][iz];
                }
            }
            /* rotate pointers */
            tmp=u0; u0=u1; u1=u2; u2=tmp;

            /* 3 - inject source */
            for     (ix=0; ix<tr->nx; ix++)
                for (iz=0; iz<tr->nz; iz++)
                    u1[ix+tr->nb][iz+tr->nb] += ww[it][ix][iz]*vvpad[ix+tr->nb][iz+tr->nb];

            /* 4 - apply abc */
            if (tr->abc) pfwiabc_apply(u0[0],tr);
            if (tr->abc) pfwiabc_apply(u1[0],tr);

        } /* it loop */
        
    }

    free(dd);
    free(*ww); free(ww);
}

void timerev_close()
/*< finalize >*/
{
    pfwiabc_close();
    free(tr);
    free(*vvpad); free(vvpad);
    free(*u0); free(u0);
    free(*u1); free(u1);
    free(*u2); free(u2);
}

/***************************************************************/
void ctimerev(int ngrp, float ***ww, float **dd)
/*< correlative time reversal imaging condition >*/
{
    int ix, iz, it, ig, counter, *beg, *end;

    for         (it=0; it<tr->nt; it++)
        for     (ix=0; ix<tr->nx; ix++)
            for (iz=0; iz<tr->nz; iz++)
                ww[it][ix][iz] = 1.0f;

    /* set start and end index */
    beg=np_intalloc(ngrp);
    end=np_intalloc(ngrp);
    counter = 0;
    for (ig=0; ig<ngrp; ig++) {
        beg[ig] = counter;
        counter += tr->nx/ngrp;
        end[ig] = counter;
    }
    end[ngrp-1] = tr->nx;
    if (tr->verb) {
        for (ig=0; ig<ngrp; ig++) {
            printf("beg[%d]=%d\n",ig,beg[ig]);
            printf("end[%d]=%d\n",ig,end[ig]);
        }
    }

    for (ig=0; ig<ngrp; ig++) { /* loop over subgroups of receivers */

            for     (ix=0; ix<tr->nxpad; ix++)
                for (iz=0; iz<tr->nzpad; iz++)
                {
                    u0[ix][iz] = 0.0f;
                    u1[ix][iz] = 0.0f;
                    u2[ix][iz] = 0.0f;
                }

        for (it=tr->nt-1; it>-1; it--){
            if (tr->verb) printf("Time reversal: %d/%d;\n", it, 0);

            /* time stepping */
            for     (ix=NOP; ix<tr->nxpad-NOP; ix++){
                for (iz=NOP; iz<tr->nzpad-NOP; iz++){
                    u2[ix][iz] = Lap (u1,ix,iz,tr->idx2,tr->idz2,vvpad) + 2.0f*u1[ix][iz] - u0[ix][iz];
                }
            }
            /* rotate pointers */
            tmp=u0; u0=u1; u1=u2; u2=tmp;
            if (tr->abc) pfwiabc_apply(u1[0],tr);
            if (tr->abc) pfwiabc_apply(u0[0],tr);

            /* inject data */
            for (ix=tr->nb+beg[ig]; ix<tr->nb+end[ig]; ix++)
                u1[ix][tr->depth] += dd[ix-tr->nb][it];

            /* image source */
            for     (ix=0; ix<tr->nx; ix++)
                for (iz=0; iz<tr->nz; iz++)
                    ww[it][ix][iz] *= u1[ix+tr->nb][iz+tr->nb];

        } /* it loop */
        if (tr->verb) printf(".\n");

    } /* ig loop */


}

/***************************************************************/
/* absorbing boundary */
static float *decay=NULL;

void pfwiabc_init(tri2d tri)
/*< initialization >*/
{
    if(tri->nb) decay =  np_floatalloc(tri->nb);
    pfwiabc_cal(tri->nb,tri->cb,decay);
}
   

void pfwiabc_close(void)
/*< free memory allocation>*/
{
    if(NULL!=decay) free(decay);
    decay = NULL;
}

void pfwiabc_apply(float *a /*2-D matrix*/,
               tri2d tri) 
/*< boundary decay>*/
{
    int iz, ix;

    /* top & bottom */
    for (iz=0; iz < tri->nb; iz++) {  
        for (ix=0; ix < tri->nxpad; ix++) {
	  a[tri->nzpad*ix +              iz] *= decay[iz];
	  a[tri->nzpad*ix + tri->nzpad-1-iz] *= decay[iz];
        }
    }
    /* left & right */
    for (iz=0; iz < tri->nzpad; iz++) {  
        for (ix=0; ix < tri->nb; ix++) {
	  a[tri->nzpad*              ix  + iz] *= decay[ix];
	  a[tri->nzpad*(tri->nxpad-1-ix) + iz] *= decay[ix];
        }
    }
}

void pfwiabc_cal(int nb  /* absorbing layer length*/, 
             float c /* decaying parameter*/,
             float* w /* output weight[nb] */)
/*< calculate absorbing coefficients >*/
{
    int ib;
    if(!nb) return;

    for(ib=0; ib<nb; ib++){
        w[ib]=exp(-c*c*(nb-1-ib)*(nb-1-ib));
    }
}

/***************************************************************/
/* sliding window normalization */

void threshold(bool step, int n, float hard, float *dat)
/*< in-place hard thresholding >*/
{
    int i;

    for (i=0; i<n; i++) {
        if (dat[i]<hard) {
            if (step) dat[i] = 0.0f;
            else { 
                /*dat[i] /= hard;*/
                dat[i] = 0.5*(1.+cosf(np_PI*(dat[i]/hard-1.))); /* Tukey window */
            }
        } else dat[i] = 1.0f;
    }
}

void absval(int n, float *dat)
/*< in-place absolute value >*/
{
    int i;

    for (i=0; i<n; i++)
        dat[i] = fabs(dat[i]);
}

void autopow(int n, float p, float *dat)
/*< in-place auto-correlation with abs >*/
{
    int i;

    for (i=0; i<n; i++)
        dat[i] = powf(fabs(dat[i]),p);
}

float maxval(int n, float *dat)
/*< maximum absolute value and variance (optional) >*/
{
    /* no parallelism */
    float dd, max=0;
    int i;

    for (i=0; i<n; i++) { 
        dd = dat[i];
        if (max<dd) max = dd;
    }

    return max;
}

void scale(float a, int n, float *dat)
/*< scale an array >*/
{
    /* no parallelism */
    int i;

    for (i=0; i<n; i++)
        dat[i] *= a;
}

void swnorm(bool verb, bool sw, int nz, int nx, int nt, int size, float perc, float *dat)
/*< local (sliding-window) normalization >*/
{
    int i, nzx, nzxt;
    float *dat0,den,factor;
    float max_all,pad;

    nzx = nz*nx;
    nzxt = nzx*nt;

    max_all = maxval(nzxt,dat);
    pad = max_all*perc/100.0f;
    if(verb) printf("max_all=%g\n",max_all);

    dat0 = np_floatalloc(nzxt);

    for (i=0; i<nzxt; i++) dat0[i] = dat[i];

    if (!sw) {

        scale(1./max_all,nzxt,dat);

    } else {

        for (i=0; i<size; i++) {
            if (verb) printf("i = %d/%d;\n",i,nt);
            den = maxval(nzx*(i+1+size),dat0);
            if (den <= pad) den = pad;
            factor = 1.0f/den;
            scale(factor,nzx,dat+i*nzx);
        }

        for (i=size; i<nt-size; i++) {
            if (verb) printf("i = %d/%d;\n",i,nt);
            den = maxval(nzx*(2*size+1),dat0+(i-size)*nzx);
            if (den <= pad) den = pad;
            factor = 1.0f/den;
            scale(factor,nzx,dat+i*nzx);
        }

        for (i=nt-size; i<nt; i++) {
            if (verb) printf("i = %d/%d;\n",i,nt);
            den = maxval(nzx*(size+1+(nt-1-i)),dat0+(i-size)*nzx);
            if (den <= pad) den = pad;
            factor = 1.0f/den;
            scale(factor,nzx,dat+i*nzx);
        }
        if (verb) printf(".\n");

    }

    free(dat0);
}

/*********************************************************/
/* smoothing */
void smooth(int n1, int n2, int n3,
            int rect1, int rect2, int rect3,
            int nrep,
	    float *dat)
/*< Generate reflectivity map with smoothing >*/
{   
    int i, j, i0, irep, n123; 
    np_triangle trig;
    int n[3],s[3],rect[3];

    n[0]=n1; n[1]=n2; n[2]=n3;
    s[0]=1;  s[1]=n1; s[2]=n1*n2;
    rect[0]=rect1; rect[1]=rect2; rect[2]=rect3;
    n123=n1*n2*n3;
    
    /* 2-d triangle smoothing */
    for (i=0;i<3;i++) {
        if (rect[i] <= 1) continue;
        trig = np_triangle_init (rect[i],n[i],false);
        for (j=0; j < n123/n[i]; j++) {
            i0 = np_first_index (i,j,3,n,s);
            for (irep=0; irep < nrep; irep++) {
                np_smooth2 (trig,i0,s[i],false,dat);
            }
        }
        np_triangle_close(trig);
    }
}
