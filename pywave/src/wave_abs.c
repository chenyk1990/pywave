/*below is the including part*/
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "wave_alloc.h"


/** Part II: Absorbing boundary condition ********/
/*Note: more powerful and efficient ABC can be incorporated*/
static int nx, ny, nz, nx2, ny2, nz2, nbt, nbb, nblx, nbrx, nbly, nbry;
static float ct, cb, clx, crx, cly, cry;
static float *wt, *wb, *wlx, *wrx, *wly, *wry;

void vel_expand(float *vel, 				/* input velocity */
				float *vel2,				/* output velocity */
				int nz,   int nx,   int ny, /* size of input velocity */
				int nbt,  int nbb, 			/* ABC size in z  */
				int nblx, int nbrx,			/* ABC size in x  */
				int nbly, int nbry			/* ABC size in y  */)
/*< expand velocity model for ABC, revised on June 2022 YC>*/
{
	int i,j,iz,ix,iy;

// #ifdef _OPENMP
// #pragma omp parallel default(shared) private(iz,ix,iy,i,j)
{
// #endif
    for (iz=0; iz < nz; iz++) {  
        for (ix=0; ix < nx; ix++) {
        for (iy=0; iy < ny; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*(iy+nbly) + (nz+nbt+nbb)*(ix+nblx) + iz+nbt;
	  	j = nz*nx*iy + nz*ix + iz;
	  	vel2[i] = vel[j];
        }
        }
    }
    
	/*top*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nbt; iz++) {  
        for (ix=0; ix < nx+nblx+nbrx; ix++) {
        for (iy=0; iy < ny+nbly+nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + nbt;
	  	vel2[i] = vel2[j];
        }
        }
    }
	
	/*bottom*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nbb; iz++) {  
        for (ix=0; ix < nx+nblx+nbrx; ix++) {
        for (iy=0; iy < ny+nbly+nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + (nz+nbt+nbb-1-iz);
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + (nz+nbt-1);
	  	vel2[i] = vel2[j];
        }
        }
    }

	/*left x*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz+nbt+nbb; iz++) {  
        for (ix=0; ix < nblx; ix++) {
        for (iy=0; iy < ny+nbly+nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*nblx  + iz;
	  	vel2[i] = vel2[j];
        }
        }
    }
    
	/*right x*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz+nbt+nbb; iz++) {  
        for (ix=0; ix < nbrx; ix++) {
        for (iy=0; iy < ny+nbly+nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*(nx+nblx+nbrx-1-ix) + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*(nx+nblx-1) + iz;
	  	vel2[i] = vel2[j];
        }
        }
    }  
    
	/*left y*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz+nbt+nbb; iz++) {  
        for (ix=0; ix < nx+nblx+nbrx; ix++) {
        for (iy=0; iy < nbly; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*iy + (nz+nbt+nbb)*ix + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*nbly + (nz+nbt+nbb)*ix + iz;
	  	vel2[i] = vel2[j];
        }
        }
    }
    
	/*right y*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz+nbt+nbb; iz++) {  
        for (ix=0; ix < nx+nblx+nbrx; ix++) {
        for (iy=0; iy < nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*(ny+nbly+nbry-1-iy) + (nz+nbt+nbb)*ix + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*(ny+nbly-1) + (nz+nbt+nbb)*ix + iz;
	  	vel2[i] = vel2[j];
        }
        }
    }  

// #ifdef _OPENMP
}
// #endif
}

void abc_cal(int abc /* decaying type*/,
             int nb  /* absorbing layer length*/, 
             float c /* decaying parameter*/,
             float* w /* output weight[nb] */)
/*< find absorbing coefficients >*/
{
    int ib;
    /*const float pi=SF_PI;*/
    if(!nb) return;
    switch(abc) {
    default:
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(ib)
// #endif
        for(ib=0; ib<nb; ib++){
	    w[ib]=exp(-c*c*(nb-1-ib)*(nb-1-ib));
	}
    }
}

void abc_init(int n1,  int n2, int n3,    /*model size*/
	      int n12, int n22, int n32,   /*padded model size*/
	      int nb1, int nb2,    /*top, bottom*/
	      int nb3, int nb4,   /*left x, right x*/
	      int nb5, int nb6,   /*left y, right y*/
	      float c1, float c2, /*top, bottom*/
	      float c3, float c4, /*left x, right x*/
	      float c5, float c6 /*left y, right y*/)
/*< initialization >*/
{
    int c;
    nz = n1;
    nx = n2;
    ny = n3;
    nz2= n12;
    nx2= n22;
    ny2= n32;
    nbt = nb1;
    nbb = nb2;
    nblx = nb3;
    nbrx = nb4;
    nbly = nb5;
    nbry = nb6;
    ct = c1;
    cb = c2;
    clx = c3;
    crx = c4;
    cly = c5;
    cry = c6;
    if(nbt) wt =  np_floatalloc(nbt);
    if(nbb) wb =  np_floatalloc(nbb);
    if(nblx) wlx =  np_floatalloc(nblx);
    if(nbrx) wrx =  np_floatalloc(nbrx);
    if(nbly) wly =  np_floatalloc(nbly);
    if(nbry) wry =  np_floatalloc(nbry);
    c=0;
    abc_cal(c,nbt,ct,wt);
    abc_cal(c,nbb,cb,wb);
    abc_cal(c,nblx,clx,wlx);
    abc_cal(c,nbrx,crx,wrx);
    abc_cal(c,nbly,cly,wly);
    abc_cal(c,nbry,cry,wry);      
}
   

void abc_close(void)
/*< free memory allocation>*/
{
    if(nbt) free(wt);
    if(nbb) free(wb);
    if(nblx) free(wlx);
    if(nbrx) free(wrx);
    if(nbly) free(wly);
    if(nbry) free(wry);
}

void abc_apply(float *a /*2-D matrix*/) 
/*< boundary decay>*/
{
    int i;
    int iz, iy, ix;
	
    /* top */
// #ifdef _OPENMP
// #pragma omp parallel default(shared) private(iz,ix,iy,i)
{
// #endif

// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nbt; iz++) {  
        for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < ny2; iy++) {
	  i = nz2*nx2*iy + nz2*ix + iz;
	  a[i] *= wt[iz];
        }
        }
    }
    
    /* bottom */
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nbb; iz++) {  
        for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < ny2; iy++) {
	  i = nz2*nx2*iy + nz2*ix + nz2-1-iz;
	  a[i] *= wb[iz];
        }
    }
    }
      
    /* left x*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz2; iz++) {  
        for (ix=0; ix < nblx; ix++) {
        for (iy=0; iy < ny2; iy++) { 
	  i = nz2*nx2*iy+nz2*ix + iz;
	  a[i] *= wlx[ix];
        }
        }
    }
    
    /* right x*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz2; iz++) {  
        for (ix=0; ix < nbrx; ix++) {
        for (iy=0; iy < ny2; iy++) {     
	  i = nz2*nx2*iy + nz2*(nx2-1-ix) + iz;
          a[i] *= wrx[ix];
        }
        }
    }
        
    /* left y*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz2; iz++) {  
       for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < nbly; iy++) { 
	  i = nz2*nx2*iy+nz2*ix + iz;
	  a[i] *= wly[iy];
        }
        }
    }
        
    /* right y*/
// #ifdef _OPENMP
// #pragma omp for
// #endif
    for (iz=0; iz < nz2; iz++) {  
       for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < nbry; iy++) {    
	  i = nz2*nx2*(ny2-1-iy) + nz2*ix + iz;
          a[i] *= wry[iy];
        }
        }
    }
// #ifdef _OPENMP
}
// #endif
}
