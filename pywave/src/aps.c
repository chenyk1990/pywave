#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#define SF_MAX(a,b) ((a) < (b) ? (b) : (a))
#define SF_MIN(a,b) ((a) < (b) ? (a) : (b))
#define SF_MAX_DIM 9
#define SF_PI (3.14159265358979323846264338328)

#include "wave_alloc.h"
#include "wave_komplex.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

float gauss(int n, int m)
/* Gaussian function */
{
    return exp(-2.*SF_PI*SF_PI*m*m/(n*n));
}

int psm(float **wvfld, float ***dat, float **dat_v, float *img, float *vel, psmpar par, bool tri)
/*< pseudo-spectral method >*/
{
    /*survey parameters*/
    int   nx, ny, nz;
    float dx, dy, dz;
    int   ns;
    int   *spx, *spy, *spz;
    int   gpz, gpy, gpx, gplx, gply;
    int   gpz_v, gpx_v, gpl_v;
    int   jsnap;
    /*fft related*/
    bool  cmplx;
    int   pad1;
    /*absorbing boundary*/
    bool abc;
    int nbt, nbb, nblx, nbrx, nbly, nbry;
    float ct,cb,clx,crx,cly,cry;
    /*source parameters*/
    int src; /*source type*/
    int nt;
    float dt,*f0,*t0,*A;
    /*misc*/
    bool verb, ps;
    float vref;
    
    int nx1, ny1, nz1; /*domain of interest*/
    int it,iz,ik,ix,iy,i,j;     /* index variables */
    int nk,nzxy,nz2,nx2,ny2,nzxy2,nkz,nkx,nth;
    int it1, it2, its;
    float dkx,dky,dkz,kx0,ky0,kz0,vref2,kx,ky,kz,k,t;
    float c, old;

    /*wave prop arrays*/
    float *vv;
    sf_complex *cwave,*cwavem;
    float *wave,*curr,*prev,*lapl;

    /*source*/
    float **rick;
    float freq;
    int fft_size;

    /*passing the parameters*/
    nx    = par->nx;
    ny    = par->ny;
    nz    = par->nz;
    dx    = par->dx;
    dy    = par->dy;
    dz    = par->dz;
    ns= par->ns;
    spx   = par->spx;
    spy   = par->spy;
    spz   = par->spz;
    gpz   = par->gpz;
    gpy   = par->gpy;
    gpx   = par->gpx;
    gplx   = par->gplx;
    gply   = par->gply;
    gpz_v = par->gpz_v;
    gpx_v = par->gpx_v;
    gpl_v = par->gpl_v;
    jsnap  = par->jsnap;
    cmplx = par->cmplx;
    pad1  = par->pad1;
    abc   = par->abc;
    nbt   = par->nbt;
    nbb   = par->nbb;
    nblx   = par->nblx;
    nbrx   = par->nbrx;
    nbly   = par->nbly;
    nbry   = par->nbry;
    ct    = par->ct;
    cb    = par->cb;
    clx    = par->clx;
    crx    = par->crx;
    cly    = par->cly;
    cry    = par->cry;
    src   = par->src;
    nt    = par->nt;
    dt    = par->dt;
    f0    = par->f0;
    t0    = par->t0;
    A     = par->A;
    verb  = par->verb;
    ps    = par->ps;
    vref  = par->vref;
    

#ifdef _OPENMP
#pragma omp parallel
    {
      nth = omp_get_num_threads();
    }
#else
    nth = 1;
#endif
    if (verb) sf_warning(">>>> Using %d threads <<<<<", nth);

    nz1 = nz-nbt-nbb;
    nx1 = nx-nblx-nbrx;
    ny1 = ny-nbly-nbry;
    
    nk = fft3_init(cmplx,pad1,nz,nx,ny,&nz2,&nx2,&ny2);
    nzxy = nz*nx*ny;
    nzxy2 = nz2*nx2*ny2;
    
    dkz = 1./(nz2*dz); kz0 = (cmplx)? -0.5/dz:0.;
    dkx = 1./(nx2*dx); kx0 = -0.5/dx;
    dky = 1./(ny2*dy); ky0 = -0.5/dy;
    nkz = (cmplx)? nz2:(nz2/2+1);
    nkx = (cmplx)? nx2:(nx2/2+1);
    
    if(nk!=ny2*nx2*nkz) sf_error("wavenumber dimension mismatch!");
    sf_warning("dkz=%f,dkx=%f,dky=%f,kz0=%f,kx0=%f,ky0=%f",dkz,dkx,dky,kz0,kx0,ky0);
    sf_warning("nk=%d,nkz=%d,nz2=%d,nx2=%d,ny2=%d",nk,nkz,nz2,nx2,ny2);

    if(abc)
      abc_init(nz,nx,ny,nz2,nx2,ny2,nbt,nbb,nblx,nbrx,nbly,nbry,ct,cb,clx,crx,cly,cry);

    /* allocate and read/initialize arrays */
    vv     = sf_floatalloc(nzxy); 
    lapl   = sf_floatalloc(nk);
    wave   = sf_floatalloc(nzxy2);
    curr   = sf_floatalloc(nzxy2);
    prev   = sf_floatalloc(nzxy2);
    cwave  = sf_complexalloc(nk);
    cwavem = sf_complexalloc(nk);

    if (!tri && src==0) {

      rick = sf_floatalloc2(nt,ns);
      for (i=0; i<ns; i++) {
	for (it=0; it<nt; it++) {
	  rick[i][it] = 0.f;
	}
	rick[i][(int)(t0[i]/dt)] = A[i]; /*time delay*/
	freq = f0[i]*dt;           /*peak frequency*/
	fft_size = 2*kiss_fft_next_fast_size((nt+1)/2);
	ricker_init(fft_size, freq, 0);
	sf_freqfilt(nt,rick[i]);
	ricker_close();
      }
    } else{
    	 rick = NULL;}

    for (iz=0; iz < nzxy; iz++) {
        vv[iz] = vel[iz]*vel[iz]*dt*dt;
    }
    vref *= dt;
    vref2 = vref*vref;
    for (iz=0; iz < nzxy2; iz++) {
	curr[iz] = 0.;
	prev[iz] = 0.;
    }

    /* constructing the pseudo-analytical op */
    for (iy=0; iy < ny2; iy++) {
	ky = ky0+iy*dky;
    for (ix=0; ix < nx2; ix++) {
	kx = kx0+ix*dkx;
	for (iz=0; iz < nkz; iz++) {
	    kz = kz0+iz*dkz;
	    k = 2*SF_PI*hypot(ky,hypot(kx,kz));
	    if (ps) lapl[iz+ix*nkz+iy*nkz*nx2] = -k*k;
	    else lapl[iz+ix*nkz+iy*nkz*nx2] = 2.*(cos(vref*k)-1.)/vref2;
	}
    }
    }

    if (tri) { /* time-reversal imaging */
	/* step backward in time */
	it1 = nt-1;
	it2 = -1;
	its = -1;	
    } else { /* modeling */
	/* step forward in time */
	it1 = 0;
	it2 = nt;
	its = +1;
    }

    /* MAIN LOOP */
    for (it=it1; it!=it2; it+=its) {
      
        if(verb) sf_warning("it=%d/%d;",it,nt);

	/* matrix multiplication */
	fft3(curr,cwave);

	for (ik = 0; ik < nk; ik++) {
#ifdef SF_HAS_COMPLEX_H
	  cwavem[ik] = cwave[ik]*lapl[ik];
#else
	  cwavem[ik] = sf_cmul(cwave[ik],lapl[ik]);
#endif
	}
	
	ifft3(wave,cwavem);

#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iy,ix,iz,i,j,old,c)
#endif
	for (iy = 0; iy < ny; iy++) {
	for (ix = 0; ix < nx; ix++) {
	    for (iz=0; iz < nz; iz++) {
		i = iz+ix*nz+iy*nz*nx;  /* original grid */
		j = iz+ix*nz2+iy*nz2*nx2; /* padded grid */

		old = c = curr[j];
		c += c - prev[j];
		prev[j] = old;
		c += wave[j]*vv[i];
		curr[j] = c;
	    }
	}
	}

	if (tri) {
	  /* inject data */
	  if (NULL!=dat) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iy,ix)
#endif
	    for (iy = 0; iy < gply; iy++) {
	    for (ix = 0; ix < gplx; ix++) {
	      curr[gpz+(ix+gpx)*nz2+(iy+gpy)*nz2*nx2] += vv[gpz+(ix+gpx)*nz+(iy+gpy)*nz*nx]*dat[iy][ix][it];
	    }
	    }
	  }
	  if (NULL!=dat_v) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iz)
#endif
	    for (iz = 0; iz < gpl_v; iz++) {
	      curr[gpz_v+iz+(gpx_v)*nz2] += vv[gpz_v+iz+(gpx_v)*nz]*dat_v[iz][it];
	    }
	  }
	} else {
	  t = it*dt;
	  for (i=0; i<ns; i++) {
	    for(iy=-1;iy<=1;iy++) {
	    for(ix=-1;ix<=1;ix++) {
	      for(iz=-1;iz<=1;iz++) {
		ik = (spz[i]+nbt)+iz+nz*(spx[i]+nblx+ix)+nz*nx*(spy[i]+nbly+iy);
		j = (spz[i]+nbt)+iz+nz2*(spx[i]+nblx+ix)+nz2*nx2*(spy[i]+nbly+iy);
		if (src==0) {
		  curr[j] += vv[ik]*rick[i][it]/(abs(ix)+abs(iy)+abs(iz)+1);
		} else {
		  curr[j] += vv[ik]*Ricker(t, f0[i], t0[i], A[i])/(abs(ix)+abs(iy)+abs(iz)+1);
		}
	      }
	    }
	    }
	  }
	}
	
	/*apply abc*/
	if (abc) {
	  abc_apply(curr);
	  abc_apply(prev);
	}

	if (!tri) {
	  /* record data */
	  if (NULL!=dat) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(ix)
#endif
	    for (iy = 0; iy < gply; iy++) {
	    for (ix = 0; ix < gplx; ix++) {
	      dat[iy][ix][it] = curr[gpz+(ix+gpx)*nz2+(iy+gpy)*nz2*nx2];
	    }
	    }
	  }
	  if (NULL!=dat_v) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iz)
#endif
	    for (iz = 0; iz < gpl_v; iz++) {
	      dat_v[iz][it] = curr[gpz_v+iz+(gpx_v)*nz2];
	    }
	  }
	}
		
	/* save wavefield */
	if (jsnap > 0 && it%jsnap==0) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iy,ix,iz,i,j)
#endif
	  for (iy=0; iy<ny1; iy++) {
	  for (ix=0; ix<nx1; ix++) {
	    for (iz=0; iz<nz1; iz++) {
	      i = iz + nz1*ix + nz1*nx1*iy;
	      j = iz+nbt + (ix+nblx)*nz2 + (iy+nbly)*nz2*nx2; /* padded grid */
	      wvfld[it/jsnap][i] = curr[j];
	    }
	  }
	}
	}
    }
    if(verb) sf_warning(".");
    if (tri) {
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(iy,ix,iz)
#endif
    for (iy = 0; iy < ny1; iy++) {
    for (ix = 0; ix < nx1; ix++) {
	for (iz = 0; iz < nz1; iz++) {
	  img[iz + nz1*ix + nz1*nx1*iy] = curr[iz+nbt + (ix+nblx)*nz2 + (iy+nbly)*nz2*nx2];
	}
    }
    }
    }

    /*free up memory*/
//     fft3_finalize();
    if (abc) abc_close();
    free(vv);
    free(lapl);   
    free(wave);
    free(curr);
    free(prev);
    free(cwave);
    free(cwavem);
    
    return 0;
}

static PyObject *aps3dc(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim;
    float *data;
    
    int niter,verb,rect0,n1,ntw,opt,sym,window;
    float dt,alpha,ot;
    int ifb,inv;
    
    /*data and parameters interface*/
	PyArg_ParseTuple(args, "Oiiiiiiiff", &f1,&n1,&verb,&window,&inv,&sym,&opt,&ntw,&dt,&ot);

// 	dout=aps3dc(vel,nt,nx,ny,nz,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);
	
    int i, j, m;
    if (ntw%2 == 0)
        ntw = (ntw+1);
    m = (ntw-1)/2;
    
	ndata=n1;

    int i1, iw, nt, nw, i2, n2, n12, n1w;
    int *rect;
    float t, w, w0, dw, mean=0.0f;
    float *mm, *ww;

    
    if(opt)
	    nt = 2*kiss_fft_next_fast_size((ntw+1)/2);
    else
        nt=ntw;
    if (nt%2) nt++;
    nw = nt/2+1;
    dw = 1./(nt*dt);
	w0 = 0.;

	printf("n1=%d,nw=%d,nt=%d\n",n1,nw,nt);
	printf("dw=%g,w0=%g,dt=%g\n",dw,w0,dt);
    
    kiss_fftr_cfg cfg;
    kiss_fft_cpx *pp, ce, *outp;
    float *p, *inp, *tmp;
    float wt, shift;

    
    p = np_floatalloc(nt);
    pp = (kiss_fft_cpx*) np_complexalloc(nw);
    cfg = kiss_fftr_alloc(nt,inv?1:0,NULL,NULL);
    wt = sym? 1./sqrtf((float) nt): 1.0/nt;

    printf("sym=%d,wt=%g\n",sym,wt);
    
    inp = np_floatalloc(n1);
    tmp = np_floatalloc(n1*nw*2);
//     outp = (kiss_fft_cpx*) np_complexalloc(n1*nw);

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

	data  = (float*)malloc(ndata * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    
    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        inp[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("ndata=%d,ntw=%d\n",ndata,ntw);
        
	
	

#include <rsf.h>

/** Part I: Ricker wavelet ********/
float Ricker(float t, float f0, float t0, float A) 
/*< ricker wavelet:
 * f0: peak frequency
 * t0: time lag
 * A: amplitude
 * ************************>*/
{
        float x=pow(SF_PI*f0*(t-t0),2);
        return -A*exp(-x)*(1-2*x);
}

static kiss_fft_cpx *shape;

void ricker_init(int nfft   /* time samples */, 
		 float freq /* frequency */,
		 int order  /* derivative order */)
/*< initialize >*/
{
    int iw, nw;
    float dw, w;
    kiss_fft_cpx cw;

    /* determine frequency sampling (for real to complex FFT) */
    nw = nfft/2+1;
    dw = 1./(nfft*freq);
 
    shape = (kiss_fft_cpx*) sf_complexalloc(nw);

    for (iw=0; iw < nw; iw++) {
	w = iw*dw;
	w *= w;

	switch (order) {
	    case 2: /* half-order derivative */
		cw.r = 2*SF_PI/nfft;
		cw.i = iw*2*SF_PI/nfft;
		cw = sf_csqrtf(cw);
		shape[iw].r = cw.r*w*expf(1-w)/nfft;
		shape[iw].i = cw.i*w*expf(1-w)/nfft;
		break;
	    case 0:
	    default:
		shape[iw].r = w*expf(1-w)/nfft;
		shape[iw].i = 0.;
		break;
	}
    }

    sf_freqfilt_init(nfft,nw);
    sf_freqfilt_cset(shape);
}

void ricker_close(void) 
/*< free allocated storage >*/
{
    free(shape);
    sf_freqfilt_close();
}

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

#ifdef _OPENMP
#pragma omp parallel default(shared) private(iz,ix,iy,i,j)
{
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
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
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nz+nbt+nbb; iz++) {  
        for (ix=0; ix < nx+nblx+nbrx; ix++) {
        for (iy=0; iy < nbry; iy++) {
	  	i = (nz+nbt+nbb)*(nx+nblx+nbrx)*(ny+nbly+nbry-1-iy) + (nz+nbt+nbb)*ix + iz;
	  	j = (nz+nbt+nbb)*(nx+nblx+nbrx)*(ny+nbly-1) + (nz+nbt+nbb)*ix + iz;
	  	vel2[i] = vel2[j];
        }
        }
    }  

#ifdef _OPENMP
}
#endif
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
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(ib)
#endif
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
    if(nbt) wt =  sf_floatalloc(nbt);
    if(nbb) wb =  sf_floatalloc(nbb);
    if(nblx) wlx =  sf_floatalloc(nblx);
    if(nbrx) wrx =  sf_floatalloc(nbrx);
    if(nbly) wly =  sf_floatalloc(nbly);
    if(nbry) wry =  sf_floatalloc(nbry);
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
#ifdef _OPENMP
#pragma omp parallel default(shared) private(iz,ix,iy,i)
{
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nbt; iz++) {  
        for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < ny2; iy++) {
	  i = nz2*nx2*iy + nz2*ix + iz;
	  a[i] *= wt[iz];
        }
        }
    }
    
    /* bottom */
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nbb; iz++) {  
        for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < ny2; iy++) {
	  i = nz2*nx2*iy + nz2*ix + nz2-1-iz;
	  a[i] *= wb[iz];
        }
    }
    }
      
    /* left x*/
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nz2; iz++) {  
        for (ix=0; ix < nblx; ix++) {
        for (iy=0; iy < ny2; iy++) { 
	  i = nz2*nx2*iy+nz2*ix + iz;
	  a[i] *= wlx[ix];
        }
        }
    }
    
    /* right x*/
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nz2; iz++) {  
        for (ix=0; ix < nbrx; ix++) {
        for (iy=0; iy < ny2; iy++) {     
	  i = nz2*nx2*iy + nz2*(nx2-1-ix) + iz;
          a[i] *= wrx[ix];
        }
        }
    }
        
    /* left y*/
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nz2; iz++) {  
       for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < nbly; iy++) { 
	  i = nz2*nx2*iy+nz2*ix + iz;
	  a[i] *= wly[iy];
        }
        }
    }
        
    /* right y*/
#ifdef _OPENMP
#pragma omp for
#endif
    for (iz=0; iz < nz2; iz++) {  
       for (ix=0; ix < nx2; ix++) {
        for (iy=0; iy < nbry; iy++) {    
	  i = nz2*nx2*(ny2-1-iy) + nz2*ix + iz;
          a[i] *= wry[iy];
        }
        }
    }
#ifdef _OPENMP
}
#endif
}


/** Part III: Fourier transform ********/
static bool cmplx;
static int n1, n2, n3, nk;
static float wwt;

static float ***ff=NULL;
static sf_complex ***cc=NULL;

#ifdef SF_HAS_FFTW
static fftwf_plan cfg=NULL, icfg=NULL;
#else
static kiss_fftr_cfg cfg, icfg;
static kiss_fft_cfg cfg1, icfg1, cfg2, icfg2, cfg3, icfg3;
static kiss_fft_cpx ***tmp, *ctrace2, *ctrace3;
static sf_complex *trace2, *trace3;
#endif

int fft3_init(bool cmplx1        /* if complex transform */,
	      int pad1           /* padding on the first axis */,
	      int nx,   int ny,   int nz   /* axis 1,2,3; input data size */, 
	      int *nx2, int *ny2, int *nz2 /* axis 1,2,3; padded data size */)
/*< initialize >*/
{
#ifndef SF_HAS_FFTW
    int i2, i3;
#endif

    cmplx = cmplx1;

    /* axis 1 */

    if (cmplx) {
	nk = n1 = kiss_fft_next_fast_size(nx*pad1);

#ifndef SF_HAS_FFTW
	cfg1  = kiss_fft_alloc(n1,0,NULL,NULL);
	icfg1 = kiss_fft_alloc(n1,1,NULL,NULL);
#endif
    } else {
	nk = kiss_fft_next_fast_size(pad1*(nx+1)/2)+1;
	n1 = 2*(nk-1);

#ifndef SF_HAS_FFTW
	cfg  = kiss_fftr_alloc(n1,0,NULL,NULL);
	icfg = kiss_fftr_alloc(n1,1,NULL,NULL);
#endif
    }

    /* axis 2 */

    n2 = kiss_fft_next_fast_size(ny);

#ifndef SF_HAS_FFTW
    cfg2  = kiss_fft_alloc(n2,0,NULL,NULL);
    icfg2 = kiss_fft_alloc(n2,1,NULL,NULL);

    trace2 = sf_complexalloc(n2);
    ctrace2 = (kiss_fft_cpx *) trace2;
#endif

    /* axis 3 */

    n3 = kiss_fft_next_fast_size(nz);

#ifndef SF_HAS_FFTW
    cfg3  = kiss_fft_alloc(n3,0,NULL,NULL);
    icfg3 = kiss_fft_alloc(n3,1,NULL,NULL);

    trace3 = sf_complexalloc(n3);
    ctrace3 = (kiss_fft_cpx *) trace3;

    /* --- */

    tmp = (kiss_fft_cpx***) sf_alloc (n3,sizeof(kiss_fft_cpx**));
    tmp[0] = (kiss_fft_cpx**) sf_alloc (n2*n3,sizeof(kiss_fft_cpx*));
    tmp[0][0] = (kiss_fft_cpx*) sf_alloc (nk*n2*n3,sizeof(kiss_fft_cpx));

    for (i2=1; i2 < n2*n3; i2++) {
	tmp[0][i2] = tmp[0][0]+i2*nk;
    }

    for (i3=1; i3 < n3; i3++) {
	tmp[i3] = tmp[0]+i3*n2;
    }
#endif

    if (cmplx) {
	cc = sf_complexalloc3(n1,n2,n3);
    } else {
	ff = sf_floatalloc3(n1,n2,n3);
    }

    *nx2 = n1;
    *ny2 = n2;
    *nz2 = n3;

    wwt =  1.0/(n3*n2*n1);

    return (nk*n2*n3);
}

void fft3(float *inp      /* [n1*n2*n3] */, 
	  sf_complex *out /* [nk*n2*n3] */)
/*< 3-D FFT >*/
{
    int i1, i2, i3;
    float f;

  #ifdef SF_HAS_FFTW
    if (NULL==cfg) {
	cfg = cmplx? 
	    fftwf_plan_dft_3d(n3,n2,n1,
			      (fftwf_complex *) cc[0][0], 
			      (fftwf_complex *) out,
			      FFTW_FORWARD, FFTW_MEASURE):
	    fftwf_plan_dft_r2c_3d(n3,n2,n1,
				  ff[0][0], (fftwf_complex *) out,
				  FFTW_MEASURE);
	if (NULL == cfg) sf_error("FFTW failure.");
    }
#endif  
    
    /* FFT centering */    
    for (i3=0; i3<n3; i3++) {
	for (i2=0; i2<n2; i2++) {
	    for (i1=0; i1<n1; i1++) {
		f = inp[(i3*n2+i2)*n1+i1];
		if (cmplx) {
		    cc[i3][i2][i1] = sf_cmplx((((i3%2==0)==(i2%2==0))==(i1%2==0))? f:-f,0.);
		} else {
		    ff[i3][i2][i1] = ((i3%2==0)==(i2%2==0))? f:-f;
		}
	    }
	}
    }

#ifdef SF_HAS_FFTW
    fftwf_execute(cfg);
#else

    /* FFT over first axis */
    for (i3=0; i3 < n3; i3++) {
	for (i2=0; i2 < n2; i2++) {
	    if (cmplx) {
		kiss_fft_stride(cfg1,(kiss_fft_cpx *) cc[i3][i2],tmp[i3][i2],1);
	    } else {
		kiss_fftr (cfg,ff[i3][i2],tmp[i3][i2]);
	    }
	}
    }

    /* FFT over second axis */
    for (i3=0; i3 < n3; i3++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(cfg2,tmp[i3][0]+i1,ctrace2,nk);
	    for (i2=0; i2 < n2; i2++) {
		tmp[i3][i2][i1]=ctrace2[i2];
	    }
	}
    }

    /* FFT over third axis */
    for (i2=0; i2 < n2; i2++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(cfg3,tmp[0][0]+i2*nk+i1,ctrace3,nk*n2);
	    for (i3=0; i3<n3; i3++) {
		out[(i3*n2+i2)*nk+i1] = trace3[i3];
	    }
	}
    } 
   
#endif

}

void ifft3_allocate(sf_complex *inp /* [nk*n2*n3] */)
/*< allocate inverse transform >*/
{
#ifdef SF_HAS_FFTW
    icfg = cmplx? 
	fftwf_plan_dft_3d(n3,n2,n1,
			  (fftwf_complex *) inp, 
			  (fftwf_complex *) cc[0][0],
			  FFTW_BACKWARD, FFTW_MEASURE):
	fftwf_plan_dft_c2r_3d(n3,n2,n1,
			      (fftwf_complex *) inp, ff[0][0],
			      FFTW_MEASURE);
    if (NULL == icfg) sf_error("FFTW failure.");
 #endif
}

void ifft3(float *out      /* [n1*n2*n3] */, 
	   sf_complex *inp /* [nk*n2*n3] */)
/*< 3-D inverse FFT >*/
{
    int i1, i2, i3;

#ifdef SF_HAS_FFTW
    fftwf_execute(icfg);
#else

    /* IFFT over third axis */
    for (i2=0; i2 < n2; i2++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(icfg3,(kiss_fft_cpx *) (inp+i2*nk+i1),ctrace3,nk*n2);
	    for (i3=0; i3<n3; i3++) {
		tmp[i3][i2][i1] = ctrace3[i3];
	    }
	}
    }
    
    /* IFFT over second axis */
    for (i3=0; i3 < n3; i3++) {
	for (i1=0; i1 < nk; i1++) {
	    kiss_fft_stride(icfg2,tmp[i3][0]+i1,ctrace2,nk);		
	    for (i2=0; i2<n2; i2++) {
		tmp[i3][i2][i1] = ctrace2[i2];
	    }
	}
    }

    /* IFFT over first axis */
    for (i3=0; i3 < n3; i3++) {
	for (i2=0; i2 < n2; i2++) {
	    if (cmplx) {
		kiss_fft_stride(icfg1,tmp[i3][i2],(kiss_fft_cpx *) cc[i3][i2],1);		
	    } else {
		kiss_fftri(icfg,tmp[i3][i2],ff[i3][i2]);
	    }
	}
    }

#endif

    /* FFT centering and normalization */
    for (i3=0; i3<n3; i3++) {
	for (i2=0; i2<n2; i2++) {
	    for (i1=0; i1<n1; i1++) {
		if (cmplx) {
		    out[(i3*n2+i2)*n1+i1] = ((((i3%2==0)==(i2%2==0))==(i1%2==0))? wwt:-wwt)*crealf(cc[i3][i2][i1]);
		} else {
		    out[(i3*n2+i2)*n1+i1] = (((i3%2==0)==(i2%2==0))? wwt: - wwt)*ff[i3][i2][i1];
		}
	    }
	}
    }
}


/** Part IV: pseudo-spectral wave extrapolation ********/
typedef struct Psmpar {
  /*survey parameters*/
  int   nx, ny, nz;
  float dx, dy, dz;
  int   ns;
  int   *spx, *spy, *spz;
  int   gpz, gpx, gpy, gplx, gply;
  int   gpz_v, gpx_v, gpl_v;
  int   jsnap;
  /*fft related*/
  bool  cmplx;
  int   pad1;
  /*absorbing boundary*/
  bool abc;
  int nbt, nbb, nblx, nbrx, nbly, nbry;
  float ct,cb,clx,crx,cly,cry;
  /*source parameters*/
  int src; /*source type*/
  int nt;
  float dt,*f0,*t0,*A;
  /*misc*/
  bool verb, ps;
  float vref;
} * psmpar; /*psm parameters*/
/*^*/




/** Part V: main program ********/
// int main(int argc, char* argv[])
// {

    /*survey parameters*/
    int   nx, ny, nz;
    float dx, dy, dz;
    int   ns;
    int   *spx, *spy, *spz;
    int   gpz, gpx, gpy, gplx, gply; /*geophone positions (z,x,y) and geophone length (z,x,y)*/
    int   gpz_v, gpx_v, gpy_v, gpl_v;
    int   jsnap;
    /*fft related*/
    bool  cmplx;
    int   pad1;
    /*absorbing boundary*/
    bool abc,ifvpad;
    int nbt, nbb, nblx, nbrx, nbly, nbry; /*boundaries for top/bottom, left/right x, left/right y*/
    float ct,cb,clx,crx,cly,cry; 		  /*decaying parameter*/
    /*source parameters*/
    int src; /*source type*/
    int nt,ntsnap;
    float dt,*f0,*t0,*A;
    /*misc*/
    bool verb, ps, tri; /*tri: time-reversal imaging*/
    float vref;

    psmpar par;
    int nx1, ny1, nz1; /*domain of interest*/
    int it;
    float *vel,*vel2,***dat,**dat_v,**wvfld,*img; /*velocity profile*/
    sf_file Fi,Fo,Fd,Fd_v,snaps,Fvpad; /* I/O files */
    sf_axis az,ax,ay; /* cube axes */

    sf_init(argc,argv);

    if (!sf_getint("jsnap",&jsnap)) jsnap=0; /* interval for snapshots */
    if (!sf_getbool("cmplx",&cmplx)) cmplx=true; /* use complex fft */
    if (!sf_getint("pad1",&pad1)) pad1=1; /* padding factor on the first axis */
    if(!sf_getbool("abc",&abc)) abc=false; /* absorbing flag */
    if (abc) {
      if(!sf_getint("nbt",&nbt)) sf_error("Need nbt!");
      if(!sf_getint("nbb",&nbb)) nbb = nbt;
      if(!sf_getint("nblx",&nblx)) nblx = nbt;
      if(!sf_getint("nbrx",&nbrx)) nbrx = nbt;
      if(!sf_getint("nbly",&nbly)) nbly = nbt;
      if(!sf_getint("nbry",&nbry)) nbry = nbt;
      if(!sf_getfloat("ct",&ct)) sf_error("Need ct!");
      if(!sf_getfloat("cb",&cb)) cb = ct;
      if(!sf_getfloat("clx",&clx)) clx = ct;
      if(!sf_getfloat("crx",&crx)) crx = ct;
      if(!sf_getfloat("cly",&cly)) cly = ct;
      if(!sf_getfloat("cry",&cry)) cry = ct;
    } else {
      nbt = 0; nbb = 0; nblx = 0; nbrx = 0; nbly = 0; nbry = 0;
      ct = 0; cb = 0; clx = 0; crx = 0; cly = 0; cry = 0;
    }
    if (!sf_getbool("verb",&verb)) verb=false; /* verbosity */
    if (!sf_getbool("ps",&ps)) ps=false; /* use pseudo-spectral */
    if (ps) sf_warning("Using pseudo-spectral...");
    else sf_warning("Using pseudo-analytical...");
    if (!sf_getbool("tri",&tri)) tri=false; /* if choose time reversal imaging */
    if (tri) sf_warning("Time-reversal imaging");
    else sf_warning("Forward modeling");
    if (!sf_getfloat("vref",&vref)) vref=1500; /* reference velocity (default using water) */

    /* setup I/O files */
    Fi = sf_input ("in");
    Fo = sf_output("out");
    
    
    if (tri) {
      gplx = -1;
      gply = -1;
      gpl_v = -1;
      if (NULL==sf_getstring("dat") && NULL==sf_getstring("dat_v"))
	sf_error("Need Data!");
      if (NULL!=sf_getstring("dat")) {
	Fd = sf_input("dat");
	sf_histint(Fd,"n1",&nt);
	sf_histfloat(Fd,"d1",&dt);
	sf_histint(Fd,"n2",&gplx);
	sf_histint(Fd,"n3",&gply);
      } else Fd = NULL;
      if (NULL!=sf_getstring("dat_v")) {
	Fd_v = sf_input("dat_v");
	sf_histint(Fd_v,"n1",&nt);
	sf_histfloat(Fd_v,"d1",&dt);
	sf_histint(Fd_v,"n2",&gpl_v);
      } else Fd_v = NULL;
      src = -1; ns = -1;
      spx = NULL; spy = NULL; spz = NULL;
      f0 = NULL; t0 = NULL; A = NULL;
    } else {
      Fd = NULL;
      if (!sf_getint("nt",&nt)) sf_error("Need nt!");
      if (!sf_getfloat("dt",&dt)) sf_error("Need dt!");
      if (!sf_getint("gplx",&gplx)) gplx = -1; /* geophone length X*/
      if (!sf_getint("gply",&gply)) gply = -1; /* geophone length Y*/
      if (!sf_getint("gpl_v",&gpl_v)) gpl_v = -1; /* geophone height */
      if (!sf_getint("src",&src)) src=0; /* source type */
      if (!sf_getint("ns",&ns)) ns=1; /* source type */
      spx = sf_intalloc(ns);
      spy = sf_intalloc(ns);
      spz = sf_intalloc(ns);
      f0  = sf_floatalloc(ns);
      t0  = sf_floatalloc(ns);
      A   = sf_floatalloc(ns);
      if (!sf_getints("spx",spx,ns)) sf_error("Need spx!"); /* shot position x */
      if (!sf_getints("spy",spy,ns)) sf_error("Need spy!"); /* shot position y */
      if (!sf_getints("spz",spz,ns)) sf_error("Need spz!"); /* shot position z */
      if (!sf_getfloats("f0",f0,ns)) sf_error("Need f0! (e.g. 30Hz)");   /*  wavelet peak freq */
      if (!sf_getfloats("t0",t0,ns)) sf_error("Need t0! (e.g. 0.04s)");  /*  wavelet time lag */
      if (!sf_getfloats("A",A,ns)) sf_error("Need A! (e.g. 1)");     /*  wavelet amplitude */
    }
    if (!sf_getint("gpx",&gpx)) gpx = -1; /* geophone position x */
    if (!sf_getint("gpy",&gpy)) gpy = -1; /* geophone position y */
    if (!sf_getint("gpz",&gpz)) gpz = -1; /* geophone position z */
    if (!sf_getint("gpx_v",&gpx_v)) gpx_v = -1; /* geophone position x */
    if (!sf_getint("gpy_v",&gpy_v)) gpy_v = -1; /* geophone position y */
    if (!sf_getint("gpz_v",&gpz_v)) gpz_v = -1; /* geophone position z */

    if (SF_FLOAT != sf_gettype(Fi)) sf_error("Need float input");

    /* Read/Write axes */
    az = sf_iaxa(Fi,1); nz = sf_n(az); dz = sf_d(az);
    ax = sf_iaxa(Fi,2); nx = sf_n(ax); dx = sf_d(ax);
    ay = sf_iaxa(Fi,3); ny = sf_n(ay); dy = sf_d(ay);


    /*change on Jun 2022, YC*/
    nz1 = nz;
    nx1 = nx;
    ny1 = ny;
    nz = nz+nbt+nbb;
    nx = nx+nblx+nbrx;
    ny = ny+nbly+nbry;
    /*change on Jun 2022, YC*/
    
    if(verb)sf_warning("ny=%d,nbly=%d,nbry=%d",ny,nbly,nbry);
    if(verb)sf_warning("nz1=%d,nx1=%d,ny1=%d",nz1,nx1,ny1);
    if (gpx==-1) gpx = nblx;
    if (gpy==-1) gpy = nbly;
    if (gpz==-1) gpz = nbt;
    if (gplx==-1) gplx = nx1;
    if (gply==-1) gply = ny1;
    if (gpx_v==-1) gpx_v = nblx;
    if (gpy_v==-1) gpy_v = nbly;
    if (gpz_v==-1) gpz_v = nbt;
    if (gpl_v==-1) gpl_v = nz1;
    ntsnap=0;
    if (jsnap)
        for (it=0;it<nt;it++)
            if (it%jsnap==0) ntsnap++;
    if (tri) { /*output final wavefield*/
      sf_setn(az,nz1);
      sf_setn(ax,nx1);
      sf_setn(ay,ny1);
      sf_oaxa(Fo,az,1);
      sf_oaxa(Fo,ax,2);
      sf_oaxa(Fo,ay,3);   
      sf_settype(Fo,SF_FLOAT);
    } else { /*output data*/
      sf_setn(ax,gplx);
      sf_setn(ay,gply);
      sf_putint(Fo,"n3",gply);
      sf_warning("ny=%d,nbly=%d,nbry=%d",ny,nblx,nbly);
      sf_warning("gplx=%d,gply=%d",gplx,gply);
      /*output horizontal data is mandatory*/
      sf_putint(Fo,"n1",nt);
      sf_putfloat(Fo,"d1",dt);
      sf_putfloat(Fo,"o1",0.);
      sf_putstring(Fo,"label1","Time");
      sf_putstring(Fo,"unit1","s");
      sf_oaxa(Fo,ax,2);
      sf_settype(Fo,SF_FLOAT);
      /*output vertical data is optional*/
      if (NULL!=sf_getstring("dat_v")) {
	Fd_v = sf_output("dat_v");
	sf_setn(az,gpl_v);
	sf_putint(Fd_v,"n1",nt);
	sf_putfloat(Fd_v,"d1",dt);
	sf_putfloat(Fd_v,"o1",0.);
	sf_putstring(Fd_v,"label1","Time");
	sf_putstring(Fd_v,"unit1","s");
	sf_oaxa(Fd_v,az,2);
	sf_putint(Fd_v,"n3",1);
	sf_settype(Fd_v,SF_FLOAT);	
      } else Fd_v = NULL;
    }

    if (NULL!=sf_getstring("vpad")) {
	   Fvpad=sf_output("vpad");
       sf_putint(Fvpad,"n1",nz);
       sf_putint(Fvpad,"n2",nx);
       sf_putint(Fvpad,"n3",ny);
       ifvpad=true;
      } else 
      {
      Fvpad = NULL;
       ifvpad=false;
      }
      
    if (jsnap > 0) {
	snaps = sf_output("snaps");
	/* (optional) snapshot file */
	sf_setn(az,nz1);
	sf_setn(ax,nx1);
	sf_setn(ay,ny1);
	sf_oaxa(snaps,az,1);
	sf_oaxa(snaps,ax,2);
	sf_oaxa(snaps,ay,3);
	sf_putint(snaps,"n4",ntsnap);
	sf_putfloat(snaps,"d4",dt*jsnap);
	sf_putfloat(snaps,"o4",0.);
	sf_putstring(snaps,"label4","Time");
	sf_putstring(snaps,"unit4","s");
    } else snaps = NULL;

    par = (psmpar) sf_alloc(1,sizeof(*par));
    vel = sf_floatalloc(nz1*ny1*nx1); 	/*change on Jun 2022, YC*/
    vel2= sf_floatalloc(nz*ny*nx); 		/*change on Jun 2022, YC*/
    
    if (tri && NULL==Fd) {dat = NULL;  }
    else { dat = sf_floatalloc3(nt,gplx,gply);}

    if (NULL!=Fd_v) dat_v = sf_floatalloc2(nt,gpl_v);
    else dat_v = NULL;

    if (tri) img = sf_floatalloc(nz1*ny1*nx1);
    else img = NULL;

    if (jsnap>0) wvfld = sf_floatalloc2(nx1*ny1*nz1,ntsnap);
    else wvfld = NULL;
    

    sf_floatread(vel,nz1*ny1*nx1,Fi);
	vel_expand(vel,vel2,nz1,nx1,ny1,nbt,nbb,nblx,nbrx,nbly,nbry);  /*change on Jun 2022, YC*/
	
    if (tri) {
      if (NULL!=Fd)   sf_floatread(dat[0][0],gplx*gply*nt,Fd);
      if (NULL!=Fd_v) sf_floatread(dat_v[0],gpl_v*nt,Fd_v);
    }
    /*passing the parameters*/
    par->nx    = nx;  
    par->ny    = ny;
    par->nz    = nz;
    par->dx    = dx;
    par->dy    = dy;
    par->dz    = dz;
    par->ns	   = ns;
    par->spx   = spx;
    par->spy   = spy;
    par->spz   = spz;
    par->gpx   = gpx;
    par->gpy   = gpy;
    par->gpz   = gpz;
    par->gplx   = gplx;
    par->gply   = gply;
    par->gpz_v = gpz_v;
    par->gpx_v = gpx_v;
    par->gpl_v = gpl_v;
    par->jsnap  = jsnap;
    par->cmplx = cmplx;
    par->pad1  = pad1;
    par->abc   = abc;
    par->nbt   = nbt;
    par->nbb   = nbb;
    par->nblx   = nblx;
    par->nbrx   = nbrx;
    par->nbly   = nbly;
    par->nbry   = nbry;
    par->ct    = ct;
    par->cb    = cb;
    par->clx    = clx;
    par->crx    = crx;
    par->cly    = cly;
    par->cry    = cry;
    par->src   = src;
    par->nt    = nt;
    par->dt    = dt;
    par->f0    = f0;
    par->t0    = t0;
    par->A     = A;
    par->verb  = verb;
    par->ps    = ps;
    par->vref  = vref;

    /*do the work*/
    psm(wvfld, dat, dat_v, img, vel2, par, tri);

    if (tri) {
      sf_floatwrite(img,nz1*ny1*nx1,Fo);
    } else {
      sf_floatwrite(dat[0][0],gplx*gply*nt,Fo);
      if (NULL!=Fd_v)
	sf_floatwrite(dat_v[0],gpl_v*nt,Fd_v);
    }

    if (jsnap>0)
      sf_floatwrite(wvfld[0],nz1*nx1*ny1*ntsnap,snaps);
      
    if(ifvpad)
    	sf_floatwrite(vel2,nz*nx*ny,Fvpad);
      
//     exit (0);
// }

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	/*sub-function goes here*/
    }
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];
    
    if(!inv)
    {

        for(i=0;i<ndata*nw;i++)
        {
            tmp[i]=outp[i].r;
            tmp[i+ndata*nw]=outp[i].i;
        }
	dims[0]=ndata*nw*2+3;dims[1]=1;
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<ndata*nw*2;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = tmp[i];
	printf("w0=%g,dw=%g,nw=%d\n",w0,dw,nw);
	(*((float*)PyArray_GETPTR1(vecout,0+ndata*nw*2))) = w0;
	(*((float*)PyArray_GETPTR1(vecout,1+ndata*nw*2))) = dw;
	(*((float*)PyArray_GETPTR1(vecout,2+ndata*nw*2))) = nw;
	
	}else{
	
	dims[0]=n1;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = inp[i];
	}
	
	return PyArray_Return(vecout);
	
}

static PyObject *st1dc(PyObject *self, PyObject *args){
    
    /**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

    int ndata;    /*integer parameter*/
    float fpar; /*float parameter*/
    int ndim;
    float *data;
    
    int niter,verb,rect0,n1,ntw,opt,sym,window;
    float dt,alpha,ot;
    int ifb,inv;
    
    float fhi,flo;
    int nflo,nfhi,nf, nw;
    
    PyArg_ParseTuple(args, "Oiiifff", &f1,&n1,&verb,&inv,&flo,&fhi,&dt);

    if(!inv)
    {
        nflo = (int) (flo*n1+0.5);
        nfhi = (int) (fhi*n1+0.5);
        nf = nfhi-nflo+1;
        
    }else{
        nflo = (int) (flo*n1+0.5);
        nfhi = (int) (fhi*n1+0.5);
        nf = nfhi-nflo+1;
    }
    
    int i, j, m;
    nw=nf;
    ndata=n1;

    int i1, iw, nt, i2, n2, n12;
    int *rect;
    float t, w, w0, dw, mean=0.0f;
    float *mm, *ww;

    dw=1./(n1*dt);
    w0=flo/dt;

    printf("n1=%d,nw=%d,nt=%d\n",n1,nw,n1);
    printf("dw=%g,w0=%g,dt=%g\n",dw,w0,dt);
    
    kiss_fftr_cfg cfg;
    kiss_fft_cpx *pp, ce, *outp;
    float *p, *inp, *tmp;
    float wt, snfhift;
    
    inp = np_floatalloc(n1);
    tmp = np_floatalloc(n1*nw*2);
    outp = (kiss_fft_cpx*) np_complexalloc(n1*nw);

    if(!inv)
    {
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

    data  = (float*)malloc(ndata * sizeof(float));
    
    if (*sp != ndata)
    {
        printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
        return NULL;
    }
    
    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        inp[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
    printf("ndata=%d\n",ndata);
        
        int i, i1, k, l2;
        float s, *g;

        kiss_fft_cpx *d, *pp, *qq;
        kiss_fft_cfg tfft, itfft;

        nw = 2*kiss_fft_next_fast_size((n1+1)/2);
        printf("In this program, nf=%d,nw=%d are different, remember\n",nf,nw);
        
        tfft = kiss_fft_alloc(nw,0,NULL,NULL);
        itfft = kiss_fft_alloc(nw,1,NULL,NULL);

        pp = (kiss_fft_cpx*) np_complexalloc(nw);
        qq = (kiss_fft_cpx*) np_complexalloc(nw);
        d =  (kiss_fft_cpx*) np_complexalloc(nw);
        g = np_floatalloc(nw);

        s = 0.;
        for (i = 0; i < n1; i++) {
            d[i].r = inp[i];
            d[i].i = 0.;
            s += inp[i];
        }
        s /= n1;

        for (i=n1; i < nw; i++) {
            d[i].r = 0.;
            d[i].i = 0.;
        }

        kiss_fft_stride (tfft,d,pp,1);
        
        l2 = (nw+1)/2;
        for (i=1; i < l2; i++) {
            pp[i].r *= 2.;
            pp[i].i *= 2.;
        }
        l2 = nw/2+1;
        for (i=l2; i < nw; i++) {
            pp[i].r = 0.;
            pp[i].i = 0.;
        }

        for (i1=nflo; i1 <= nfhi; i1++) {
            if (0 == i1) {
                for (i=0; i < n1; i++) {
                    outp[(i1-nflo)*n1+i] = np_cmplx(s,0.);
                }
            } else {
                g[0] = gauss(i1, 0);
                l2 = nw/2 + 1;
                for (i=1; i < l2; i++) {
                    g[i] = g[nw-i] = gauss(i1, i);
                }

                for (i=0; i < nw; i++) {
                    s = g[i];
                    k = i1 + i;
                    if (k >= nw) k -= nw;
                    qq[i].r = pp[k].r * s;
                    qq[i].i = pp[k].i * s;
                }

                kiss_fft_stride(itfft,qq,d,1);
                
                for (i=0; i < n1; i++) {
                    outp[(i1-nflo)*n1+i] = np_cmplx(d[i].r/n1,d[i].i/n1);
                }
            }
        }
        free(pp);
        free(qq);
        free(d);
        free(g);
        
    }else{
    /*This part is to reconstruct the data given the basis functions and their weights (i.e., TF spectrum)*/
    
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    for (i=0; i<n1*nw*2; i++)
    {
        tmp[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
    for (i=0; i<n1*nw; i++)
    {
        outp[i]=np_cmplx(tmp[i],tmp[i+n1*nw]); /* first/second in 3rd dimension is real/imag */
    }

        printf("nw before this line =%d\n",nw);
        int i, i1, l2;

        kiss_fft_cpx *d, *pp;
        kiss_fft_cfg itfft;

        nw = 2*kiss_fft_next_fast_size((n1+1)/2);
        itfft = kiss_fft_alloc(nw,1,NULL,NULL);

        pp = (kiss_fft_cpx*) np_complexalloc(nw);
        d =  (kiss_fft_cpx*) np_complexalloc(nw);

        printf("nw after this line =%d\n",nw);
        
        for (i=0; i < nw; i++) {
            pp[i].r = 0.;
            pp[i].i = 0.;
        }

        for (i1=nflo; i1 <= nfhi; i1++) {
            for (i=0; i < n1; i++) {
                pp[i1-nflo].r += crealf(outp[(i1-nflo)*n1+i]);
                pp[i1-nflo].i += cimagf(outp[(i1-nflo)*n1+i]);
            }
        }
     
        l2 = (nw+1)/2;
        for (i=1; i < l2; i++) {
            pp[i].r /= 2.;
            pp[i].i /= 2.;
        }
        l2 = nw/2+1;
        for (i=l2; i < nw; i++) {
            pp[i].r = pp[nw-i].r;
            pp[i].i = -pp[nw-i].i;
        }
        kiss_fft_stride(itfft,pp,d,1);
            
        for (i=0; i < n1; i++) {
            inp[i] = d[i].r/n1;
        }
        free(pp);
        free(d);
        
    /*sub-function goes here*/
    }
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];
    
    if(!inv)
    {

        for(i=0;i<ndata*nf;i++)
        {
            tmp[i]=outp[i].r;
            tmp[i+ndata*nf]=outp[i].i;
        }
    dims[0]=ndata*nf*2+3;dims[1]=1;
    vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
    for(i=0;i<ndata*nf*2;i++)
        (*((float*)PyArray_GETPTR1(vecout,i))) = tmp[i];
    printf("w0=%g,dw=%g,nw=%d\n",w0,dw,nf);
    (*((float*)PyArray_GETPTR1(vecout,0+ndata*nf*2))) = w0;
    (*((float*)PyArray_GETPTR1(vecout,1+ndata*nf*2))) = dw;
    (*((float*)PyArray_GETPTR1(vecout,2+ndata*nf*2))) = nf;
    }else{
    
    dims[0]=n1;dims[1]=1;
    /* Parse tuples separately since args will differ between C fcns */
    /* Make a new double vector of same dimension */
    vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
    for(i=0;i<dims[0];i++)
        (*((float*)PyArray_GETPTR1(vecout,i))) = inp[i];
    }
    
    return PyArray_Return(vecout);
    
}

/*documentation for each functions.*/
static char ftfacfun_document[] = "Document stuff for this C module...";

/*defining our functions like below:
  function_name, function, METH_VARARGS flag, function documents*/
static PyMethodDef functions[] = {
  {"aps3dc", aps3dc, METH_VARARGS, ftfacfun_document},
  {NULL, NULL, 0, NULL}
};

/*initializing our module informations and settings in this structure
for more informations, check head part of this file. there are some important links out there.*/
static struct PyModuleDef ftfacfunModule = {
  PyModuleDef_HEAD_INIT, /*head informations for Python C API. It is needed to be first member in this struct !!*/
  "apscfun",  /*module name (FFT-based time-frequency analysis)*/
  NULL, /*means that the module does not support sub-interpreters, because it has global state.*/
  -1,
  functions  /*our functions list*/
};

/*runs while initializing and calls module creation function.*/
PyMODINIT_FUNC PyInit_ftfacfun(void){
  
    PyObject *module = PyModule_Create(&ftfacfunModule);
    import_array();
    return module;
}
