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

/* Laplacian coefficients */
static float c0=-30./12.,c1=+16./12.,c2=- 1./12.;

/*fft utitlies to be removed*/
int fft2_init(bool cmplx1        /* if complex transform */,
	      int pad1           /* padding on the first axis */,
	      int nx,   int ny   /* input data size */, 
	      int *nx2, int *ny2 /* padded data size */);
		
void fft2(float *inp      /* [n1*n2] */, 
	  np_complex *out /* [nk*n2] */);
	  
void ifft2(float *out      /* [n1*n2] */, 
	   np_complex *inp /* [nk*n2] */);

void fft2_finalize();

int fdm2d(float **wvfld, float **dat, float **dat_v, float *img, float *vel, psmpar par, bool tri)
/*< acoustic finite difference method >*/
{
    /*survey parameters*/
    int   nx, nz;
    float dx, dz;
    int   ns;
    int   *spx, *spz;
    int   gpz, gpx, gplx;
    int   gpz_v, gpx_v, gpl_v;
    int   jsnap;
    /*fft related*/
    bool  cmplx;
    int   pad1;
    /*absorbing boundary*/
    bool abc;
    int nbt, nbb, nblx, nbrx;
    float ct,cb,clx,crx;
    /*source parameters*/
    int src; /*source type*/
    int nt;
    float dt,*f0,*t0,*A;
    /*misc*/
    bool verb, ps;
    float vref;
    
    int nx1, nz1; /*domain of interest*/
    int it,iz,ik,ix,i,j;     /* index variables */
    int nk,nzx,nz2,nx2,nzx2,nkz,nth;
    int it1, it2, its;
    float dkx,dkz,kx0,kz0,vref2,kx,kz,k,t;
    float c, old;

    /*wave prop arrays*/
    float *vv;
    np_complex *cwave,*cwavem;
    float *wave,*curr,*prev,*lapl;

    /*source*/
    float **rick;
    float freq;
    int fft_size;

    /*passing the parameters*/
    nx    = par->nx;
    nz    = par->nz;
    dx    = par->dx;
    dz    = par->dz;
    ns= par->ns;
    spx   = par->spx;
    spz   = par->spz;
    gpz   = par->gpz;
    gpx   = par->gpx;
    gplx   = par->gplx;
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
    ct    = par->ct;
    cb    = par->cb;
    clx    = par->clx;
    crx    = par->crx;
    src   = par->src;
    nt    = par->nt;
    dt    = par->dt;
    f0    = par->f0;
    t0    = par->t0;
    A     = par->A;
    verb  = par->verb;
    ps    = par->ps;
    vref  = par->vref;
    
    nz1 = nz-nbt-nbb;
    nx1 = nx-nblx-nbrx;

    nk = fft2_init(cmplx,pad1,nz,nx,&nz2,&nx2);
    nzx = nz*nx;
    nzx2 = nz2*nx2;
    
    dkz = 1./(nz2*dz); kz0 = (cmplx)? -0.5/dz:0.;
    dkx = 1./(nx2*dx); kx0 = -0.5/dx;
    nkz = (cmplx)? nz2:(nz2/2+1);

    printf("dkz=%f,dkx=%f,kz0=%f,kx0=%f\n",dkz,dkx,kz0,kx0);
    printf("nk=%d,nkz=%d,nz2=%d,nx2=%d\n",nk,nkz,nz2,nx2);

    if(abc)
      abc_init(nz,nx,1,nz2,nx2,1,nbt,nbb,nblx,nbrx,0,0,ct,cb,clx,crx,clx,crx);

    /* allocate and read/initialize arrays */
    vv     = np_floatalloc(nzx); 
    lapl   = np_floatalloc(nk);
    wave   = np_floatalloc(nzx2);
    curr   = np_floatalloc(nzx2);
    prev   = np_floatalloc(nzx2);
    cwave  = np_complexalloc(nk);
    cwavem = np_complexalloc(nk);
    
    if (!tri && src==0) {
      rick = np_floatalloc2(nt,ns);
      for (i=0; i<ns; i++) {
	for (it=0; it<nt; it++) {
	  rick[i][it] = 0.f;
	}
	rick[i][(int)(t0[i]/dt)] = A[i]; /*time delay*/
	freq = f0[i]*dt;           /*peak frequency*/
	fft_size = 2*kiss_fft_next_fast_size((nt+1)/2);
	ricker_init(fft_size, freq, 0);
	np_freqfilt(nt,rick[i]);
	ricker_close();
      }
    } else{
    	 rick = NULL;}

    for (iz=0; iz < nzx; iz++) {
        vv[iz] = vel[iz]*vel[iz]*dt*dt;
    }
    vref *= dt;
    vref2 = vref*vref;
    for (iz=0; iz < nzx2; iz++) {
	curr[iz] = 0.;
	prev[iz] = 0.;
    }

    /* constructing the pseudo-analytical op */
    for (ix=0; ix < nx2; ix++) {
	kx = kx0+ix*dkx;
	for (iz=0; iz < nkz; iz++) {
	    kz = kz0+iz*dkz;
	    k = 2*np_PI*hypot(kx,kz);
	    if (ps) lapl[iz+ix*nkz] = -k*k;
	    else lapl[iz+ix*nkz] = 2.*(cos(vref*k)-1.)/vref2;
	}
    }

    if (tri) { /* time-reversal propagation */
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
      
        if(verb) printf("it=%d/%d;\n",it,nt);

	/* matrix multiplication */
/*	fft2(curr,cwave);

	for (ik = 0; ik < nk; ik++) {
#ifdef SF_HAS_COMPLEX_H
	  cwavem[ik] = cwave[ik]*lapl[ik];
#else
	  cwavem[ik] = np_cmul(cwave[ik],lapl[ik]);
#endif
	}
	
	ifft2(wave,cwavem);*/
	
	/*wave   = np_floatalloc(nzx2);*/
	/*nzx2 = nz2*nx2;*/
	float idz,idx;
    idz = 1/(dz*dz);
    idx = 1/(dx*dx);
	for (iz=2; iz<nz2-2; iz++) {
	    for (ix=2; ix<nx2-2; ix++) {
		wave[ix*nz2+iz] = 
/*		    c0* uo[ix  ][iz  ] * (idx+idz) + 
		    c1*(uo[ix-1][iz  ] + uo[ix+1][iz  ])*idx +
		    c2*(uo[ix-2][iz  ] + uo[ix+2][iz  ])*idx +
		    c1*(uo[ix  ][iz-1] + uo[ix  ][iz+1])*idz +
		    c2*(uo[ix  ][iz-2] + uo[ix  ][iz+2])*idz;	  */
		    c0* curr[ix*nz2+iz] * (idx+idz) + 
		    c1*(curr[(ix-1)*nz2+iz] + curr[(ix+1)*nz2+iz])*idx +
		    c2*(curr[(ix-2)*nz2+iz] + curr[(ix+2)*nz2+iz])*idx +
		    c1*(curr[(ix)*nz2+iz-1] + curr[(ix)*nz2+iz+1])*idz +
		    c2*(curr[(ix)*nz2+iz-2] + curr[(ix)*nz2+iz+2])*idz;	  
	    }
	}	
	
	for (ix = 0; ix < nx; ix++) {
	    for (iz=0; iz < nz; iz++) {
		i = iz+ix*nz;  /* original grid */
		j = iz+ix*nz2; /* padded grid */

		old = c = curr[j];
		c += c - prev[j];
		prev[j] = old;
		c += wave[j]*vv[i];
		curr[j] = c;
	    }
	}

	if (tri) {
	  /* inject data */
	  if (NULL!=dat) {
	    for (ix = 0; ix < gplx; ix++) {
	      curr[gpz+(ix+gpx)*nz2] += vv[gpz+(ix+gpx)*nz]*dat[ix][it];
	    }
	  }
	  if (NULL!=dat_v) {
	    for (iz = 0; iz < gpl_v; iz++) {
	      curr[gpz_v+iz+(gpx_v)*nz2] += vv[gpz_v+iz+(gpx_v)*nz]*dat_v[iz][it];
	    }
	  }
	} else {
	  t = it*dt;
	  for (i=0; i<ns; i++) {
	    for(ix=-1;ix<=1;ix++) {
	      for(iz=-1;iz<=1;iz++) {
		ik = (spz[i]+nbt)+iz+nz*(spx[i]+nblx+ix);
		j = (spz[i]+nbt)+iz+nz2*(spx[i]+nblx+ix);
		if (src==0) {
		  curr[j] += vv[ik]*rick[i][it]/(abs(ix)+abs(iz)+1);
		} else {
		  curr[j] += vv[ik]*Ricker(t, f0[i], t0[i], A[i])/(abs(ix)+abs(iz)+1);
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
	    for (ix = 0; ix < gplx; ix++) {
	      dat[ix][it] = curr[gpz+(ix+gpx)*nz2];
	    }
	  }
	  if (NULL!=dat_v) {
	    for (iz = 0; iz < gpl_v; iz++) {
	      dat_v[iz][it] = curr[gpz_v+iz+(gpx_v)*nz2];
	    }
	  }
	}

	/* save wavefield */
	if (jsnap > 0 && it%jsnap==0) {
	  for (ix=0; ix<nx1; ix++) {
	    for (iz=0; iz<nz1; iz++) {
	      i = iz + nz1*ix;
	      j = iz+nbt + (ix+nblx)*nz2; /* padded grid */
	      wvfld[it/jsnap][i] = curr[j];
	    }
	  }
	}
    }
    
    if(verb) printf(".\n");

    if (tri) {
      for (ix = 0; ix < nx1; ix++) {
	for (iz = 0; iz < nz1; iz++) {
	  img[iz + nz1*ix] = curr[iz+nbt + (ix+nblx)*nz2];
	}
      }
    }

    /*free up memory*/
    fft2_finalize();
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
