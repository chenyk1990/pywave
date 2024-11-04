/*below is the including part*/
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "wave_alloc.h"
#include "wave_ricker.h"
#include "wave_abc.h"
#include "wave_komplex.h"
#include "wave_psp.h"
#include "wave_freqfilt.h"
#define SF_PI (3.14159265358979323846264338328)


int fft3_init(bool cmplx1        /* if complex transform */,
	      int pad1           /* padding on the first axis */,
	      int nx,   int ny,   int nz   /* axis 1,2,3; input data size */, 
	      int *nx2, int *ny2, int *nz2 /* axis 1,2,3; padded data size */);
		
void fft3(float *inp      /* [n1*n2*n3] */, 
	  np_complex *out /* [nk*n2*n3] */);
	  
void ifft3(float *out      /* [n1*n2*n3] */, 
	   np_complex *inp /* [nk*n2*n3] */);
	  
int psm(float **wvfld, float ***dat, float **dat_v, float *img, float *vel, psmpar par, bool tri)
/*< pseudo-spectral method >*/
{
	float sum;
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
    np_complex *cwave,*cwavem;
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

// #ifdef _OPENMP
// #pragma omp parallel
//     {
//       nth = omp_get_num_threads();
//     }
// #else
//     nth = 1;
// #endif
//     if (verb) np_warning(">>>> Using %d threads <<<<<", nth);

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
    
    printf("cmplx=%d,dkz=%f,dkx=%f,dky=%f,kz0=%f,kx0=%f,ky0=%f\n",cmplx,dkz,dkx,dky,kz0,kx0,ky0);
    printf("nk=%d,nkz=%d,nz2=%d,nx2=%d,ny2=%d\n",nk,nkz,nz2,nx2,ny2);

    if(abc)
      abc_init(nz,nx,ny,nz2,nx2,ny2,nbt,nbb,nblx,nbrx,nbly,nbry,ct,cb,clx,crx,cly,cry);

    /* allocate and read/initialize arrays */
    vv     = np_floatalloc(nzxy); 
    lapl   = np_floatalloc(nk);
    wave   = np_floatalloc(nzxy2);
    curr   = np_floatalloc(nzxy2);
    prev   = np_floatalloc(nzxy2);
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
      
        if(verb) printf("it=%d/%d;\n",it,nt);

	/* matrix multiplication */
	fft3(curr,cwave);

	for (ik = 0; ik < nk; ik++) {
// #ifdef SF_HAS_COMPLEX_H
// 	  cwavem[ik] = cwave[ik]*lapl[ik];
// #else
	  cwavem[ik] = np_crmul(cwave[ik],lapl[ik]);
// #endif
	}
	
	ifft3(wave,cwavem);
	
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iy,ix,iz,i,j,old,c)
// #endif
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
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iy,ix)
// #endif
	    for (iy = 0; iy < gply; iy++) {
	    for (ix = 0; ix < gplx; ix++) {
	      curr[gpz+(ix+gpx)*nz2+(iy+gpy)*nz2*nx2] += vv[gpz+(ix+gpx)*nz+(iy+gpy)*nz*nx]*dat[iy][ix][it];
	    }
	    }
	  }
	  if (NULL!=dat_v) {
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iz)
// #endif
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
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(ix)
// #endif
	    for (iy = 0; iy < gply; iy++) {
	    for (ix = 0; ix < gplx; ix++) {
	      dat[iy][ix][it] = curr[gpz+(ix+gpx)*nz2+(iy+gpy)*nz2*nx2];
	    }
	    }
	  }
	  if (NULL!=dat_v) {
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iz)
// #endif
	    for (iz = 0; iz < gpl_v; iz++) {
	      dat_v[iz][it] = curr[gpz_v+iz+(gpx_v)*nz2];
	    }
	  }
	}
		
	/* save wavefield */
	if (jsnap > 0 && it%jsnap==0) {
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iy,ix,iz,i,j)
// #endif
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
    if(verb) printf(".\n");
    if (tri) {
// #ifdef _OPENMP
// #pragma omp parallel for default(shared) private(iy,ix,iz)
// #endif
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