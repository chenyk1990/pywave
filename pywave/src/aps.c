#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#define np_MAX(a,b) ((a) < (b) ? (b) : (a))
#define np_MIN(a,b) ((a) < (b) ? (a) : (b))
#define np_MAX_DIM 9
#define np_PI (3.14159265358979323846264338328)

#include "wave_alloc.h"
#include "wave_komplex.h"
#include "wave_psp.h"
#include "wave_abc.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

static PyObject *aps3dc(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf2=NULL;
    
	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim;
    float *data;
    
    int niter,verb,rect0,n1,ntw,opt=0,sym,window;
    float dt,alpha,ot;
    int ifb,inv;
    
    int   nx, ny, nz;
    float dx, dy, dz;
    int   ns;
//     int   *spx, *spy, *spz;
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
    float *f0,*t0,*A;
    /*misc*/
    int ps, tri; /*tri: time-reversal imaging*/
    float vref;
    int i;

    psmpar par;
    int nx1, ny1, nz1; /*domain of interest*/
    int it;
    float *vel,*vel2,***dat,**dat_v,**wvfld,*img; /*velocity profile*/
    
    float oz,ox,oy; 
    int ifsnaps;
    
    /*data and parameters interface*/
	PyArg_ParseTuple(args, "OOiiiiiiiiiiiffffffff", &f1,&f2,&tri,&nt,&nx,&ny,&nz,&ns,&verb,&jsnap,&ifsnaps,&abc,&nbt,&ct,&dt,&ox,&dx,&oy,&dy,&oz,&dz);

// 	source=np.concatenate([sx,sy,sz,f,t,A],axis=0,dtype='float32'); #remember: source size is ns*6
// 	
// 	dout=aps3dc(vel,source,tri,nt,nx,ny,nz,ns,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);
	
// 	dout=aps3dc(vel,tri,nt,nx,ny,nz,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);

	printf("tri=%d,nt=%d,nx=%d,ny=%d,nz=%d,ns=%d\n",tri,nt,nx,ny,nz,ns);
	printf("verb=%d,jsnap=%d,ifsnaps=%d,abc=%d,nbt=%d\n",verb,jsnap,ifsnaps,abc,nbt);
	printf("ct=%g,dt=%g,ox=%g,dx=%g,oy=%g,dy=%g,oz=%g,dz=%g\n",ct,dt,ox,dx,oy,dy,oz,dz);
	
// 	int nw;
//     int i1, iw, i2, n2, n12, n1w;
//     int *rect;
//     float t, w, w0, dw, mean=0.0f;
//     float *mm, *ww;
//     int i, j, m;
//     if (ntw%2 == 0)
//         ntw = (ntw+1);
//     m = (ntw-1)/2;

	
// 	if(tri==0)
// 	{ndata=nx*ny*nz;}
// 	else
// 	{ndata=nx*ny*nt;}
	ndata=nx*ny*nz;

// 	printf("n1=%d,nw=%d,nt=%d 22\n",n1,nw,nt);
	
//     if(opt)
// 	    nt = 2*kiss_fft_next_fast_size((ntw+1)/2);
//     else
//         nt=ntw;
//     if (nt%2) nt++;
//     nw = nt/2+1;
//     dw = 1./(nt*dt);
// 	w0 = 0.;

// 	printf("n1=%d,nw=%d,nt=%d 33\n",n1,nw,nt);
// 	printf("dw=%g,w0=%g,dt=%g\n",dw,w0,dt);
    
//     kiss_fftr_cfg cfg;
//     kiss_fft_cpx *pp, ce, *outp;
//     float *p, *inp, *tmp;
//     float wt, shift;

    
//     p = np_floatalloc(nt);
//     pp = (kiss_fft_cpx*) np_complexalloc(nw);
//     cfg = kiss_fftr_alloc(nt,inv?1:0,NULL,NULL);
//     wt = sym? 1./sqrtf((float) nt): 1.0/nt;
// 
//     printf("sym=%d,wt=%g\n",sym,wt);
    
//     inp = np_floatalloc(n1);
//     tmp = np_floatalloc(n1*nw*2);
//     outp = (kiss_fft_cpx*) np_complexalloc(n1*nw);

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

// 	data  = (float*)malloc(ndata * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    

        
/** Part V: main program ********/
// int main(int argc, char* argv[])
// {

    /*survey parameters*/

//     np_file Fi,Fo,Fd,Fd_v,snaps,Fvpad; /* I/O files */
//     np_axis az,ax,ay; /* cube axes */

//     np_init(argc,argv);

//     if (!np_getint("jsnap",&jsnap)) jsnap=0; /* interval for snapshots */
//     if (!np_getbool("cmplx",&cmplx)) cmplx=true; /* use complex fft */
//     if (!np_getint("pad1",&pad1)) pad1=1; /* padding factor on the first axis */
//     if(!np_getbool("abc",&abc)) abc=false; /* absorbing flag */
    
    cmplx=0;
// 	jsnap=0;
	pad1=1;
	abc=1;
	src=0;
	
    if (abc) {
//       if(!np_getint("nbt",&nbt)) np_error("Need nbt!");
//       if(!np_getint("nbb",&nbb)) nbb = nbt;
//       if(!np_getint("nblx",&nblx)) nblx = nbt;
//       if(!np_getint("nbrx",&nbrx)) nbrx = nbt;
//       if(!np_getint("nbly",&nbly)) nbly = nbt;
//       if(!np_getint("nbry",&nbry)) nbry = nbt;
//       if(!np_getfloat("ct",&ct)) np_error("Need ct!");
//       if(!np_getfloat("cb",&cb)) cb = ct;
//       if(!np_getfloat("clx",&clx)) clx = ct;
//       if(!np_getfloat("crx",&crx)) crx = ct;
//       if(!np_getfloat("cly",&cly)) cly = ct;
//       if(!np_getfloat("cry",&cry)) cry = ct;
	nbb=nbt;
	nblx = nbt;
	nbrx = nbt;
	nbly = nbt;
	nbry = nbt;
	cb=ct;
	clx = ct;
	crx = ct;
	cly = ct;
	cry = ct;
	
    } else {
      nbt = 0; nbb = 0; nblx = 0; nbrx = 0; nbly = 0; nbry = 0;
      ct = 0; cb = 0; clx = 0; crx = 0; cly = 0; cry = 0;
    }
//     if (!np_getbool("verb",&verb)) verb=false; /* verbosity */
//     if (!np_getbool("ps",&ps)) ps=false; /* use pseudo-spectral */
//     if (ps) np_warning("Using pseudo-spectral...");
//     else np_warning("Using pseudo-analytical...");
//     if (!np_getbool("tri",&tri)) tri=false; /* if choose time reversal imaging */
//     if (tri) np_warning("Time-reversal imaging");
//     else np_warning("Forward modeling");
//     if (!np_getfloat("vref",&vref)) vref=1500; /* reference velocity (default using water) */

    /* setup I/O files */
//     Fi = np_input ("in");
//     Fo = np_output("out");
    
    int   *spx, *spy, *spz;
    if (tri) {
//       gplx = -1;
//       gply = -1;
//       gpl_v = -1;
//       if (NULL==np_getstring("dat") && NULL==np_getstring("dat_v"))
// 	np_error("Need Data!");
//       if (NULL!=np_getstring("dat")) {
// 	Fd = np_input("dat");
// 	np_histint(Fd,"n1",&nt);
// 	np_histfloat(Fd,"d1",&dt);
// 	np_histint(Fd,"n2",&gplx);
// 	np_histint(Fd,"n3",&gply);
//       } else Fd = NULL;
//       if (NULL!=np_getstring("dat_v")) {
// 	Fd_v = np_input("dat_v");
// 	np_histint(Fd_v,"n1",&nt);
// 	np_histfloat(Fd_v,"d1",&dt);
// 	np_histint(Fd_v,"n2",&gpl_v);
//       } else Fd_v = NULL;
      src = -1; ns = -1;
      spx = NULL; spy = NULL; spz = NULL;
      f0 = NULL; t0 = NULL; A = NULL;
    } else {
//       Fd = NULL;
//       if (!np_getint("nt",&nt)) np_error("Need nt!");
//       if (!np_getfloat("dt",&dt)) np_error("Need dt!");
//       if (!np_getint("gplx",&gplx)) gplx = -1; /* geophone length X*/
//       if (!np_getint("gply",&gply)) gply = -1; /* geophone length Y*/
//       if (!np_getint("gpl_v",&gpl_v)) gpl_v = -1; /* geophone height */
//       if (!np_getint("src",&src)) src=0; /* source type */
//       if (!np_getint("ns",&ns)) ns=1; /* source type */

//     int   *spx, *spy, *spz;
//     float   *spx, *spy, *spz;
	  printf("ns=%d\n",ns);
      spx = np_intalloc(ns);
      spy = np_intalloc(ns);
      spz = np_intalloc(ns);
//       spx = np_floatalloc(ns);
//       spy = np_floatalloc(ns);
//       spz = np_floatalloc(ns);
      f0  = np_floatalloc(ns);
      t0  = np_floatalloc(ns);
      A   = np_floatalloc(ns);
	float tmp;
    for (i=0; i<ns; i++)
    {
        tmp=*((float*)PyArray_GETPTR1(arrf2,i));
        spx[i]=tmp;
        tmp=*((float*)PyArray_GETPTR1(arrf2,3+i));
        spy[i]=tmp;
        tmp=*((float*)PyArray_GETPTR1(arrf2,3*2+i));
        spz[i]=tmp;
        f0[i]=*((float*)PyArray_GETPTR1(arrf2,3*3+i));
        t0[i]=*((float*)PyArray_GETPTR1(arrf2,3*4+i));
        A[i]=*((float*)PyArray_GETPTR1(arrf2,3*5+i));
    }
    
    printf("There are %d sources to be simulated\n",ns);
    for(i=0;i<ns;i++)
    {
    printf("spx[%d]=%d\n",i,spx[i]);
    printf("spy[%d]=%d\n",i,spy[i]);
    printf("spz[%d]=%d\n",i,spz[i]);
    printf("f0[%d]=%g\n",i,f0[i]);
    printf("t0[%d]=%g\n",i,t0[i]);
    printf("A[%d]=%g\n",i,A[i]);
    }
    
    }
//       if (!np_getints("spx",spx,ns)) np_error("Need spx!"); /* shot position x */
//       if (!np_getints("spy",spy,ns)) np_error("Need spy!"); /* shot position y */
//       if (!np_getints("spz",spz,ns)) np_error("Need spz!"); /* shot position z */
//       if (!np_getfloats("f0",f0,ns)) np_error("Need f0! (e.g. 30Hz)");   /*  wavelet peak freq */
//       if (!np_getfloats("t0",t0,ns)) np_error("Need t0! (e.g. 0.04s)");  /*  wavelet time lag */
//       if (!np_getfloats("A",A,ns)) np_error("Need A! (e.g. 1)");     /*  wavelet amplitude */
//     }
//     if (!np_getint("gpx",&gpx)) gpx = -1; /* geophone position x */
//     if (!np_getint("gpy",&gpy)) gpy = -1; /* geophone position y */
//     if (!np_getint("gpz",&gpz)) gpz = -1; /* geophone position z */
//     if (!np_getint("gpx_v",&gpx_v)) gpx_v = -1; /* geophone position x */
//     if (!np_getint("gpy_v",&gpy_v)) gpy_v = -1; /* geophone position y */
//     if (!np_getint("gpz_v",&gpz_v)) gpz_v = -1; /* geophone position z */
// 
//     if (np_FLOAT != np_gettype(Fi)) np_error("Need float input");

    /* Read/Write axes */
//     az = np_iaxa(Fi,1); nz = np_n(az); dz = np_d(az);
//     ax = np_iaxa(Fi,2); nx = np_n(ax); dx = np_d(ax);
//     ay = np_iaxa(Fi,3); ny = np_n(ay); dy = np_d(ay);


    /*change on Jun 2022, YC*/
    nz1 = nz;
    nx1 = nx;
    ny1 = ny;
    nz = nz+nbt+nbb;
    nx = nx+nblx+nbrx;
    ny = ny+nbly+nbry;
    /*change on Jun 2022, YC*/
    
//     if(verb)np_warning("ny=%d,nbly=%d,nbry=%d",ny,nbly,nbry);
//     if(verb)np_warning("nz1=%d,nx1=%d,ny1=%d",nz1,nx1,ny1);
//     if (gpx==-1) gpx = nblx;
//     if (gpy==-1) gpy = nbly;
//     if (gpz==-1) gpz = nbt;
//     if (gplx==-1) gplx = nx1;
//     if (gply==-1) gply = ny1;
	gplx = nx1;
	gply = ny1;
	gpl_v = nz1;
	gpx=nblx;
	gpy=nbly;
	gpz=nbt;
	vref=1500;
	ps=1;
//     if (gpx_v==-1) gpx_v = nblx;
//     if (gpy_v==-1) gpy_v = nbly;
//     if (gpz_v==-1) gpz_v = nbt;
//     if (gpl_v==-1) gpl_v = nz1;
    ntsnap=0;
    if (jsnap)
        for (it=0;it<nt;it++)
            if (it%jsnap==0) ntsnap++;
            
//     if (tri) { /*output final wavefield*/
//       np_setn(az,nz1);
//       np_setn(ax,nx1);
//       np_setn(ay,ny1);
//       np_oaxa(Fo,az,1);
//       np_oaxa(Fo,ax,2);
//       np_oaxa(Fo,ay,3);   
//       np_settype(Fo,np_FLOAT);
//     } else { /*output data*/
//       np_setn(ax,gplx);
//       np_setn(ay,gply);
//       np_putint(Fo,"n3",gply);
//       np_warning("ny=%d,nbly=%d,nbry=%d",ny,nblx,nbly);
//       np_warning("gplx=%d,gply=%d",gplx,gply);
//       /*output horizontal data is mandatory*/
//       np_putint(Fo,"n1",nt);
//       np_putfloat(Fo,"d1",dt);
//       np_putfloat(Fo,"o1",0.);
//       np_putstring(Fo,"label1","Time");
//       np_putstring(Fo,"unit1","s");
//       np_oaxa(Fo,ax,2);
//       np_settype(Fo,np_FLOAT);
//       /*output vertical data is optional*/
//       if (NULL!=np_getstring("dat_v")) {
// 	Fd_v = np_output("dat_v");
// 	np_setn(az,gpl_v);
// 	np_putint(Fd_v,"n1",nt);
// 	np_putfloat(Fd_v,"d1",dt);
// 	np_putfloat(Fd_v,"o1",0.);
// 	np_putstring(Fd_v,"label1","Time");
// 	np_putstring(Fd_v,"unit1","s");
// 	np_oaxa(Fd_v,az,2);
// 	np_putint(Fd_v,"n3",1);
// 	np_settype(Fd_v,np_FLOAT);	
//       } else Fd_v = NULL;
//     }

//     if (NULL!=np_getstring("vpad")) {
// 	   Fvpad=np_output("vpad");
//        np_putint(Fvpad,"n1",nz);
//        np_putint(Fvpad,"n2",nx);
//        np_putint(Fvpad,"n3",ny);
//        ifvpad=true;
//       } else 
//       {
//       Fvpad = NULL;
       ifvpad=false;
//       }
      
//     if (jsnap > 0) {
// 	snaps = np_output("snaps");
// 	/* (optional) snapshot file */
// 	np_setn(az,nz1);
// 	np_setn(ax,nx1);
// 	np_setn(ay,ny1);
// 	np_oaxa(snaps,az,1);
// 	np_oaxa(snaps,ax,2);
// 	np_oaxa(snaps,ay,3);
// 	np_putint(snaps,"n4",ntsnap);
// 	np_putfloat(snaps,"d4",dt*jsnap);
// 	np_putfloat(snaps,"o4",0.);
// 	np_putstring(snaps,"label4","Time");
// 	np_putstring(snaps,"unit4","s");
//     } else snaps = NULL;

    par = (psmpar) np_alloc(1,sizeof(*par));
    vel = np_floatalloc(nz1*ny1*nx1); 	/*change on Jun 2022, YC*/
    vel2= np_floatalloc(nz*ny*nx); 		/*change on Jun 2022, YC*/

    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("input data done, ndata=%d\n",ndata);
	
//     if (tri && NULL==Fd) {dat = NULL;  }
//     else { dat = np_floatalloc3(nt,gplx,gply);}

	if(tri)
	{
	printf("Doing TRI, reading data\n");
	dat = np_floatalloc3(nt,gplx,gply);
    for (i=0; i<nt*gplx*gply; i++)
    {
        dat[0][0][i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
	printf("Doing TRI, reading data done\n");
    }
// 	for(i=0;i<ndata;i++)
// // 	dat[0][0][i]=0;
// 	printf("vel[%d]=%g\n",i,vel[i]);

	if(tri==0)
	{
	dat=np_floatalloc3(nt,gplx,gply);

	for(i=0;i<nt*gplx*gply;i++)
	dat[0][0][i]=0;
	}
	
	
	int ifvdata=0;
	if(ifvdata==1)dat_v = np_floatalloc2(nt,gpl_v);
    else dat_v = NULL;

	
	
    if (tri) img = np_floatalloc(nz1*ny1*nx1);
    else img = NULL;

    if (jsnap>0) wvfld = np_floatalloc2(nx1*ny1*nz1,ntsnap);
    else wvfld = NULL;
    

//     np_floatread(vel,nz1*ny1*nx1,Fi);
	vel_expand(vel,vel2,nz1,nx1,ny1,nbt,nbb,nblx,nbrx,nbly,nbry);  /*change on Jun 2022, YC*/
// 	for(i=0;i<nz1*nx1*ny1;i++)
// // 	dat[0][0][i]=0;
// 	printf("vel2[%d]=%g\n",i,vel2[i]);

//     if (tri) {
// //       if (NULL!=Fd)   np_floatread(dat[0][0],gplx*gply*nt,Fd);
// //       if (NULL!=Fd_v) np_floatread(dat_v[0],gpl_v*nt,Fd_v);
// 
// 
//     }

// 	float sum=0;
// 	for(i=0;i<ndata;i++)
// 	sum=sum+vel[i];

// 	for(i=0;i<ndata;i++)
// 	if(vel[i]!=vel[0])
// 	printf("vel[%d]=%g\n",i,vel[i]);
	
// 	printf("vel sum = %g\n",sum);

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

	printf("par->nx=%d,par->ny=%d,par->nz=%d\n",par->nx,par->ny,par->nz);
	printf("par->dx=%g,par->dy=%g,par->dz=%g\n",par->dx,par->dy,par->dz);
	printf("par->ct=%g,par->cb=%g,par->clx=%g,par->cly=%g\n",par->ct,par->cb,par->clx,par->cly);
	
	
	printf("par->verb=%d,par->ps=%d,par->vref=%g\n",par->verb,par->ps,par->vref);
		
    /*do the work*/
    psm(wvfld, dat, dat_v, img, vel2, par, tri);
	
    if (tri) {
//       np_floatwrite(img,nz1*ny1*nx1,Fo);
    } else {
//       np_floatwrite(dat[0][0],gplx*gply*nt,Fo);
//       if (NULL!=Fd_v)
// 	np_floatwrite(dat_v[0],gpl_v*nt,Fd_v);
    }

//     if (jsnap>0)
//       np_floatwrite(wvfld[0],nz1*nx1*ny1*ntsnap,snaps);
      
//     if(ifvpad)
//     	np_floatwrite(vel2,nz*nx*ny,Fvpad);
      
//     exit (0);
// }
	
	/*sub-function goes here*/
//     }
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];

	int nwfd;
	if(jsnap>0)
	{nwfd=nz1*nx1*ny1*ntsnap;
	printf("ntsnap=%d\n",ntsnap);
	}
	else
	nwfd=0;
	
	dims[0]=nt*nx1*ny1+nwfd;dims[1]=1;
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<nt*nx1*ny1;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = dat[0][0][i];
		
	if(jsnap>0)
	{
	
	for(i=0;i<nwfd;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+nt*nx1*ny1))) = wvfld[0][i];
		
	}
// 	printf("w0=%g,dw=%g,nw=%d\n",w0,dw,nw);
// 	(*((float*)PyArray_GETPTR1(vecout,0+ndata*nw*2))) = w0;
// 	(*((float*)PyArray_GETPTR1(vecout,1+ndata*nw*2))) = dw;
// 	(*((float*)PyArray_GETPTR1(vecout,2+ndata*nw*2))) = nw;
	
// 	}else{
	
	dims[0]=n1;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
// 	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
// 	for(i=0;i<dims[0];i++)
// 		(*((float*)PyArray_GETPTR1(vecout,i))) = inp[i];
// 	}
	
	return PyArray_Return(vecout);
	
}

static PyObject *aps2dc(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf2=NULL;
    
	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim;
    float *data;
    
    int niter,verb,rect0,n1,ntw,opt=0,sym,window;
    float dt,alpha,ot;
    int ifb,inv;
    
    int   nx, ny, nz;
    float dx, dy, dz;
    int   ns;
//     int   *spx, *spy, *spz;
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
    float *f0,*t0,*A;
    /*misc*/
    int ps, tri; /*tri: time-reversal imaging*/
    float vref;
    int i;

    psmpar par;
    int nx1, ny1, nz1; /*domain of interest*/
    int it;
    float *vel,*vel2,**dat,**dat_v,**wvfld,*img; /*velocity profile*/
    
    float oz,ox,oy; 
    int ifsnaps;
    
    /*data and parameters interface*/
	PyArg_ParseTuple(args, "OOiiiiiiiiiiffffff", &f1,&f2,&tri,&nt,&nx,&nz,&ns,&verb,&jsnap,&ifsnaps,&abc,&nbt,&ct,&dt,&ox,&dx,&oz,&dz);

// 	source=np.concatenate([sx,sy,sz,f,t,A],axis=0,dtype='float32'); #remember: source size is ns*6
// 	
// 	dout=aps3dc(vel,source,tri,nt,nx,ny,nz,ns,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);
	
// 	dout=aps3dc(vel,tri,nt,nx,ny,nz,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);

	printf("tri=%d,nt=%d,nx=%d,nz=%d,ns=%d\n",tri,nt,nx,nz,ns);
	printf("verb=%d,jsnap=%d,ifsnaps=%d,abc=%d,nbt=%d\n",verb,jsnap,ifsnaps,abc,nbt);
	printf("ct=%g,dt=%g,ox=%g,dx=%g,oz=%g,dz=%g\n",ct,dt,ox,dx,oz,dz);
	
// 	int nw;
//     int i1, iw, i2, n2, n12, n1w;
//     int *rect;
//     float t, w, w0, dw, mean=0.0f;
//     float *mm, *ww;
//     int i, j, m;
//     if (ntw%2 == 0)
//         ntw = (ntw+1);
//     m = (ntw-1)/2;

	
// 	if(tri==0)
// 	{ndata=nx*ny*nz;}
// 	else
// 	{ndata=nx*ny*nt;}
	ndata=nx*nz;

// 	printf("n1=%d,nw=%d,nt=%d 22\n",n1,nw,nt);
	
//     if(opt)
// 	    nt = 2*kiss_fft_next_fast_size((ntw+1)/2);
//     else
//         nt=ntw;
//     if (nt%2) nt++;
//     nw = nt/2+1;
//     dw = 1./(nt*dt);
// 	w0 = 0.;

// 	printf("n1=%d,nw=%d,nt=%d 33\n",n1,nw,nt);
// 	printf("dw=%g,w0=%g,dt=%g\n",dw,w0,dt);
    
//     kiss_fftr_cfg cfg;
//     kiss_fft_cpx *pp, ce, *outp;
//     float *p, *inp, *tmp;
//     float wt, shift;

    
//     p = np_floatalloc(nt);
//     pp = (kiss_fft_cpx*) np_complexalloc(nw);
//     cfg = kiss_fftr_alloc(nt,inv?1:0,NULL,NULL);
//     wt = sym? 1./sqrtf((float) nt): 1.0/nt;
// 
//     printf("sym=%d,wt=%g\n",sym,wt);
    
//     inp = np_floatalloc(n1);
//     tmp = np_floatalloc(n1*nw*2);
//     outp = (kiss_fft_cpx*) np_complexalloc(n1*nw);

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

// 	data  = (float*)malloc(ndata * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    

        
/** Part V: main program ********/
// int main(int argc, char* argv[])
// {

    /*survey parameters*/

//     np_file Fi,Fo,Fd,Fd_v,snaps,Fvpad; /* I/O files */
//     np_axis az,ax,ay; /* cube axes */

//     np_init(argc,argv);

//     if (!np_getint("jsnap",&jsnap)) jsnap=0; /* interval for snapshots */
//     if (!np_getbool("cmplx",&cmplx)) cmplx=true; /* use complex fft */
//     if (!np_getint("pad1",&pad1)) pad1=1; /* padding factor on the first axis */
//     if(!np_getbool("abc",&abc)) abc=false; /* absorbing flag */
    
    cmplx=0;
// 	jsnap=0;
	pad1=1;
	abc=1;
	src=0;
	
    if (abc) {
//       if(!np_getint("nbt",&nbt)) np_error("Need nbt!");
//       if(!np_getint("nbb",&nbb)) nbb = nbt;
//       if(!np_getint("nblx",&nblx)) nblx = nbt;
//       if(!np_getint("nbrx",&nbrx)) nbrx = nbt;
//       if(!np_getint("nbly",&nbly)) nbly = nbt;
//       if(!np_getint("nbry",&nbry)) nbry = nbt;
//       if(!np_getfloat("ct",&ct)) np_error("Need ct!");
//       if(!np_getfloat("cb",&cb)) cb = ct;
//       if(!np_getfloat("clx",&clx)) clx = ct;
//       if(!np_getfloat("crx",&crx)) crx = ct;
//       if(!np_getfloat("cly",&cly)) cly = ct;
//       if(!np_getfloat("cry",&cry)) cry = ct;
	nbb=nbt;
	nblx = nbt;
	nbrx = nbt;
// 	nbly = nbt;
// 	nbry = nbt;
	cb=ct;
	clx = ct;
	crx = ct;
// 	cly = ct;
// 	cry = ct;
	
    } else {
      nbt = 0; nbb = 0; nblx = 0; nbrx = 0; 
      ct = 0; cb = 0; clx = 0; crx = 0; 
    }
//     if (!np_getbool("verb",&verb)) verb=false; /* verbosity */
//     if (!np_getbool("ps",&ps)) ps=false; /* use pseudo-spectral */
//     if (ps) np_warning("Using pseudo-spectral...");
//     else np_warning("Using pseudo-analytical...");
//     if (!np_getbool("tri",&tri)) tri=false; /* if choose time reversal imaging */
//     if (tri) np_warning("Time-reversal imaging");
//     else np_warning("Forward modeling");
//     if (!np_getfloat("vref",&vref)) vref=1500; /* reference velocity (default using water) */

    /* setup I/O files */
//     Fi = np_input ("in");
//     Fo = np_output("out");
    
    int   *spx, *spz;
    if (tri) {
//       gplx = -1;
//       gply = -1;
//       gpl_v = -1;
//       if (NULL==np_getstring("dat") && NULL==np_getstring("dat_v"))
// 	np_error("Need Data!");
//       if (NULL!=np_getstring("dat")) {
// 	Fd = np_input("dat");
// 	np_histint(Fd,"n1",&nt);
// 	np_histfloat(Fd,"d1",&dt);
// 	np_histint(Fd,"n2",&gplx);
// 	np_histint(Fd,"n3",&gply);
//       } else Fd = NULL;
//       if (NULL!=np_getstring("dat_v")) {
// 	Fd_v = np_input("dat_v");
// 	np_histint(Fd_v,"n1",&nt);
// 	np_histfloat(Fd_v,"d1",&dt);
// 	np_histint(Fd_v,"n2",&gpl_v);
//       } else Fd_v = NULL;
      src = -1; ns = -1;
      spx = NULL; spz = NULL;
      f0 = NULL; t0 = NULL; A = NULL;
    } else {
//       Fd = NULL;
//       if (!np_getint("nt",&nt)) np_error("Need nt!");
//       if (!np_getfloat("dt",&dt)) np_error("Need dt!");
//       if (!np_getint("gplx",&gplx)) gplx = -1; /* geophone length X*/
//       if (!np_getint("gply",&gply)) gply = -1; /* geophone length Y*/
//       if (!np_getint("gpl_v",&gpl_v)) gpl_v = -1; /* geophone height */
//       if (!np_getint("src",&src)) src=0; /* source type */
//       if (!np_getint("ns",&ns)) ns=1; /* source type */

//     int   *spx, *spy, *spz;
//     float   *spx, *spy, *spz;
	  printf("ns=%d\n",ns);
      spx = np_intalloc(ns);
      spz = np_intalloc(ns);
//       spx = np_floatalloc(ns);
//       spz = np_floatalloc(ns);
      f0  = np_floatalloc(ns);
      t0  = np_floatalloc(ns);
      A   = np_floatalloc(ns);
	float tmp;
    for (i=0; i<ns; i++)
    {
        tmp=*((float*)PyArray_GETPTR1(arrf2,i));
        spx[i]=tmp;
        tmp=*((float*)PyArray_GETPTR1(arrf2,ns*1+i));
        spz[i]=tmp;
        f0[i]=*((float*)PyArray_GETPTR1(arrf2,ns*2+i));
        t0[i]=*((float*)PyArray_GETPTR1(arrf2,ns*3+i));
        A[i]=*((float*)PyArray_GETPTR1(arrf2,ns*4+i));
    }
    
    printf("There are %d sources to be simulated\n",ns);
    for(i=0;i<ns;i++)
    {
    printf("spx[%d]=%d\n",i,spx[i]);
    printf("spz[%d]=%d\n",i,spz[i]);
    printf("f0[%d]=%g\n",i,f0[i]);
    printf("t0[%d]=%g\n",i,t0[i]);
    printf("A[%d]=%g\n",i,A[i]);
    }
    
    }
//       if (!np_getints("spx",spx,ns)) np_error("Need spx!"); /* shot position x */
//       if (!np_getints("spy",spy,ns)) np_error("Need spy!"); /* shot position y */
//       if (!np_getints("spz",spz,ns)) np_error("Need spz!"); /* shot position z */
//       if (!np_getfloats("f0",f0,ns)) np_error("Need f0! (e.g. 30Hz)");   /*  wavelet peak freq */
//       if (!np_getfloats("t0",t0,ns)) np_error("Need t0! (e.g. 0.04s)");  /*  wavelet time lag */
//       if (!np_getfloats("A",A,ns)) np_error("Need A! (e.g. 1)");     /*  wavelet amplitude */
//     }
//     if (!np_getint("gpx",&gpx)) gpx = -1; /* geophone position x */
//     if (!np_getint("gpy",&gpy)) gpy = -1; /* geophone position y */
//     if (!np_getint("gpz",&gpz)) gpz = -1; /* geophone position z */
//     if (!np_getint("gpx_v",&gpx_v)) gpx_v = -1; /* geophone position x */
//     if (!np_getint("gpy_v",&gpy_v)) gpy_v = -1; /* geophone position y */
//     if (!np_getint("gpz_v",&gpz_v)) gpz_v = -1; /* geophone position z */
// 
//     if (np_FLOAT != np_gettype(Fi)) np_error("Need float input");

    /* Read/Write axes */
//     az = np_iaxa(Fi,1); nz = np_n(az); dz = np_d(az);
//     ax = np_iaxa(Fi,2); nx = np_n(ax); dx = np_d(ax);
//     ay = np_iaxa(Fi,3); ny = np_n(ay); dy = np_d(ay);


    /*change on Jun 2022, YC*/
    nz1 = nz;
    nx1 = nx;
    nz = nz+nbt+nbb;
    nx = nx+nblx+nbrx;
    /*change on Jun 2022, YC*/
    
//     if(verb)np_warning("ny=%d,nbly=%d,nbry=%d",ny,nbly,nbry);
//     if(verb)np_warning("nz1=%d,nx1=%d,ny1=%d",nz1,nx1,ny1);
//     if (gpx==-1) gpx = nblx;
//     if (gpy==-1) gpy = nbly;
//     if (gpz==-1) gpz = nbt;
//     if (gplx==-1) gplx = nx1;
//     if (gply==-1) gply = ny1;
	gplx = nx1;
	gpl_v = nz1;
	gpx=nblx;
	gpz=nbt;
	vref=1500;
	ps=1;
//     if (gpx_v==-1) gpx_v = nblx;
//     if (gpy_v==-1) gpy_v = nbly;
//     if (gpz_v==-1) gpz_v = nbt;
//     if (gpl_v==-1) gpl_v = nz1;
    ntsnap=0;
    if (jsnap)
        for (it=0;it<nt;it++)
            if (it%jsnap==0) ntsnap++;
            
//     if (tri) { /*output final wavefield*/
//       np_setn(az,nz1);
//       np_setn(ax,nx1);
//       np_setn(ay,ny1);
//       np_oaxa(Fo,az,1);
//       np_oaxa(Fo,ax,2);
//       np_oaxa(Fo,ay,3);   
//       np_settype(Fo,np_FLOAT);
//     } else { /*output data*/
//       np_setn(ax,gplx);
//       np_setn(ay,gply);
//       np_putint(Fo,"n3",gply);
//       np_warning("ny=%d,nbly=%d,nbry=%d",ny,nblx,nbly);
//       np_warning("gplx=%d,gply=%d",gplx,gply);
//       /*output horizontal data is mandatory*/
//       np_putint(Fo,"n1",nt);
//       np_putfloat(Fo,"d1",dt);
//       np_putfloat(Fo,"o1",0.);
//       np_putstring(Fo,"label1","Time");
//       np_putstring(Fo,"unit1","s");
//       np_oaxa(Fo,ax,2);
//       np_settype(Fo,np_FLOAT);
//       /*output vertical data is optional*/
//       if (NULL!=np_getstring("dat_v")) {
// 	Fd_v = np_output("dat_v");
// 	np_setn(az,gpl_v);
// 	np_putint(Fd_v,"n1",nt);
// 	np_putfloat(Fd_v,"d1",dt);
// 	np_putfloat(Fd_v,"o1",0.);
// 	np_putstring(Fd_v,"label1","Time");
// 	np_putstring(Fd_v,"unit1","s");
// 	np_oaxa(Fd_v,az,2);
// 	np_putint(Fd_v,"n3",1);
// 	np_settype(Fd_v,np_FLOAT);	
//       } else Fd_v = NULL;
//     }

//     if (NULL!=np_getstring("vpad")) {
// 	   Fvpad=np_output("vpad");
//        np_putint(Fvpad,"n1",nz);
//        np_putint(Fvpad,"n2",nx);
//        np_putint(Fvpad,"n3",ny);
//        ifvpad=true;
//       } else 
//       {
//       Fvpad = NULL;
       ifvpad=false;
//       }
      
//     if (jsnap > 0) {
// 	snaps = np_output("snaps");
// 	/* (optional) snapshot file */
// 	np_setn(az,nz1);
// 	np_setn(ax,nx1);
// 	np_setn(ay,ny1);
// 	np_oaxa(snaps,az,1);
// 	np_oaxa(snaps,ax,2);
// 	np_oaxa(snaps,ay,3);
// 	np_putint(snaps,"n4",ntsnap);
// 	np_putfloat(snaps,"d4",dt*jsnap);
// 	np_putfloat(snaps,"o4",0.);
// 	np_putstring(snaps,"label4","Time");
// 	np_putstring(snaps,"unit4","s");
//     } else snaps = NULL;

    par = (psmpar) np_alloc(1,sizeof(*par));
    vel = np_floatalloc(nz1*nx1); 	/*change on Jun 2022, YC*/
    vel2= np_floatalloc(nz*nx); 		/*change on Jun 2022, YC*/

    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("input data done, ndata=%d\n",ndata);
	
//     if (tri && NULL==Fd) {dat = NULL;  }
//     else { dat = np_floatalloc3(nt,gplx,gply);}

	if(tri)
	{
	printf("Doing TRI, reading data\n");
	dat = np_floatalloc2(nt,gplx);
    for (i=0; i<nt*gplx; i++)
    {
        dat[0][i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
	printf("Doing TRI, reading data done\n");
    }
// 	for(i=0;i<ndata;i++)
// // 	dat[0][0][i]=0;
// 	printf("vel[%d]=%g\n",i,vel[i]);

	if(tri==0)
	{
	dat=np_floatalloc2(nt,gplx);

	for(i=0;i<nt*gplx;i++)
	dat[0][i]=0;
	}
	
	
	int ifvdata=0;
	if(ifvdata==1)dat_v = np_floatalloc2(nt,gpl_v);
    else dat_v = NULL;
	
	
    if (tri) img = np_floatalloc(nz1*nx1);
    else img = NULL;

    if (jsnap>0) wvfld = np_floatalloc2(nx1*nz1,ntsnap);
    else wvfld = NULL;
    

//     np_floatread(vel,nz1*ny1*nx1,Fi);
	
	/*2D velocity expansion uses 3D function*/
	vel_expand(vel,vel2,nz1,nx1,1,nbt,nbb,nblx,nbrx,0,0);  /*if we can use existing function (e.g., 3D version), use it*/
// 	for(i=0;i<nz1*nx1*ny1;i++)
// // 	dat[0][0][i]=0;
// 	printf("vel2[%d]=%g\n",i,vel2[i]);

//     if (tri) {
// //       if (NULL!=Fd)   np_floatread(dat[0][0],gplx*gply*nt,Fd);
// //       if (NULL!=Fd_v) np_floatread(dat_v[0],gpl_v*nt,Fd_v);
// 
// 
//     }

// 	float sum=0;
// 	for(i=0;i<ndata;i++)
// 	sum=sum+vel[i];

// 	for(i=0;i<ndata;i++)
// 	if(vel[i]!=vel[0])
// 	printf("vel[%d]=%g\n",i,vel[i]);
	
// 	printf("vel sum = %g\n",sum);

    /*passing the parameters*/
    par->nx    = nx;  
    par->nz    = nz;
    par->dx    = dx;
    par->dz    = dz;
    par->ns	   = ns;
    par->spx   = spx;
    par->spz   = spz;
    par->gpx   = gpx;
    par->gpz   = gpz;
    par->gplx   = gplx;
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
    par->ct    = ct;
    par->cb    = cb;
    par->clx    = clx;
    par->crx    = crx;
    par->src   = src;
    par->nt    = nt;
    par->dt    = dt;
    par->f0    = f0;
    par->t0    = t0;
    par->A     = A;
    par->verb  = verb;
    par->ps    = ps;
    par->vref  = vref;

	printf("par->nx=%d,par->nz=%d\n",par->nx,par->nz);
	printf("par->dx=%g,par->dz=%g\n",par->dx,par->dz);
	printf("par->ct=%g,par->cb=%g,par->clx=%g,par->cly=%g\n",par->ct,par->cb,par->clx);
	
	
	printf("par->verb=%d,par->ps=%d,par->vref=%g\n",par->verb,par->ps,par->vref);
		
    /*do the work*/
    psm2d(wvfld, dat, dat_v, img, vel2, par, tri);
	
	printf("psm2d done\n");
    if (tri) {
//       np_floatwrite(img,nz1*ny1*nx1,Fo);
    } else {
//       np_floatwrite(dat[0][0],gplx*gply*nt,Fo);
//       if (NULL!=Fd_v)
// 	np_floatwrite(dat_v[0],gpl_v*nt,Fd_v);
    }

//     if (jsnap>0)
//       np_floatwrite(wvfld[0],nz1*nx1*ny1*ntsnap,snaps);
      
//     if(ifvpad)
//     	np_floatwrite(vel2,nz*nx*ny,Fvpad);
      
//     exit (0);
// }
	
	/*sub-function goes here*/
//     }
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];

	int nwfd;
	if(jsnap>0)
	{nwfd=nz1*nx1*ntsnap;
	printf("ntsnap=%d\n",ntsnap);
	}
	else
	nwfd=0;
	
	dims[0]=nt*nx1+nwfd;dims[1]=1;
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<nt*nx1;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = dat[0][i];
		
	if(jsnap>0)
	{
	
	for(i=0;i<nwfd;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+nt*nx1))) = wvfld[0][i];
		
	}
// 	printf("w0=%g,dw=%g,nw=%d\n",w0,dw,nw);
// 	(*((float*)PyArray_GETPTR1(vecout,0+ndata*nw*2))) = w0;
// 	(*((float*)PyArray_GETPTR1(vecout,1+ndata*nw*2))) = dw;
// 	(*((float*)PyArray_GETPTR1(vecout,2+ndata*nw*2))) = nw;
	
// 	}else{
	
	dims[0]=n1;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
// 	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
// 	for(i=0;i<dims[0];i++)
// 		(*((float*)PyArray_GETPTR1(vecout,i))) = inp[i];
// 	}
	
	return PyArray_Return(vecout);
	
}

/*documentation for each functions.*/
static char ftfacfun_document[] = "Document stuff for this C module...";

/*defining our functions like below:
  function_name, function, METH_VARARGS flag, function documents*/
static PyMethodDef functions[] = {
  {"aps3dc", aps3dc, METH_VARARGS, ftfacfun_document},
  {"aps2dc", aps2dc, METH_VARARGS, ftfacfun_document},
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
PyMODINIT_FUNC PyInit_apscfun(void){
  
    PyObject *module = PyModule_Create(&ftfacfunModule);
    import_array();
    return module;
}
