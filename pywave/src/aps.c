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
#include "wave_psp.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

float gauss(int n, int m)
/* Gaussian function */
{
    return exp(-2.*SF_PI*SF_PI*m*m/(n*n));
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
    

//     sf_floatread(vel,nz1*ny1*nx1,Fi);
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
