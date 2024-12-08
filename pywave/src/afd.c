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
#include "wave_fdm.h"
#include "wave_abc.h"

#ifndef KISS_FFT_H
#include "wave_kissfft.h"
#endif

static PyObject *afd3dc(PyObject *self, PyObject *args){
    
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

	printf("tri=%d,nt=%d,nx=%d,ny=%d,nz=%d,ns=%d\n",tri,nt,nx,ny,nz,ns);
	printf("verb=%d,jsnap=%d,ifsnaps=%d,abc=%d,nbt=%d\n",verb,jsnap,ifsnaps,abc,nbt);
	printf("ct=%g,dt=%g,ox=%g,dx=%g,oy=%g,dy=%g,oz=%g,dz=%g\n",ct,dt,ox,dx,oy,dy,oz,dz);

	ndata=nx*ny*nz;

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }

    
    cmplx=0;
	pad1=1;
	abc=1;
	src=0;
	
    if (abc) {
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
    
    int   *spx, *spy, *spz;
    if (tri) {
      src = -1; ns = -1;
      spx = NULL; spy = NULL; spz = NULL;
      f0 = NULL; t0 = NULL; A = NULL;
    } else {
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
	
	int ns2;ns2=ns; /*This is extremely weird, if I use ns2, not work*/
    for (i=0; i<ns; i++)
    {
        tmp=*((float*)PyArray_GETPTR1(arrf2,i));
        spx[i]=tmp;
        tmp=*((float*)PyArray_GETPTR1(arrf2,ns2*1+i));
        spy[i]=tmp;
        tmp=*((float*)PyArray_GETPTR1(arrf2,ns2*2+i));
        spz[i]=tmp;
        f0[i]=*((float*)PyArray_GETPTR1(arrf2,ns2*3+i));
        t0[i]=*((float*)PyArray_GETPTR1(arrf2,ns2*4+i));
        A[i]=*((float*)PyArray_GETPTR1(arrf2,ns2*5+i));
    }
    
    printf("There are %d sources to be simulated\n",ns2);
    
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

    /*change on Jun 2022, YC*/
    nz1 = nz;
    nx1 = nx;
    ny1 = ny;
    nz = nz+nbt+nbb;
    nx = nx+nblx+nbrx;
    ny = ny+nbly+nbry;
    /*change on Jun 2022, YC*/

	gplx = nx1;
	gply = ny1;
	gpl_v = nz1;
	gpx=nblx;
	gpy=nbly;
	gpz=nbt;
	vref=1500;
	ps=1;

    ntsnap=0;
    if (jsnap)
        for (it=0;it<nt;it++)
            if (it%jsnap==0) ntsnap++;
            
    ifvpad=true;

    par = (psmpar) np_alloc(1,sizeof(*par));
    vel = np_floatalloc(nz1*ny1*nx1); 	/*change on Jun 2022, YC*/
    vel2= np_floatalloc(nz*ny*nx); 		/*change on Jun 2022, YC*/

    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("input data done, ndata=%d\n",ndata);

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
    
	vel_expand(vel,vel2,nz1,nx1,ny1,nbt,nbb,nblx,nbrx,nbly,nbry);  /*change on Jun 2022, YC*/

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
    fdm(wvfld, dat, dat_v, img, vel2, par, tri);
    
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
	
	return PyArray_Return(vecout);
	
}

static PyObject *afd2dc(PyObject *self, PyObject *args){
    
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

	printf("tri=%d,nt=%d,nx=%d,nz=%d,ns=%d\n",tri,nt,nx,nz,ns);
	printf("verb=%d,jsnap=%d,ifsnaps=%d,abc=%d,nbt=%d\n",verb,jsnap,ifsnaps,abc,nbt);
	printf("ct=%g,dt=%g,ox=%g,dx=%g,oz=%g,dz=%g\n",ct,dt,ox,dx,oz,dz);
	
	ndata=nx*nz;

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
        
    cmplx=0;
	pad1=1;
	abc=1;
	src=0;
	
    if (abc) {
	nbb=nbt;
	nblx = nbt;
	nbrx = nbt;
	cb=ct;
	clx = ct;
	crx = ct;
    } else {
      nbt = 0; nbb = 0; nblx = 0; nbrx = 0; 
      ct = 0; cb = 0; clx = 0; crx = 0; 
    }
    
    int   *spx, *spz;
    if (tri) {
      src = -1; ns = -1;
      spx = NULL; spz = NULL;
      f0 = NULL; t0 = NULL; A = NULL;
    } else {
      spx = np_intalloc(ns);
      spz = np_intalloc(ns);
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

    /*change on Jun 2022, YC*/
    nz1 = nz;
    nx1 = nx;
    nz = nz+nbt+nbb;
    nx = nx+nblx+nbrx;
    /*change on Jun 2022, YC*/
    
	gplx = nx1;
	gpl_v = nz1;
	gpx=nblx;
	gpz=nbt;
	vref=1500;
	ps=1;
    ntsnap=0;
    if (jsnap)
        for (it=0;it<nt;it++)
            if (it%jsnap==0) ntsnap++;
            
    ifvpad=true;

    par = (psmpar) np_alloc(1,sizeof(*par));
    vel = np_floatalloc(nz1*nx1); 	/*change on Jun 2022, YC*/
    vel2= np_floatalloc(nz*nx); 		/*change on Jun 2022, YC*/

    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("input data done, ndata=%d\n",ndata);

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
	
	/*2D velocity expansion uses 3D function*/
	vel_expand(vel,vel2,nz1,nx1,1,nbt,nbb,nblx,nbrx,0,0);  /*if we can use existing function (e.g., 3D version), use it*/

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
    fdm2d(wvfld, dat, dat_v, img, vel2, par, tri);
	
	printf("fdm2d done\n");
	
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
	
	return PyArray_Return(vecout);
	
}

/*documentation for each functions.*/
static char ftfacfun_document[] = "Document stuff for this C module...";

/*defining our functions like below:
  function_name, function, METH_VARARGS flag, function documents*/
static PyMethodDef functions[] = {
  {"afd3dc", afd3dc, METH_VARARGS, ftfacfun_document},
  {"afd2dc", afd2dc, METH_VARARGS, ftfacfun_document},
  {NULL, NULL, 0, NULL}
};

/*initializing our module informations and settings in this structure
for more informations, check head part of this file. there are some important links out there.*/
static struct PyModuleDef ftfacfunModule = {
  PyModuleDef_HEAD_INIT, /*head informations for Python C API. It is needed to be first member in this struct !!*/
  "afdcfun",  /*Pseudo-spectral method for acoustic wave equation*/
  NULL, /*means that the module does not support sub-interpreters, because it has global state.*/
  -1,
  functions  /*our functions list*/
};

/*runs while initializing and calls module creation function.*/
PyMODINIT_FUNC PyInit_afdcfun(void){
  
    PyObject *module = PyModule_Create(&ftfacfunModule);
    import_array();
    return module;
}
