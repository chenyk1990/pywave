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

#include "wave_fwi.h"
#include "wave_fwiutil.h"


static PyObject *lstric(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf2=NULL;
    PyObject *f3=NULL;
    PyObject *arrf3=NULL;
    PyObject *f4=NULL;
    PyObject *arrf4=NULL;
    PyObject *f5=NULL;
    PyObject *arrf5=NULL;
    
	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim;
//     float *data;
    
    int niter,verb=1,rect0,n1,ntw,opt=0,sym,window;
    int ifb,inv;
    
    int   nx, ny, nz;
    float dx, dy, dz, dt;
    float oz,ox,oy,ot; 
    int   ns, interval, atype; 	/*new par*/
    float os, ds, sz, f0, coef;		/*new par*/
    int   gpz, gpx, gpy, gplx, gply; /*geophone positions (z,x,y) and geophone length (z,x,y)*/
    int   gpz_v, gpx_v, gpy_v, gpl_v;
    int   jsnap;
    /*fft related*/
    bool  cmplx;
    int   pad1;
    /*absorbing boundary*/
    bool abc,ifvpad;
    int nb, nbt, nbb, nblx, nbrx, nbly, nbry; /*boundaries for top/bottom, left/right x, left/right y*/
    float ct,cb,clx,crx,cly,cry; 		  /*decaying parameter*/
    /*source parameters*/
//     int src; /*source type*/
    int nt,ntsnap;
//     float t0,*A;
    /*misc*/
    int ps, tri; /*tri: time-reversal imaging*/
    float vref;
    int i;

    psmpar par;
    int nx1, ny1, nz1; /*domain of interest*/
    int it;
    float *vel2,**dat,**dat_v,**wvfld,*img; /*velocity profile*/
    float *vel, *q, *wav; 
    

    int ifsnaps;
    
    /*data and parameters interface*/
	PyArg_ParseTuple(args, "OOOOO", &f1,&f2,&f3,&f4,&f5);
// 	PyArg_ParseTuple(args, "O", &f5);
	printf("Check 1\n");
// 	printf("tri=%d,nt=%d,nx=%d,nz=%d,ns=%d\n",tri,nt,nx,nz,ns);
// 	printf("verb=%d,jsnap=%d,ifsnaps=%d,abc=%d,nbt=%d\n",verb,jsnap,ifsnaps,abc,nbt);
// 	printf("ct=%g,dt=%g,ox=%g,dx=%g,oz=%g,dz=%g\n",ct,dt,ox,dx,oz,dz);
	
// 	ndata=nx*nz;

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);
    arrf4 = PyArray_FROM_OTF(f4, NPY_FLOAT, NPY_IN_ARRAY);
    arrf5 = PyArray_FROM_OTF(f5, NPY_FLOAT, NPY_IN_ARRAY);
    
//     nd2=PyArray_NDIM(arrf1);
//     
//     npy_intp *sp=PyArray_SHAPE(arrf1);
	
//     if (*sp != ndata)
//     {
//     	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
//     	return NULL;
//     }
//         
//     cmplx=0;
// 	pad1=1;
// 	abc=1;
// 	src=0;
// 	
//     if (abc) {
// 	nbb=nbt;
// 	nblx = nbt;
// 	nbrx = nbt;
// 	cb=ct;
// 	clx = ct;
// 	crx = ct;
//     } else {
//       nbt = 0; nbb = 0; nblx = 0; nbrx = 0; 
//       ct = 0; cb = 0; clx = 0; crx = 0; 
//     }
//     
//     int   *spx, *spz;
//     if (tri) {
//       src = -1; ns = -1;
//       spx = NULL; spz = NULL;
//       f0 = NULL; t0 = NULL; A = NULL;
//     } else {
//       spx = np_intalloc(ns);
//       spz = np_intalloc(ns);
//       f0  = np_floatalloc(ns);
//       t0  = np_floatalloc(ns);
//       A   = np_floatalloc(ns);
// 	float tmp;
//     for (i=0; i<ns; i++)
//     {
//         tmp=*((float*)PyArray_GETPTR1(arrf2,i));
//         spx[i]=tmp;
//         tmp=*((float*)PyArray_GETPTR1(arrf2,ns*1+i));
//         spz[i]=tmp;
//         f0[i]=*((float*)PyArray_GETPTR1(arrf2,ns*2+i));
//         t0[i]=*((float*)PyArray_GETPTR1(arrf2,ns*3+i));
//         A[i]=*((float*)PyArray_GETPTR1(arrf2,ns*4+i));
//     }
//     
//     printf("There are %d sources to be simulated\n",ns);
//     for(i=0;i<ns;i++)
//     {
//     printf("spx[%d]=%d\n",i,spx[i]);
//     printf("spz[%d]=%d\n",i,spz[i]);
//     printf("f0[%d]=%g\n",i,f0[i]);
//     printf("t0[%d]=%g\n",i,t0[i]);
//     printf("A[%d]=%g\n",i,A[i]);
//     }
//     
//     }

    /*change on Jun 2022, YC*/
//     nz1 = nz;
//     nx1 = nx;
//     nz = nz+nbt+nbb;
//     nx = nx+nblx+nbrx;
//     /*change on Jun 2022, YC*/
//     
// 	gplx = nx1;
// 	gpl_v = nz1;
// 	gpx=nblx;
// 	gpz=nbt;
// 	vref=1500;
// 	ps=1;
//     ntsnap=0;
//     if (jsnap)
//         for (it=0;it<nt;it++)
//             if (it%jsnap==0) ntsnap++;
//             
//     ifvpad=true;
// 
//     par = (psmpar) np_alloc(1,sizeof(*par));
//     vel = np_floatalloc(nz1*nx1); 	/*change on Jun 2022, YC*/
//     vel2= np_floatalloc(nz*nx); 		/*change on Jun 2022, YC*/
// 
//     /*reading data*/
//     for (i=0; i<ndata; i++)
//     {
//         vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
//     }
// 	printf("input data done, ndata=%d\n",ndata);
// 
// 	if(tri)
// 	{
// 			pararray=np.array([
// 			par['nz'],
// 			par['nx'],
// 			par['dz'],
// 			par['dx'],
// 			par['z0'],
// 			par['x0'],
// 			par['nt'],
// 			par['dt'],
// 			par['t0'],
// 			par['inv'],
// 			par['ns'],
// 			par['ds'],
// 			par['os'],
// 			par['nb'],				#boundary width
// 			par['coef'],			#absorbing boundary coefficient
// 			par['f0'],				#reference frequency
// 			par['acqui_type'],		#1, fixed acquisition; 
// 			par['interval',]		#wavefield storing interval
// 			],dtype='float')
	printf("Check 2\n");
		
	float *pararray;
	pararray= np_floatalloc(30);
	float ***data, ****src, ***mwt;
	printf("Check 3\n");
    for (i=0; i<20; i++)
    {
        pararray[i]=*((float*)PyArray_GETPTR1(arrf5,i));
    }
			
	printf("Check 4\n");
    nz=pararray[0];
    nx=pararray[1];
    dz=pararray[2];
    dx=pararray[3];
    oz=pararray[4];
    ox=pararray[5];
    nt=pararray[6];
    dt=pararray[7];
    ot=pararray[8];
    inv=pararray[9];
    ns=pararray[10];
    ds=pararray[11];
    os=pararray[12];
    sz=pararray[13];
    nb=pararray[14];			/*boundary width*/
    coef=pararray[15];			/*absorbing boundary coefficient*/
    f0=pararray[16];			/*reference frequency*/
	atype=pararray[17];			/*1, fixed acquisition; */
	interval=pararray[18];		/*wavefield storing interval*/

			
// 	lstric(vel, q, wav, datasrc, pararray);

	printf("nt=%d,nx=%d,nz=%d,nt=%d,dt=%g,t0=%g,inv=%d,ns=%d\n",nt,nx,nz,nt,dt,ot,inv,ns);
// 	printf("nt=%d,nx=%d,nz=%d,ns=%d\n",nt,nx,nz,ns);
	
	printf("Reading data\n");
// 	dat = np_floatalloc2(nt,gplx);

	vel=np_floatalloc(nz*nx);
	q=np_floatalloc(nz*nx);
	wav=np_floatalloc(nt);
	
    for (i=0; i<nz*nx; i++)
    {
        vel[i]=*((float*)PyArray_GETPTR1(arrf1,i));
        q[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
    for (i=0; i<nt; i++)
    {
        wav[i]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
    
    if(inv)
    {
    	src=np_floatalloc4(nz,nx,nt,ns);
    	data=np_floatalloc3(nt,nx,ns);
    	for (i=0;i<nx*nt*ns;i++)
    		data[0][0][i]=*((float*)PyArray_GETPTR1(arrf4,i));
    	
    }else{
    	data=np_floatalloc3(nt,nx,ns);
    	src=np_floatalloc4(nz,nx,nt,ns);
    	for (i=0;i<nz*nx*nt*ns;i++)
    		src[0][0][0][i]=*((float*)PyArray_GETPTR1(arrf4,i));
    }

	np_sou soupar;
	np_acqui acpar;
	np_vec array;
	np_pas paspar=NULL;


	soupar=(np_sou)np_alloc(1, sizeof(*soupar));
	acpar=(np_acqui)np_alloc(1, sizeof(*acpar));
	array=(np_vec)np_alloc(1, sizeof(*array));

	/* parameters I/O */
// 	if(!np_getint("media", &media)) media=1;
	/* if 1, acoustic media; if 2, visco-acoustic media */
// 	if(!np_getint("function", &function)) function=2;
	/* if 1, forward modeling; if 2, FWI; if 3, RTM */

	acpar->nz=nz;
	acpar->nx=nx;
	acpar->dz=dz;
	acpar->dx=dx;
	acpar->z0=oz;
	acpar->x0=ox;
	acpar->nt=nt;
	acpar->dt=dt;
	acpar->t0=ot;
	acpar->nb=nb;	/* boundary width */
	acpar->coef=coef;	/* absorbing boundary coefficient */
	acpar->acqui_type=atype;	/* if 1, fixed acquisition; if 2, marine acquisition; if 3, symmetric acquisition */
	acpar->ns=ns;	/* shot number */
	acpar->ds=ds;	/* shot interval */
	acpar->s0=os;	/* shot origin */
	acpar->sz=5;	/* source depth */
	acpar->nr=acpar->nx;	/* number of receiver */
	acpar->dr=acpar->dx;	/* receiver interval */
	acpar->r0=acpar->x0;	/* receiver origin */
	acpar->rz=1;	/* receiver depth */
	acpar->f0=f0; /* reference frequency */
	acpar->interval=interval; /* wavefield storing interval */
	soupar->fhi=0.5/acpar->dt; 
	soupar->flo=0.; 
	soupar->rectx=2; 
	soupar->rectz=2; 

	/*initialize paspar using acpar*/
	paspar = passive_init(acpar);

	/* get prepared */
	preparation(vel, q, wav, acpar, soupar, array);

	float sum=0;
	for(int ii=0;ii<acpar->nz*acpar->nx;ii++)
	sum=sum+array->vv[ii];
	printf("before sum0=%g\n",sum);
	
	

	sum=0;
	for(int ii=0;ii<acpar->nt*acpar->nz*acpar->nx;ii++)
	sum=sum+src[0][0][0][ii];
	printf("before sum=%g\n",sum);
	
    lstri(data, mwt, src, acpar, array, paspar, verb);
    
	sum=0;
	for(int ii=0;ii<acpar->nt*acpar->nz*acpar->nx;ii++)
	sum=sum+src[0][0][0][ii];
	printf("before sum=%g\n",sum);
    
	printf("Doing TRI, reading data done\n");
//     }

// 	if(tri==0)
// 	{
// 	dat=np_floatalloc2(nt,gplx);
// 
// 	for(i=0;i<nt*gplx;i++)
// 	dat[0][i]=0;
// 	}
// 	
// 	
// 	int ifvdata=0;
// 	if(ifvdata==1)dat_v = np_floatalloc2(nt,gpl_v);
//     else dat_v = NULL;
// 	
// 	
//     if (tri) img = np_floatalloc(nz1*nx1);
//     else img = NULL;
// 
//     if (jsnap>0) wvfld = np_floatalloc2(nx1*nz1,ntsnap);
//     else wvfld = NULL;
// 	
// 	/*2D velocity expansion uses 3D function*/
// 	vel_expand(vel,vel2,nz1,nx1,1,nbt,nbb,nblx,nbrx,0,0);  /*if we can use existing function (e.g., 3D version), use it*/
// 
//     /*passing the parameters*/
//     par->nx    = nx;  
//     par->nz    = nz;
//     par->dx    = dx;
//     par->dz    = dz;
//     par->ns	   = ns;
//     par->spx   = spx;
//     par->spz   = spz;
//     par->gpx   = gpx;
//     par->gpz   = gpz;
//     par->gplx   = gplx;
//     par->gpz_v = gpz_v;
//     par->gpx_v = gpx_v;
//     par->gpl_v = gpl_v;
//     par->jsnap  = jsnap;
//     par->cmplx = cmplx;
//     par->pad1  = pad1;
//     par->abc   = abc;
//     par->nbt   = nbt;
//     par->nbb   = nbb;
//     par->nblx   = nblx;
//     par->nbrx   = nbrx;
//     par->ct    = ct;
//     par->cb    = cb;
//     par->clx    = clx;
//     par->crx    = crx;
//     par->src   = src;
//     par->nt    = nt;
//     par->dt    = dt;
//     par->f0    = f0;
//     par->t0    = t0;
//     par->A     = A;
//     par->verb  = verb;
//     par->ps    = ps;
//     par->vref  = vref;
// 
// 	printf("par->nx=%d,par->nz=%d\n",par->nx,par->nz);
// 	printf("par->dx=%g,par->dz=%g\n",par->dx,par->dz);
// 	printf("par->ct=%g,par->cb=%g,par->clx=%g,par->cly=%g\n",par->ct,par->cb,par->clx);
// 	printf("par->verb=%d,par->ps=%d,par->vref=%g\n",par->verb,par->ps,par->vref);
// 		
//     /*do the work*/
//     psm2d(wvfld, dat, dat_v, img, vel2, par, tri);
// 	
// 	printf("psm2d done\n");
	
	
// dd=zeros(acpar.nt,acpar.nx);
// ww=zeros(acpar.nz,acpar.nx,acpar.nt);
// 
// if paspar.inv
//     mwt=zeros(acpar.nz,acpar.nx,acpar.nt);
// else
//     mwt=[];
// end
// 
// for is=0:acpar.ns-1
//     if paspar.inv
//         dd=data(:,:,is+1);
//     else
//         ww=src(:,:,:,is+1);
//     end
//     
//     [dd, dwt, ww, mwt]=lstri_op(dd, [], ww, mwt, acpar, array, paspar, verb);
//     
//     if paspar.inv
//         fprintf('ns=%d\n',acpar.ns);
//         src(:,:,:,is+1)=ww;
// %         fprintf('size(src)\n');
// %         size(src)
//     else
//         data(:,:,is+1)=dd;
//     end
//     
// end

// 	lstri_op(dd, dwt, ww, mwt, acpar, array, paspar, verb);

	
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];

// 	int nwfd;
// 	if(jsnap>0)
// 	{nwfd=nz1*nx1*ntsnap;
// 	printf("ntsnap=%d\n",ntsnap);
// 	}
// 	else
// 	nwfd=0;
	
	dims[0]=nt*nx*ns;dims[1]=1;
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<nt*nx*ns;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = data[0][0][i];
		
// 	if(jsnap>0)
// 	{
// 	
// 	for(i=0;i<nwfd;i++)
// 		(*((float*)PyArray_GETPTR1(vecout,i+nt*nx1))) = wvfld[0][i];
// 		
// 	}
	
	return PyArray_Return(vecout);
	
}

/*documentation for each functions.*/
static char ftfacfun_document[] = "Document stuff for this C module...";

/*defining our functions like below:
  function_name, function, METH_VARARGS flag, function documents*/
static PyMethodDef functions[] = {
  {"lstric", lstric, METH_VARARGS, ftfacfun_document},
  {NULL, NULL, 0, NULL}
};

/*initializing our module informations and settings in this structure
for more informations, check head part of this file. there are some important links out there.*/
static struct PyModuleDef ftfacfunModule = {
  PyModuleDef_HEAD_INIT, /*head informations for Python C API. It is needed to be first member in this struct !!*/
  "pfwicfun",  /*Pseudo-spectral method for acoustic wave equation*/
  NULL, /*means that the module does not support sub-interpreters, because it has global state.*/
  -1,
  functions  /*our functions list*/
};

/*runs while initializing and calls module creation function.*/
PyMODINIT_FUNC PyInit_pfwicfun(void){
  
    PyObject *module = PyModule_Create(&ftfacfunModule);
    import_array();
    return module;
}
