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

static PyObject *stft1dc(PyObject *self, PyObject *args){
    
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
	PyArg_ParseTuple(args, "Oiiiiiiiff", &f1,&n1,&verb,&window,&inv,&sym,&opt,&ntw,&dt,&ot);

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
	printf("ndata=%d,ntw=%d\n",ndata,ntw);
        
        for (i=0; i < n1; i++)  {
        for (j=0; j < ntw; j++) {
            if (i+j-m < 0 || i+j-m >= n1) {
            p[j] = 0.;
            } else {
            p[j] = inp[i+j-m];
            }
        }

        if (window) {
            for (i1=0; i1 < ntw; i1++) {
                p[i1] *= (0.5 - 0.5*cosf(2*SF_PI*i1 / (ntw - 1)));
            }
        }
        
        if (sym) {
            for (i1=0; i1 < ntw; i1++) {
            p[i1] *= wt;
            }
        }
        
        for (i1=ntw; i1 < nt; i1++) {
            p[i1]=0.0;
        }
            
        kiss_fftr (cfg,p,pp);
        
        if (0. != ot) {
            for (i1=0; i1 < nw; i1++) {
            shift = -2.0*SF_PI*i1*dw*ot;
            ce.r = cosf(shift);
            ce.i = sinf(shift);
            pp[i1]=np_cmul(pp[i1],ce);
            }
        }
        for (i1=0; i1 < nw; i1++) {
            outp[i1*n1+i] = np_cmplx(pp[i1].r,pp[i1].i);
        }
        }
        
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
        for (i=0; i < n1; i++)  {
        inp[i] = 0.;
        }
        for (i=0; i < n1; i++)  {
        for (i1=0; i1 < nw; i1++) {
            pp[i1].r = np_crealf(outp[i1*n1+i]);
            pp[i1].i = np_cimagf(outp[i1*n1+i]);
        }
        if (0. != ot) {
            for (i1=0; i1 < nw; i1++) {
            shift = +2.0*SF_PI*i1*dw*w0;
            ce.r = cosf(shift);
            ce.i = sinf(shift);
            pp[i1]=np_cmul(pp[i1],ce);
            }
        }
        
        kiss_fftri(cfg,pp,p);
        
        for (i1=0; i1 < ntw; i1++) {
            p[i1] *= wt;
            if (i+i1-m >= 0 && i+i1-m < n1) {
            inp[i+i1-m] += p[i1];
            }
        }
        }

        for (i=0; i < m; i++) {
        inp[i] = inp[i]/((i+m+1)*1.);
        }
        for (i=m; i < n1-m; i++)  {
        inp[i] = inp[i]/(ntw*1.);
        }
        for (i=n1-m; i < n1; i++) {
        inp[i] = inp[i]/((n1-i+m)*1.);
        }

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
  {"stft1dc", stft1dc, METH_VARARGS, ftfacfun_document},
  {"st1dc", st1dc, METH_VARARGS, ftfacfun_document},
  {NULL, NULL, 0, NULL}
};

/*initializing our module informations and settings in this structure
for more informations, check head part of this file. there are some important links out there.*/
static struct PyModuleDef ftfacfunModule = {
  PyModuleDef_HEAD_INIT, /*head informations for Python C API. It is needed to be first member in this struct !!*/
  "ftfacfun",  /*module name (FFT-based time-frequency analysis)*/
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
