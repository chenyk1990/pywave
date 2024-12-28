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

// #include "wave_commons.h"
// #include "wave_lbfgs.h"
// #include "wave_gradient.h"
#include "wave_fwi.h"
#include "wave_fwiutil.h"
#include "wave_fwigradient.h"
/*^*/

// void fwi(np_file Fdat, np_file Finv, np_file Fgrad, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_optim optpar, bool verb, int media)
// /*< fwi >*/
// {
// 	int iter=0, flag;
// 	int nz, nx, nzx, nm=0;
// 	float fcost;
// 	float *x=NULL, *direction, *grad;
// 	np_gradient gradient=NULL;
// 	FILE *fp=NULL;
// 
// 	nz=acpar->nz;
// 	nx=acpar->nx;
// 	nzx=nz*nx;
// 
// 	/* gradient type */
// 	if(fwipar->grad_type==1) {
// 		if(media==1) gradient=gradient_av;
// 		else gradient=gradient_v;
// 		nm=nzx;
// 		x=array->vv;
// 	}
// 
// 	/* initialize */
// 	gradient_init(Fdat, soupar, acpar, array, fwipar, verb);
// 
// 	/* calculate first gradient */
// 	grad=np_floatalloc(nm);
// 	gradient(x, &fcost, grad);
// 
// 	/* output first gradient */
// 	if(mpipar->cpuid==0) np_floatwrite(grad, nm, Fgrad);
// 
// 	if(fwipar->onlygrad) return; // program terminates 
// 
// 	if(mpipar->cpuid==0) fp=fopen("iterate.txt","a");
// 	direction=np_floatalloc(nm);
// 	optpar->sk=np_floatalloc2(nm, optpar->npair);
// 	optpar->yk=np_floatalloc2(nm, optpar->npair);
// 
// 	optpar->igrad=0;
// 	optpar->ipair=0;
// 	optpar->alpha=1.;
// 	optpar->ils=0;
// 	optpar->fk=fcost;
// 	optpar->f0=fcost;
// 	if(mpipar->cpuid==0){
// 		l2norm(nm, grad, &optpar->gk_norm);
// 		print_iteration(fp, iter, optpar);
// 	}
// 
// 	/* optimization loop */
// 	for(iter=0; iter<optpar->niter; iter++){
// 		if(mpipar->cpuid==0) np_warning("--------iter=%d---------", iter);
// 
// 		optpar->ils=0;
// 
// 		//reverse(nm, grad, direction);
// 		if(iter==0){
// 			reverse(nm, grad, direction);
// 		}else{
// 			lbfgs_update(nm, x, grad, optpar->sk, optpar->yk, optpar);
// 			lbfgs_direction(nm, grad, direction, optpar->sk, optpar->yk, optpar);
// 		} 
// 
// 		lbfgs_save(nm, x, grad, optpar->sk, optpar->yk, optpar);
// 		line_search(nm, x, grad, direction, gradient, optpar, &flag, mpipar->cpuid);
// 		
// 		if(mpipar->cpuid==0){
// 			l2norm(nm, grad, &optpar->gk_norm);
// 			print_iteration(fp, iter+1, optpar);
// 		}
// 
// 		if(mpipar->cpuid==0 && flag==2){
// 			fprintf(fp, "Line Search Failed\n");
// 			break;
// 		}
// 
// 		if(mpipar->cpuid==0 && optpar->fk/optpar->f0 < optpar->conv_error){
// 			fprintf(fp, "Convergence Criterion Reached\n");
// 			break;
// 		}
// 	} // end of iter
// 
// 	if(mpipar->cpuid==0 && iter==optpar->niter){
// 		fprintf(fp, "Maximum Iteration Number Reached\n");
// 	}
// 
// 	if(mpipar->cpuid==0){
// 		np_floatwrite(x, nm, Finv);
// 	}
// 	if(mpipar->cpuid==0) fclose(fp);
// }

void lstri(float ***data, float ***mwt, float ****src, np_acqui acpar, np_vec array, np_pas paspar, bool verb)
/*< passive source inversion >*/
{
    float **dd, ***ww;
    int nturn, iturn, is, rdn;


//     dd = np_floatalloc2(acpar->nt, acpar->nx);
//     ww = np_floatalloc3(acpar->nz, acpar->nx, acpar->nt);
    
    if (paspar->inv) mwt = np_floatalloc3(acpar->nz, acpar->nx, acpar->nt);
    else mwt = NULL;

    for(is=0; is<acpar->ns; is++){

//             if (paspar->inv) {
                /* shift pointer and load data */
                dd=data[0]+acpar->nt*acpar->nx*is;
//             } else {
                /* shift pointer and load data */
                ww=src[0]+acpar->nt*acpar->nz*acpar->nx*is;
//             }

            /* do the computation */
            lstri_op(dd, NULL, ww, mwt, acpar, array, paspar, verb);

//             if (paspar->inv) {
//                 /* write source */
//                 fseeko(temp, is*acpar->nz*acpar->nx*acpar->nt*sizeof(float), SEEK_SET);
//                 fwrite(ww[0][0], sizeof(float), acpar->nz*acpar->nx*acpar->nt, temp);
//                 if (NULL!=Fmwt && is==0) np_floatwrite(mwt[0][0], acpar->nz*acpar->nx*acpar->nt, Fmwt);
//                 
//                 
//             } else {
//                 /* write data */
//                 fseeko(temp, is*acpar->nt*acpar->nx*sizeof(float), SEEK_SET);
//                 fwrite(dd[0], sizeof(float), acpar->nt*acpar->nx, temp);
//             }
    }

    
//         temp=fopen(filename, "rb");
//         if (paspar->inv) {
//             for(is=0; is<acpar->ns; is++){
//                 fseeko(temp, is*acpar->nz*acpar->nx*acpar->nt*sizeof(float), SEEK_SET);
//                 if (!fread(ww[0][0], sizeof(float), acpar->nz*acpar->nx*acpar->nt, temp))
//                     abort();
//                 np_floatwrite(ww[0][0], acpar->nz * acpar->nx * acpar->nt, Fsrc);
//             }
//         } else {
//             for(is=0; is<acpar->ns; is++){
//                 fseeko(temp, is*acpar->nt*acpar->nx*sizeof(float), SEEK_SET);
//                 if (!fread(dd[0], sizeof(float), acpar->nt*acpar->nx, temp))
//                     abort();
//                 np_floatwrite(dd[0], acpar->nt * acpar->nx, Fdat);
//             }
//         }
//         fclose(temp);
//         remove(filename);
// 
//     MPI_Barrier(comm);
// 
//     /* close */
//     free(*dd); free(dd); 
//     free(**ww); free(*ww); free(ww);
//     if (paspar->inv) { free(**mwt); free(*mwt); free(mwt); }

}

// data, vinv, grad, src, mwt, soupar, acpar, array, fwipar, optpar, paspar, verb)

// void pfwi(float *data, float *inv, float *grad, float *mwt, float *src, np_sou soupar, np_acqui acpar, np_vec array, np_fwi fwipar, np_optim optpar, np_pas paspar, bool verb)
// /*< passive fwi >*/
// {
// 	int iter=0, flag;
// 	int nz, nx, nzx, nm=0;
// 	float fcost;
// 	float *x=NULL, *direction, *grad;
// 	np_gradient gradient=NULL;
// 	FILE *fp=NULL;
// 
// 	nz=acpar->nz;
// 	nx=acpar->nx;
// 	nzx=nz*nx;
// 
// 	/* gradient type */
//         gradient=gradient_pas_av;
//         nm=nzx;
//         x=array->vv;
// 
// 
// // nz=acpar.nz;
// // nx=acpar.nx;
// // nzx=nz*nx;
// // 
// // %gradient type
// // % gradient=gradient_pas_av;
// // nm=nzx;
// // x=array.vv;
// // %initialize
// // % []=gradient_pas_init(Fdat, Fsrc, Fmwt, mpipar, soupar, acpar, array, fwipar, paspar, verb);
// // 
// // %calculate first gradient
// // % grad=zeros(nm,1);
// // [grad,fcost,src,mwt]=gradient_pas_av(data,x,soupar,acpar,array,fwipar,paspar,verb);
// // % figure(20);
// // % subplot(1,2,1);imagesc(array.vv);
// // % subplot(1,2,2);imagesc(grad);
// // 
// 
// 	/* initialize */
// 	gradient_pas_init(Fdat, Fsrc, Fmwt, mpipar, soupar, acpar, array, fwipar, paspar, verb);
// 
// 	/* calculate first gradient */
// 	grad=np_floatalloc(nm);
// 	gradient(x, &fcost, grad);
// 
// 	/* output first gradient */
// 	if(mpipar->cpuid==0) np_floatwrite(grad, nm, Fgrad);
// 
// 	if(fwipar->onlygrad) return; /* program terminates */
// 
// 	if(mpipar->cpuid==0) fp=fopen("iterate.txt","a");
// 	direction=np_floatalloc(nm);
// 	optpar->sk=np_floatalloc2(nm, optpar->npair);
// 	optpar->yk=np_floatalloc2(nm, optpar->npair);
// 
// 	optpar->igrad=0;
// 	optpar->ipair=0;
// 	optpar->alpha=1.;
// 	optpar->ils=0;
// 	optpar->fk=fcost;
// 	optpar->f0=fcost;
//         if(mpipar->cpuid==0) {
//             l2norm(nm, grad, &optpar->gk_norm);
//             print_iteration(fp, iter, optpar);
//         }
// 
// 	/* optimization loop */
//         for(iter=0; iter<optpar->niter; iter++){
// //             if(mpipar->cpuid==0) np_warning("--------iter=%d---------", iter);
// 
//             if (iter%optpar->repeat==0) optpar->alpha=1.;
// 
//             optpar->ils=0;
// 
//             if(iter==0){
//                 reverse(nm, grad, direction);
//             }else{
//                 lbfgs_update(nm, x, grad, optpar->sk, optpar->yk, optpar);
//                 lbfgs_direction(nm, grad, direction, optpar->sk, optpar->yk, optpar);
//             } 
// 
//             lbfgs_save(nm, x, grad, optpar->sk, optpar->yk, optpar);
//             line_search(nm, x, grad, direction, gradient, optpar, &flag, mpipar->cpuid);
//             if(mpipar->cpuid==0) {
//                 l2norm(nm, grad, &optpar->gk_norm);
//                 print_iteration(fp, iter+1, optpar);
//             }
// 
//             if(mpipar->cpuid==0 && flag==2){
//                 fprintf(fp, "Line Search Failed\n");
//                 break;
//             }
// 
//             if(mpipar->cpuid==0 && optpar->fk/optpar->f0 < optpar->conv_error){
//                 fprintf(fp, "Convergence Criterion Reached\n");
//                 break;
//             }
//         } // end of iter
// 
//         if(mpipar->cpuid==0 && iter==optpar->niter){
//             fprintf(fp, "Maximum Iteration Number Reached\n");
//         }
// 
// 	if(mpipar->cpuid==0){
//             np_floatwrite(x, nm, Finv);
//         }
// 
// 	if(mpipar->cpuid==0) fclose(fp);
// 
// }
