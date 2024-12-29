
#ifndef fwilbfgs_h
#define fwilbfgs_h

#include "wave_fwiutil.h"
/*^*/

void lbfgs_save(int n, float *x, float *grad, float **sk, float **yk, np_optim opt);
/*< save current model and gradient >*/

void lbfgs_update(int n, float *x, float *grad, float **sk, float **yk, np_optim opt);
/*< update current sk and yk >*/

void lbfgs_direction(int n, float *grad, float *r, float **sk, float **yk, np_optim opt);
/*< calculate search direction >*/

void clip(float *x, int n, float min, float max);
/*< clip data >*/

void line_search(int n, float *x, float *grad, float *direction, np_gradient gradient, np_optim opt, int *flag);
/*< line search (Wolfe condition) >*/

#endif