/* Complex number operations */
#include <math.h>

#include "npre_komplex.h"

#ifndef KISS_FFT_H
#include "npre_kissfft.h"
#endif

#include "npre_dtype.h"

#define SF_PI (3.14159265358979323846264338328)

// #include "error.h"
// #include "_defs.h"
// 
// #include "kiss_fft.h"
// #include "c99.h"
// /*^*/

// #define crealf  np_crealf
// #define creal   np_creal
// #define cimagf  np_cimagf
// #define cimag   np_cimag
// #define conjf   np_conjf
// #define cabsf   np_cabsf
// #define cabs    np_cabsd
// #define cargf   np_cargf
// #define carg    np_carg
// #define ccosf   np_ccosf
// #define csinf   np_csinf
// #define ctanf   np_ctanf
// #define cacosf  np_cacosf
// #define casinf  np_casinf
// #define catanf  np_catanf
// #define ccoshf  np_ccoshf
// #define csinhf  np_csinhf
// #define ctanhf  np_ctanhf
// #define cacoshf np_cacoshf
// #define casinhf np_casinhf
// #define catanhf np_catanhf
// #define cexpf   np_cexpf
// #define clogf   np_clogf
// #define csqrtf  np_csqrtf
// #define cpowf   np_cpowf
/*^*/

kiss_fft_cpx np_cmplx(float re, float im)
/*< complex number >*/
{
    kiss_fft_cpx c;
    c.r = re;
    c.i = im;
    return c;
}

double np_creal(np_double_complex c)
/*< real part >*/
{
    return c.r;
}

double np_cimag(np_double_complex c)
/*< imaginary part >*/
{
    return c.i;
}

np_double_complex np_dcneg(np_double_complex a)
/*< unary minus >*/
{
    a.r = -a.r;
    a.i = -a.i;
    return a;
}

np_double_complex np_dcadd(np_double_complex a, np_double_complex b)
/*< complex addition >*/
{
    a.r += b.r;
    a.i += b.i;
    return a;
}

np_double_complex np_dcsub(np_double_complex a, np_double_complex b)
/*< complex subtraction >*/
{
    a.r -= b.r;
    a.i -= b.i;
    return a;
}

np_double_complex np_dcmul(np_double_complex a, np_double_complex b)
/*< complex multiplication >*/
{
    np_double_complex c;
    c.r = a.r*b.r-a.i*b.i;
    c.i = a.i*b.r+a.r*b.i;
    return c;
}

kiss_fft_cpx np_dccmul(np_double_complex a, kiss_fft_cpx b)
/*< complex multiplication >*/
{
    kiss_fft_cpx c;
    c.r = a.r*b.r-a.i*b.i;
    c.i = a.i*b.r+a.r*b.i;
    return c;
}

np_double_complex np_dcdmul(np_double_complex a, kiss_fft_cpx b)
/*< complex multiplication >*/
{
    np_double_complex c;
    c.r = a.r*b.r-a.i*b.i;
    c.i = a.i*b.r+a.r*b.i;
    return c;
}

np_double_complex np_dcrmul(np_double_complex a, double b)
/*< complex by real multiplication >*/
{
    a.r *= b;
    a.i *= b;
    return a;
}

np_double_complex np_dcdiv(np_double_complex a, np_double_complex b)
/*< complex division >*/
{
    np_double_complex c;
    double r,den;
    if (fabsf(b.r)>=fabsf(b.i)) {
	r = b.i/b.r;
	den = b.r+r*b.i;
	c.r = (a.r+r*a.i)/den;
	c.i = (a.i-r*a.r)/den;
    } else {
	r = b.r/b.i;
	den = b.i+r*b.r;
	c.r = (a.r*r+a.i)/den;
	c.i = (a.i*r-a.r)/den;
    }
    return c;
}

double np_carg(np_double_complex z)
/*< replacement for cargf >*/
{
    extern double atan2(double,double);
    return atan2(z.i,z.r);
}

double np_cabsd(np_double_complex z)
/*< replacement for cabs >*/
{
    extern double hypot(double,double);
    return hypot(z.r,z.i);
}

float np_cabs(np_complex c)
/*< complex absolute value >*/
{
    return hypotf(crealf(c),cimagf(c));
}

float np_crealf(kiss_fft_cpx c)
/*< real part >*/
{
    return c.r;
}

float np_cimagf(kiss_fft_cpx c)
/*< imaginary part >*/
{
    return c.i;
}

void cprint (np_complex c)
/*< print a complex number (for debugging purposes) >*/
{
    printf("%g+%gi",crealf(c),cimagf(c));
}

kiss_fft_cpx np_cadd(kiss_fft_cpx a, kiss_fft_cpx b)
/*< complex addition >*/
{
    a.r += b.r;
    a.i += b.i;
    return a;
}

kiss_fft_cpx np_csub(kiss_fft_cpx a, kiss_fft_cpx b)
/*< complex subtraction >*/
{
    a.r -= b.r;
    a.i -= b.i;
    return a;
}

kiss_fft_cpx np_csqrtf (kiss_fft_cpx c)
/*< complex square root >*/
{


    extern float copysignf(float x, float y);

    float d, r, s;
    kiss_fft_cpx v;

    if (c.i == 0) {
      if (c.r < 0) {
	  v.r = 0.;
	  v.i = copysignf (sqrtf (-c.r), c.i);
      } else {
	  v.r =  fabsf (sqrtf (c.r));
	  v.i =  copysignf (0, c.i);
      }
    } else if (c.r == 0) {
	r = sqrtf (0.5 * fabsf (c.i));
	v.r = r;
	v.i = copysignf (r, c.i);
    } else {
	d = hypotf (c.r, c.i);
	/* Use the identity   2  Re res  Im res = Im x
	   to avoid cancellation error in  d +/- Re x.  */
	if (c.r > 0) {
	    r = sqrtf (0.5f * d + 0.5f * c.r);
	    s = (0.5f * c.i) / r;
        } else {
	    s = sqrtf (0.5f * d - 0.5f * c.r);
	    r = fabsf ((0.5f * c.i) / s);
        }
	v.r = r;
	v.i = copysignf (s, c.i);
    }
    return v;
}

kiss_fft_cpx np_cdiv(kiss_fft_cpx a, kiss_fft_cpx b)
/*< complex division >*/
{
    kiss_fft_cpx c;
    float r,den;
    if (fabsf(b.r)>=fabsf(b.i)) {
	r = b.i/b.r;
	den = b.r+r*b.i;
	c.r = (a.r+r*a.i)/den;
	c.i = (a.i-r*a.r)/den;
    } else {
	r = b.r/b.i;
	den = b.i+r*b.r;
	c.r = (a.r*r+a.i)/den;
	c.i = (a.i*r-a.r)/den;
    }
    return c;
}

kiss_fft_cpx np_cmul(kiss_fft_cpx a, kiss_fft_cpx b)
/*< complex multiplication >*/
{
    kiss_fft_cpx c;
    c.r = a.r*b.r-a.i*b.i;
    c.i = a.i*b.r+a.r*b.i;
    return c;
}

kiss_fft_cpx np_crmul(kiss_fft_cpx a, float b)
/*< complex by real multiplication >*/
{
    a.r *= b;
    a.i *= b;
    return a;
}

kiss_fft_cpx np_cneg(kiss_fft_cpx a)
/*< unary minus >*/
{
    a.r = -a.r;
    a.i = -a.i;
    return a;
}


kiss_fft_cpx np_conjf(kiss_fft_cpx z)
/*< complex conjugate >*/
{
    z.i = -z.i;
    return z;
}

float np_cabsf(kiss_fft_cpx z)
/*< replacement for cabsf >*/
{
    return hypotf(z.r,z.i);
}


float np_cargf(kiss_fft_cpx z)
/*< replacement for cargf >*/
{
    return atan2f(z.i,z.r);
}

kiss_fft_cpx np_ctanhf(kiss_fft_cpx z)
/*< complex hyperbolic tangent >*/
{
    float x, y, d;

    x = z.r;
    y = z.i;

    d = coshf(2*x) + cosf(2*y);
    z.r = sinhf(2*x)/ d;
    z.i = sinf (2*y)/ d;

    return z;
}

kiss_fft_cpx np_ccosf(kiss_fft_cpx z)
/*< complex cosine >*/
{
    float x, y;

    x = z.r;
    y = z.i;

    z.r = coshf(y)*cosf(x);
    z.i = -sinhf(y)*sinf(x);

    return z;
}

kiss_fft_cpx np_ccoshf(kiss_fft_cpx z)
/*< complex hyperbolic cosine >*/
{
    float x, y;

    x = z.r;
    y = z.i;

    z.r = coshf(x)*cosf(y);
    z.i = sinhf(x)*sinf(y);

    return z;
}


kiss_fft_cpx np_csinf(kiss_fft_cpx z)
/*< complex sine >*/
{
    float x, y;

    x = z.r;
    y = z.i;

    z.r = coshf(y)*sinf(x);
    z.i = sinhf(y)*cosf(x);

    return z;
}

kiss_fft_cpx np_csinhf(kiss_fft_cpx z)
/*< complex hyperbolic sine >*/
{
    float x, y;

    x = z.r;
    y = z.i;

    z.r = sinhf(x)*cosf(y);
    z.i = coshf(x)*sinf(y);

    return z;
}

kiss_fft_cpx np_clogf(kiss_fft_cpx z)
/*< complex natural logarithm >*/
{
    float x, y;

    x = z.r;
    y = z.i;

    z.r = logf(hypotf(x,y));
    z.i = atan2f(y,x);

    return z;
}

kiss_fft_cpx np_cexpf(kiss_fft_cpx z)
/*< complex exponential >*/
{
    float x, y;

    x = expf(z.r);
    y = z.i;

    z.r = x*cosf(y);
    z.i = x*sinf(y);

    return z;
}

kiss_fft_cpx np_ctanf(kiss_fft_cpx z)
/*< complex tangent >*/
{
    return np_cdiv(np_csinf(z),np_ccosf(z));
}

kiss_fft_cpx np_casinf(kiss_fft_cpx z)
/*< complex hyperbolic arcsine >*/
{
    float x, y;
    kiss_fft_cpx z2;

    x = z.r;
    y = z.i;

    if (0.0 == y) {
	z2.r = asinf(x);
	z2.i = 0.0;
    } else { 
	z.r = 1.0 - (x - y) * (x + y);
	z.i = -2.0 * x * y;
	z = np_csqrtf(z);

	z.r -= y;
	z.i += x;
	z = np_clogf(z);

	z2.r =  z.i;
	z2.i = -z.r;
    }
  
    return z2;
}

kiss_fft_cpx np_cacosf(kiss_fft_cpx z)
/*< complex hyperbolic arccosine >*/
{
    z = np_casinf(z);
    z.r = SF_PI/2 - z.r;
    z.i = - z.i;
    return z;
}

kiss_fft_cpx np_catanf(kiss_fft_cpx z)
/*< complex arctangent >*/
{
    kiss_fft_cpx z2;

    z2.r = -z.r;
    z2.i = 1.0-z.i;
    z.i += 1.0;
    
    z2 = np_clogf(np_cdiv(z,z2));
    z.r = -0.5f*z2.i;
    z.i = 0.5f*z2.r; 
    /* signs? */

    return z;
}     

kiss_fft_cpx np_catanhf(kiss_fft_cpx z)
/*< complex hyperbolic arctangent >*/
{
    kiss_fft_cpx z2;

    z2.r = -z.i;
    z2.i =  z.r;
    z2 = np_catanf(z2);
    z.r =  z2.i;
    z.i = -z2.r; 
    /* signs? */

    return z;
}     

kiss_fft_cpx np_casinhf(kiss_fft_cpx z)
/*< complex hyperbolic sine >*/
{
    kiss_fft_cpx z2;

    z2.r = -z.i;
    z2.i =  z.r;
    z2 = np_casinf(z2);
    z.r =  z2.i;
    z.i = -z2.r; 
    /* signs? */

    return z;
}     

kiss_fft_cpx np_cacoshf(kiss_fft_cpx z)
/*< complex hyperbolic cosine >*/
{
    kiss_fft_cpx z2;

    z2 = np_casinf(z);
    z.r = z2.i;
    z.i = SF_PI/2.0-z2.r;
    /* signs? */

    return z;
}

kiss_fft_cpx np_cpowf(kiss_fft_cpx a, kiss_fft_cpx b)
/*< complex power >*/
{
    float i, r, rho, theta;
    kiss_fft_cpx c;

    r = np_cabsf(a);
    i = np_cargf (a);

    if (r == 0.0) {
	c.r = 0.0;
	c.i = 0.0;
    } else {
	theta = i * b.r;
 
	if (b.i == 0.0) {
	    rho = powf (r,b.r);
	} else {
	    r = logf(r);

	    theta += r * b.i;
	    rho = expf(r * b.r - i * b.i);
	}

	c.r = rho * cosf (theta);
	c.i = rho * sinf (theta);
    }

    return c;
}

