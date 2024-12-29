
#ifndef butter_h
#define butter_h

#include <stdbool.h>
#include "wave_butter.h"
/*^*/
typedef struct Np_Butter *np_butter;
/* abstract data type */
/*^*/

struct Np_Butter {
    bool low;
    int nn;
    float **den, mid;
};

np_butter np_butter_init(bool low     /* low-pass (or high-pass) */, 
		   float cutoff /* cut off frequency */, 
		   int nn       /* number of poles */);
/*< initialize >*/

void np_butter_close(np_butter bw);
/*< Free allocated storage >*/

void np_butter_apply (const np_butter bw, int nx, float *x /* data [nx] */);
/*< filter the data (in place) >*/

void np_reverse (int n1, float* trace); 
/*< reverse a trace >*/

#endif