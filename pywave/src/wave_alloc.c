#include <stdlib.h>
#include <stdio.h>
// #include <complex.h>

#include "wave_alloc.h"

void *np_alloc (size_t n    /* number of elements */, 
			  size_t size /* size of one element */)
/*< output-checking allocation >*/
{
    void *ptr; 
    
    size *= n;
    
    ptr = malloc (size);

    if (NULL == ptr)
	{
	printf("cannot allocate %lu bytes:", size);
	return NULL;
	}

    return ptr;
}

float *np_floatalloc (size_t n /* number of elements */)
	  /*< float allocation >*/ 
{
    float *ptr;
    ptr = (float*) np_alloc (n,sizeof(float));
    return ptr;
}

float **np_floatalloc2 (size_t n1 /* fast dimension */, 
				  size_t n2 /* slow dimension */)
/*< float 2-D allocation, out[0] points to a contiguous array >*/ 
{
    size_t i2;
    float **ptr;
    
    ptr = (float**) np_alloc (n2,sizeof(float*));
    ptr[0] = np_floatalloc (n1*n2);
    for (i2=1; i2 < n2; i2++) {
	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

float ***np_floatalloc3 (size_t n1 /* fast dimension */, 
				   size_t n2 /* slower dimension */, 
				   size_t n3 /* slowest dimension */)
/*< float 3-D allocation, out[0][0] points to a contiguous array >*/ 
{
    size_t i3;
    float ***ptr;
    
    ptr = (float***) np_alloc (n3,sizeof(float**));
    ptr[0] = np_floatalloc2 (n1,n2*n3);
    for (i3=1; i3 < n3; i3++) {
	ptr[i3] = ptr[0]+i3*n2;
    }
    return ptr;
}

float ****np_floatalloc4 (size_t n1 /* fast dimension */, 
				    size_t n2 /* slower dimension */, 
				    size_t n3 /* slower dimension */, 
				    size_t n4 /* slowest dimension */)
/*< float 4-D allocation, out[0][0][0] points to a contiguous array >*/ 
{
    size_t i4;
    float ****ptr;
    
    ptr = (float****) np_alloc (n4,sizeof(float***));
    ptr[0] = np_floatalloc3 (n1,n2,n3*n4);
    for (i4=1; i4 < n4; i4++) {
	ptr[i4] = ptr[0]+i4*n3;
    }
    return ptr;
}

int *np_intalloc (size_t n /* number of elements */)
	  /*< int allocation >*/  
{
    int *ptr;
    ptr = (int*) np_alloc (n,sizeof(int));
    return ptr;
}

int **np_intalloc2 (size_t n1 /* fast dimension */, 
			      size_t n2 /* slow dimension */)
/*< float 2-D allocation, out[0] points to a contiguous array >*/  
{
    size_t i2;
    int **ptr;
    
    ptr = (int**) np_alloc (n2,sizeof(int*));
    ptr[0] = np_intalloc (n1*n2);
    for (i2=1; i2 < n2; i2++) {
	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

int ***np_intalloc3 (size_t n1 /* fast dimension */, 
			       size_t n2 /* slower dimension */, 
			       size_t n3 /* slowest dimension */)
/*< int 3-D allocation, out[0][0] points to a contiguous array >*/ 
{
    size_t i3;
    int ***ptr;
    
    ptr = (int***) np_alloc (n3,sizeof(int**));
    ptr[0] = np_intalloc2 (n1,n2*n3);
    for (i3=1; i3 < n3; i3++) {
	ptr[i3] = ptr[0]+i3*n2;
    }
    return ptr;
}

np_complex *np_complexalloc (size_t n /* number of elements */) 
	  /*< complex allocation >*/
{
    np_complex *ptr;
    ptr = (np_complex*) np_alloc (n,sizeof(np_complex));
    return ptr;
}

np_complex **np_complexalloc2 (size_t n1 /* fast dimension */, 
					 size_t n2 /* slow dimension */)
	  /*< complex 2-D allocation >*/ 
{
    size_t i2;
    np_complex **ptr;
    
    ptr = (np_complex**) np_alloc (n2,sizeof(np_complex*));
    ptr[0] = np_complexalloc (n1*n2);
    for (i2=1; i2 < n2; i2++) {
	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

np_complex ***np_complexalloc3 (size_t n1 /* fast dimension */, 
					  size_t n2 /* slower dimension */, 
					  size_t n3 /* slowest dimension */)
	  /*< complex 3-D allocation >*/ 
{
    size_t i3;
    np_complex ***ptr;
    
    ptr = (np_complex***) np_alloc (n3,sizeof(np_complex**));
    ptr[0] = np_complexalloc2 (n1,n2*n3);
    for (i3=1; i3 < n3; i3++) {
	ptr[i3] = ptr[0]+i3*n2;
    }
    return ptr;
}


