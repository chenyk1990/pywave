#ifndef _memcpy_h
#define _memcpy_h

#include "npre_dtype.h"

/*real value functions*/
void mcp(float *dst /*destination*/, 
		float *src /*source*/,
		int s1d /*starting index in dst*/,
		int s1s /*starting index in src*/,
		int l1  /*copy length in axis1*/);
/*<memory copy in 1D case>*/


void mcp_ad(float *dst /*destination*/, 
		float *src /*source*/,
		int s1d /*starting index in dst*/,
		int s1s /*starting index in src*/,
		int l1  /*copy length in axis1*/);
/*<memory copy and addition in 1D case>*/


void mcp2d(float **dst /*destination*/, 
		float **src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,		
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<memory copy in 2D case>*/


void mcp_ad2d(float **dst /*destination*/, 
		float **src /*source (bigger)*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,		
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<memory copy and addition in 2D case>*/


void mcp3d(float ***dst /*destination*/, 
		float ***src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis2*/);
/*<memory copy in 3D case>*/


void mcp3d1d(float ***dst /*destination*/, 
		float *src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/,
		int n1 /*source length in axis1*/,
		int n2 /*source length in axis2*/,
		int n3 /*source length in axis3*/);
/*<memory copy in 3D case>*/

void mcp3d3d1d(float *dst /*destination*/, 
		float ***src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/,
		int n1 /*destination length in axis1*/,
		int n2 /*destination length in axis2*/,
		int n3 /*destination length in axis3*/);
/*<memory copy in 3D case>*/


void mcp3d1d1d(float *dst /*destination*/, 
		float *src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/,
		int n1 /*source length in axis1*/,
		int n2 /*source length in axis2*/,
		int n3 /*source length in axis3*/);
/*<memory copy in 3D case: 1D to 1D array>*/


void mcp3d1d1dint(int *dst /*destination*/, 
		int *src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/,
		int n1 /*source length in axis1*/,
		int n2 /*source length in axis2*/,
		int n3 /*source length in axis3*/);
/*<memory copy in 3D case: 1D to 1D integer array>*/


void mcp_ad1d3d(float *dst /*destination*/, 
		float ***src /*source (bigger)*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,
		int n1 /*source length in axis1*/,
		int n2 /*source length in axis2*/,
		int n3 /*source length in axis3*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/);
/*<memory copy and addition in 3D case>*/


void mcp_ad3d(float ***dst /*destination*/, 
		float ***src /*source (bigger)*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/);
/*<memory copy and addition in 3D case>*/


/*complex value functions*/
void cmcp(kiss_fft_cpx *dst /*destination*/, 
		kiss_fft_cpx *src /*source*/,
		int s1d /*starting index in dst*/,
		int s1s /*starting index in src*/,
		int l1  /*copy length in axis1*/);
/*<memory copy in 1D case>*/


void cmcp_ad(kiss_fft_cpx *dst /*destination*/, 
		kiss_fft_cpx *src /*source*/,
		int s1d /*starting index in dst*/,
		int s1s /*starting index in src*/,
		int l1  /*copy length in axis1*/);
/*<memory copy and addition in 1D case>*/


void cmcp2d(kiss_fft_cpx **dst /*destination*/, 
		kiss_fft_cpx **src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,		
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<memory copy in 2D case>*/


void cmcp_ad2d(kiss_fft_cpx **dst /*destination*/, 
		kiss_fft_cpx **src /*source (bigger)*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,		
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<memory copy and addition in 2D case>*/


void cmcp3d(kiss_fft_cpx ***dst /*destination*/, 
		kiss_fft_cpx ***src /*source*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis2*/);
/*<memory copy in 3D case>*/


void cmcp_ad3d(kiss_fft_cpx ***dst /*destination*/, 
		kiss_fft_cpx ***src /*source (bigger)*/,
		int s1d /*starting index 1 in dst*/,
		int s2d /*starting index 2 in dst*/,	
		int s3d /*starting index 3 in dst*/,					
		int s1s /*starting index 1 in src*/,
		int s2s /*starting index 2 in src*/,
		int s3s /*starting index 3 in src*/,		
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*copy length in axis3*/);
/*<memory copy and addition in 3D case>*/


/*Memory initialization*/
void mi2d(float **din /*input data*/, 
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<Memory initialization in 2D case>*/


void cmi2d(kiss_fft_cpx **din /*input data*/, 
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/);
/*<Memory initialization in 2D case>*/


void mi3d(float ***din /*input data*/, 
		int l1 /*copy length in axis1*/,
		int l2 /*copy length in axis2*/,
		int l3 /*length in axis3*/);
/*<Memory initialization in 3D case>*/


void cmi3d(kiss_fft_cpx ***din /*input data*/, 
		int l1 /*length in axis1*/,
		int l2 /*length in axis2*/,
		int l3 /*length in axis3*/);
/*<Memory initialization in 3D case>*/

#endif
