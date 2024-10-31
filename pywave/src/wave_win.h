#ifndef _win_h
#define _win_h


void win_weight2d(float **din /*input data*/, 
		int iw1 /*starting window 1 in dst*/,
		int iw2 /*starting window 2 in dst*/,		
		int nw1 /*no of windows 1 in src*/,
		int nw2 /*no of windows 2 in src*/,
		int n1win /*window length 1 in src*/,
		int n2win /*window legnth 2 in src*/,		
		int ov1 /*copy length in axis1*/,
		int ov2 /*copy length in axis2*/);
/*<weights of local window in 2D (checked 100% correct>*/


void win_weight3d(float ***din /*input data*/, 
		int iw1 /*starting window 1 in dst*/,
		int iw2 /*starting window 2 in dst*/,
		int iw3 /*starting window 3 in dst*/,						
		int nw1 /*no of windows 1 in src*/,
		int nw2 /*no of windows 2 in src*/,
		int nw3 /*no of windows 3 in src*/,		
		int n1win /*window length 1 in src*/,
		int n2win /*window legnth 2 in src*/,		
		int n3win /*window legnth 3 in src*/,				
		int ov1 /*copy length in axis1*/,
		int ov2 /*copy length in axis2*/,
		int ov3 /*copy length in axis3*/);
/*<weights of local window in 3D >*/

#endif
