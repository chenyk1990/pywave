import numpy as np
from apscfun import *

def aps3d(vel,nt=1501,dt=0.001,ax=[0,20,81],ay=[0,20,81],az=[0,20,81],nbt=30,ct=0.01,jsnap=4,abc=True,ifsnaps=False,ps=True,verb=True):
	'''
	aps3d: 3D acoustic wavefield modeling using the pseudo-spectral method  
	
	INPUT
	vel: velocity model [nz,nx,ny]
	nt: number of samples
	ax: axis x [ox,dx,nx]
	ay: axis y [oy,dy,ny]
	az: axis z [oz,dz,nz]
	verb
	
	OUTPUT
	data
	
	EXAMPLE
	demos/test_mod3d.py
	
	HISTORY
	Original version by Yangkang Chen, Oct 31, 2024
	
	REFERENCE
	Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021, A compact program for 3D passive seismic source-location imaging, Seismological Research Letters, 92(5), 3187–3201.
	
	See the original Madagascar version with documentation at
	https://github.com/chenyk1990/passive_imaging/blob/main/mod3d.c
	
	'''
	
	
	ox=ax[0];dx=ax[1];nx=ax[2];
	oy=ay[0];dy=ay[1];ny=ay[2];
	oz=az[0];dz=az[1];nz=az[2];
	
	dout=aps3dc(vel,nt,nx,ny,nz,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oy,dy,oz,dz);
	
	

	

	return 


