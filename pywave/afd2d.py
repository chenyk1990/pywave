import numpy as np
from afdcfun import *

def afd2d(vel,nt=1501,dt=0.001,ax=[0,20,81],az=[0,20,81],ns=2,sx=[30,40],sz=[30,40],f=[10,10],t=[0.2,0.35],A=[1,2],nbt=30,ct=0.01,jsnap=4,abc=True,ifsnaps=False,ps=True,tri=False,dat=None,verb=True):
	'''
	afd2d: 2D acoustic wavefield modeling using the finite difference method  
	
	INPUT
	vel: velocity model [nz,nx]
	nt: number of samples
	dt: temporal sampling (time interval)
	ax: axis x [ox,dx,nx]
	az: axis z [oz,dz,nz]
	nbt: 	size of ABC layers
	ct:		ABCparameter
	jsnap:	output wavefield interval
	abc:	if abc (default: True)
	ifsnaps:if output wavefield (default False)
	ps: if pseudo-spectral
	tri: if choose time reversal imaging (True: TRI; False: Forward modeling)
	verb: verbosity flag
	
	#below is for synthetic parameters
	ns:		number of shots
	sx:		source x vector in samples
	sz:		source z vector in samples
	f:		source freq vector in Hz
	t:		source time vector in sec
	A:		source amplitude
	
	OUTPUT
	data
	[wfd] if jsnap>0
	
	EXAMPLE
	demos/test_mod2d.py
	
	HISTORY
	Original version by Yangkang Chen, Oct 31, 2024
	
	REFERENCE
	Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021, A compact program for 3D passive seismic source-location imaging, Seismological Research Letters, 92(5), 3187–3201.
	
	See the original Madagascar version with documentation at
	https://github.com/chenyk1990/reproducible_research/blob/master/nonMada/microtri2d/mod2d.c
	
	'''
	vel=vel.flatten(order='F').astype(np.float32)
	ox=ax[0];dx=ax[1];nx=ax[2];
	oz=az[0];dz=az[1];nz=az[2];
	
	sx=np.array(sx);
	sz=np.array(sz);
	f=np.array(f);
	t=np.array(t);
	A=np.array(A);
	
# 	vel=vel.flatten(order='F',dtype='float32')

	if tri:
# 		if dat=='None':
# 			print('Need input data')
		print('Time-reversal imaging')
		print('dat.shape',dat.shape)
		source=dat.flatten(order='F').astype(np.float32); 
		tri=1;
	else:
		source=np.concatenate([sx,sz,f,t,A],axis=0,dtype='float32'); #remember: source size is ns*5
	
	print(source)
	print(nt,nx,nz)
	print(tri)
	dout=afd2dc(vel,source,tri,nt,nx,nz,ns,verb,jsnap,ifsnaps,abc,nbt,ct,dt,ox,dx,oz,dz);
	
	if jsnap>0:
	
		ntsnap=0;
		if jsnap:
			for it in range(nt):
				if np.mod(it,jsnap)==0:
					ntsnap=ntsnap+1;
		print('ntsnap=',ntsnap);
		wfd=dout[nt*nx:].reshape(nz,nx,ntsnap,order='F'); #[x,y,z]
		dout=dout[0:nt*nx].reshape(nt,nx,order='F'); #[x,y,z]
		# tri,&nt,&nx,&nz,&ns,&verb,&jsnap,&ifsnaps,&abs,&nbt,ct,dt,ox,dx,oz,dz);

	if jsnap>0:
		return dout,wfd
	else:
		return dout


