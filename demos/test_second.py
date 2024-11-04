## This DEMO is a 3D acoustic wave simulation using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

from pywave import aps3d
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import plot3d

nt=1501
nz=81
nx=81
ny=81

v=np.arange(nz)*20*1.2+1500;
vel=np.zeros([nz,nx,ny]);
for ii in range(nx):
	for jj in range(ny):
		vel[:,ii,jj]=v;
# plot3d(vel,cmap=plt.cm.jet,figname='vel3d.png',format='png',dpi=300)
# 
# data=aps3d(vel,nt=nt,dt=0.001,ax=[0,20,81],ay=[0,20,81],az=[0,20,81]);
# data=data.reshape(nt,81,81,order='F'); #[x,y,z]
# 

## save wavefields
[data,wfd]=aps3d(vel,nt=nt,dt=0.001,ax=[0,20,81],ay=[0,20,81],az=[0,20,81],ifsnaps=1,jsnap=4);

plot3d(data,figname='data3d.png',format='png',dpi=300)


for ii in range(0,376-100,20):
	print(ii)
	plot3d(wfd[:,:,:,ii],figname='wfd3d-%d.png'%ii,format='png',dpi=300)
	
