## This DEMO is a 3D acoustic wave simulation using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

from pywave import aps3d,aps2d
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import plot3d #pip install git+https://github.com/aaspip/pyseistr

nz=81
nx=81
dz=20
dx=20
nt=1501
dt=0.001

v=np.arange(nz)*20*1.2+1500;
vel=np.zeros([nz,nx]);
for ii in range(nx):
		vel[:,ii]=v;

## plot 3D velocity
plt.imshow(vel,aspect='auto',cmap="seismic");
# plt.gca().invert_yaxis();
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('2D velocity model')
plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (m/s)');

plt.savefig(fname='vel2d.png',format='png',dpi=300)
plt.show()


data=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],jsnap=0);
data=data.reshape(nt,nx,order='F'); #[x,z]

# data=aps2d(vel,2001,0.001,ax=[0,dx,nx],az=[0,dz,nz],jsnap=0,ns=2,sx=[40,50],sz=[40,40],f=[30,30],t=[0.1,0.21],A=[1,1]);
# data=data.reshape(2001,nx,order='F'); #[x,y,z]

## plot 2D data
plt.imshow(data,aspect='auto',cmap="seismic");
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data')
plt.savefig(fname='data2d.png',format='png',dpi=300)
plt.show()

## 
## save wavefields
[data,wfd]=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],ifsnaps=1,jsnap=4);

# plot3d(data,figname='data3d.png',format='png',dpi=300)
## plot 3D wavefields
fignames=[]
for ii in range(0,376-100,20):
	print(ii)
# 	plot3d(wfd[:,:,:,ii],vmin=-1,vmax=1,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False)
	plt.imshow(wfd[:,:,ii],aspect='auto',clim=(-1, 1));
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield at %g s'%(ii*dt*4))
	figname='wfd2d-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 3D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'wfd2ds.gif')






