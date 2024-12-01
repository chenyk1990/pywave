## This DEMO is a 3D acoustic wave simulation using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

import os

if os.path.isdir('./npys') == False:  
	os.makedirs('./npys',exist_ok=True)

if os.path.isdir('./figs') == False:  
	os.makedirs('./figs',exist_ok=True)
	
if os.path.isdir('./gifs') == False:  
	os.makedirs('./gifs',exist_ok=True)

from pywave import aps3d
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import plot3d #pip install git+https://github.com/aaspip/pyseistr

nz=81
nx=81
ny=81
dz=20
dx=20
dy=20
nt=1501
dt=0.001

v=np.arange(nz)*20*1.2+1500;
vel=np.zeros([nz,nx,ny]);
for ii in range(nx):
	for jj in range(ny):
		vel[:,ii,jj]=v;

## plot 3D velocity
plot3d(vel,figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Velocity (m/s)',showf=False,close=False)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('3D velocity model')
plt.savefig(fname='figs/vel3d.png',format='png',dpi=300)
plt.show()


# data=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],jsnap=0);
# data=data.reshape(nt,nx,ny,order='F'); #[x,y,z]
# 

## save wavefields
[data,wfd]=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],ifsnaps=1,jsnap=4);

# plot3d(data,figname='data3d.png',format='png',dpi=300)
## plot 3D data
plot3d(data,z=np.arange(nt)*dt,x=np.arange(nx)*dz,y=np.arange(nz)*dz,showf=False,close=False)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('3D synthetic data')
plt.savefig(fname='figs/data3d.png',format='png',dpi=300)
plt.show()

## plot 3D wavefields
fignames=[]
for ii in range(0,376-100,20):
	print(ii)
	plot3d(wfd[:,:,:,ii],vmin=-1,vmax=1,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('3D wavefield at %g s'%(ii*dt*4))
	figname='figs/wfd3d-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)
	
## plot 3D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'gifs/wfd3ds.gif')


    
    
    
