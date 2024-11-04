## This DEMO is a 3D acoustic wave simulation using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

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
plt.savefig(fname='vel3d.png',format='png',dpi=300)
plt.show()


data=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],jsnap=0);
data=data.reshape(nt,nx,ny,order='F'); #[x,y,z]

## plot 3D data
plot3d(data,z=np.arange(nt)*dt,x=np.arange(nx)*dz,y=np.arange(nz)*dz,showf=False,close=False)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('3D synthetic data')
plt.savefig(fname='data3d.png',format='png',dpi=300)
plt.show()



