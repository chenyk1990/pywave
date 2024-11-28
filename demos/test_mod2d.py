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
data=data.reshape(nt,nx,order='F'); #[x,y,z]

## plot 3D data
plt.imshow(vel,aspect='auto',cmap="seismic");
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data')
plt.savefig(fname='data2d.png',format='png',dpi=300)
plt.show()





