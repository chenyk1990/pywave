## This DEMO is a 2D acoustic wave simulation for active source (on surface) using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

from pywave import aps3d,aps2d
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import cseis #pip install git+https://github.com/aaspip/pyseistr

nz=81
nx=81
dz=20
dx=20
nt=2001
dt=0.001
sx=40 #on grid (40*dx)
sz=0  #on grid (0*dz)

v=np.arange(nz)*20*1.2+1500;
vel=np.zeros([nz,nx]);
for ii in range(nx):
	if ii<30:
		vel[:,ii]=1500;
	elif ii>=30 and ii<60:
		vel[:,ii]=2000;
	else:
		vel[:,ii]=3000;
vel=np.transpose(vel,[1,0]);

## plot 2D velocity
plt.imshow(vel,aspect='auto',cmap="seismic", extent=[0,dx*(nx-1),dz*(nz-1),0]);
# plt.gca().invert_yaxis();
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('2D velocity model')
plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (m/s)');
plt.plot(sx*dx,(sz+1)*dz,'*',color='r', markersize=12)
plt.savefig(fname='vel2d-active.png',format='png',dpi=300)
plt.show()


## simulation goes here
[data,wfd]=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],jsnap=4,ns=1,sx=[sx],sz=[sz],f=[10],t=[0.01],A=[1],nbt=30,ct=0.01)
wfd=wfd/wfd.max();

# ## plot 2D data
plt.imshow(data,aspect='auto',cmap=cseis(), extent=[0,dx*(nx-1),dt*(nt-1),0]);
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data')
plt.savefig(fname='data2d-active.png',format='png',dpi=300)
plt.show()

# plot3d(data,figname='data3d.png',format='png',dpi=300)
## plot 2D wavefields
fignames=[]
for ii in range(0,wfd.shape[2],20):
	print(ii)
	plt.imshow(wfd[:,:,ii],aspect='auto',clim=(-0.1,0.1), extent=[0,dx*(nx-1),dz*(nz-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield at %g s'%(ii*dt*4))
	figname='wfd2da-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 2D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'wfd2ds-active.gif')






