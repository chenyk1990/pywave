## This DEMO is a 2D acoustic wave simulation for active source (on surface) using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin
import os

if os.path.isdir('./npys') == False:  
	os.makedirs('./npys',exist_ok=True)

if os.path.isdir('./figs') == False:  
	os.makedirs('./figs',exist_ok=True)
	
if os.path.isdir('./gifs') == False:  
	os.makedirs('./gifs',exist_ok=True)
	
from pywave import aps3d,aps2d,afd2d
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
plt.savefig(fname='figs/vel2d-active.png',format='png',dpi=300)
plt.show()


## simulation goes here (PS)
[data,wfd]=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],jsnap=4,ns=1,sx=[sx],sz=[sz],f=[10],t=[0.01],A=[1],nbt=30,ct=0.01)
wfd=wfd/wfd.max();

# ## plot 2D data
plt.imshow(data,aspect='auto',cmap=cseis(), extent=[0,dx*(nx-1),dt*(nt-1),0]);
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data (PS)')
plt.savefig(fname='figs/data2d-active-ps.png',format='png',dpi=300)
plt.show()

## simulation goes here (FD)
[data2,wfd2]=afd2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],jsnap=4,ns=1,sx=[sx],sz=[sz],f=[10],t=[0.01],A=[1],nbt=30,ct=0.01)
wfd2=wfd2/wfd2.max();

plt.imshow(data2,aspect='auto',cmap=cseis(), extent=[0,dx*(nx-1),dt*(nt-1),0]);
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data (FD)')
plt.savefig(fname='figs/data2d-active-fd.png',format='png',dpi=300)
plt.show()

#plot traces
ix=20
t=np.linspace(0,(nt-1)*dt,nt)
plt.figure(figsize=(10, 4))
plt.plot(t,data[:,ix],'k-', linewidth=2, label='PS')
plt.plot(t,data2[:,ix],'r--', linewidth=2, label='FD')
plt.gca().legend(loc='lower right',fontsize='large');
  
plt.title('PS VS FD (ix=%g m)'%(ix*dx),fontsize='large', fontweight='normal')
plt.ylabel('Amplitude',fontsize='large', fontweight='normal')
plt.xlabel('Time (s)',fontsize='large', fontweight='normal')
plt.savefig(fname='figs/data2d-active-tracenew.png',format='png',dpi=300)
plt.show()

#plot 1D wavefield 
it=120
iz=20
x=np.linspace(0,(nx-1)*dx,nx)
plt.figure(figsize=(10, 4))
plt.plot(x,wfd[iz,:,it],'k-', linewidth=2, label='PS')
plt.plot(x,wfd2[iz,:,it],'r--', linewidth=2, label='FD')
plt.gca().legend(loc='lower right',fontsize='large');
  
plt.title('PS VS FD (iz=%g m, it=%g s)'%(iz*dz,it*dt*4),fontsize='large', fontweight='normal')
plt.ylabel('Amplitude',fontsize='large', fontweight='normal')
plt.xlabel('X (m)',fontsize='large', fontweight='normal')
plt.savefig(fname='figs/data2d-active-trace2new.png',format='png',dpi=300)
plt.show()

# plot 2D wavefields (PS)
fignames=[]
for ii in range(0,wfd.shape[2],20):
	print(ii)
	plt.imshow(wfd[:,:,ii],aspect='auto',clim=(-0.1,0.1), extent=[0,dx*(nx-1),dz*(nz-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield (PS) at %g s'%(ii*dt*4))
	figname='figs/wfd2da-ps-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 2D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'gifs/wfd2ds-active-ps.gif')

# plot 2D wavefields (FD)
fignames=[]
for ii in range(0,wfd2.shape[2],20):
	print(ii)
	plt.imshow(wfd2[:,:,ii],aspect='auto',clim=(-0.1,0.1), extent=[0,dx*(nx-1),dz*(nz-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield (FD) at %g s'%(ii*dt*4))
	figname='figs/wfd2da-fd-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 2D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'gifs/wfd2ds-active-fd.gif')




