## This DEMO is a 2D acoustic wave simulation and time-reversal imaging using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin
import os

if os.path.isdir('./npys') == False:  
	os.makedirs('./npys',exist_ok=True)

if os.path.isdir('./figs') == False:  
	os.makedirs('./figs',exist_ok=True)
	
if os.path.isdir('./gifs') == False:  
	os.makedirs('./gifs',exist_ok=True)
	
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

## plot 2D velocity
plt.imshow(vel,aspect='auto',cmap="seismic", extent=[0,dx*(nx-1),dz*(nz-1),0]);
# plt.gca().invert_yaxis();
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('2D velocity model')
plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (m/s)');

plt.savefig(fname='figs/vel2d.png',format='png',dpi=300)
plt.show()


data=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],jsnap=0);
data=data.reshape(nt,nx,order='F'); #[x,z]

# data=aps2d(vel,2001,0.001,ax=[0,dx,nx],az=[0,dz,nz],jsnap=0,ns=2,sx=[40,50],sz=[40,40],f=[30,30],t=[0.1,0.21],A=[1,1]);
# data=data.reshape(2001,nx,order='F'); #[x,y,z]

## plot 2D data
plt.imshow(data,aspect='auto',cmap="seismic", extent=[0,dx*(nx-1),dt*(nt-1),0]);
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('2D synthetic data')
plt.savefig(fname='figs/data2d.png',format='png',dpi=300)
plt.show()

## 
## save wavefields
[data,wfd]=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],ifsnaps=1,jsnap=4,verb=0,sx=[30,40],sz=[30,40],f=[10,10],t=[0.2,0.35],A=[1,2]);

# plot3d(data,figname='data3d.png',format='png',dpi=300)
## plot 2D wavefields
fignames=[]
for ii in range(0,376-100,20):
	print(ii)
# 	plot3d(wfd[:,:,:,ii],vmin=-1,vmax=1,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False)
	plt.imshow(wfd[:,:,ii],aspect='auto',clim=(-2, 2), extent=[0,dx*(nx-1),dz*(nz-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield at %g s'%(ii*dt*4))
	figname='figs/wfd2d-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 2D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'gifs/wfd2ds.gif')


####################################################################################################
### Doing TRI
####################################################################################################
[n1,n2]=data.shape
# data_mask0=data
ng=4
datas=np.zeros([n1,n2,ng])
dg=int(nx/ng)
print('dg=',dg)
for ii in range(ng):
	print("Group ",ii)
	inds=np.linspace(ii*dg+0,ii*dg+dg-1,dg,dtype='int')
# 	datas[:,inds,ii]=np.transpose(data[:,inds],(1, 0, 2))
	datas[:,inds,ii]=data[:,inds]
	[img,wfd]=aps2d(vel,nt,dt,ax=[0,dx,nx],az=[0,dz,nz],ifsnaps=1,jsnap=4,dat=datas[:,:,ii],tri=True);
	print('wfd max,min',wfd.max(),wfd.min())
	np.save('npys/wfd-tri-%d'%ii,wfd)
	plt.imshow(datas[:,:,ii],aspect='auto',cmap="seismic", extent=[0,dx*(nx-1),dt*(nt-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	# plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Time (s)",fontsize='large', fontweight='normal')
	plt.title('2D synthetic data - %d'%(ii+1))
	plt.savefig(fname='figs/data2d-%d.png'%(ii+1),format='png',dpi=300)
	plt.show()

imag=1
for ii in range(4):
	print("Group ",ii)
	tmp=np.load('npys/wfd-tri-%d.npy'%ii)
	imag=imag*tmp*tmp
imag=np.sum(imag,axis=2)
imag=imag/imag.max()

plt.imshow(imag,aspect='auto',cmap="seismic",clim=[0,0.5]);
# plt.gca().invert_yaxis();
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('Location image')
plt.colorbar(orientation='horizontal',shrink=0.6,label='Amplitude');
plt.savefig(fname='figs/location-new.png',format='png',dpi=300)
plt.show()

## All
jj=0
wfd=np.load('npys/wfd-tri-%d.npy'%jj)

fignames=[]
for ii in range(375,0,-20):
	print(ii)
# 	plot3d(wfd[:,:,:,ii],vmin=-1,vmax=1,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False)
	plt.imshow(wfd[:,:,ii],aspect='auto',clim=(-2, 2), extent=[0,dx*(nx-1),dz*(nz-1),0]);
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('2D wavefield at %g s'%(1.5-(375-ii)*dt*4))
	figname='figs/wfd2d-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)

## plot 2D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'gifs/wfd2ds-%d.gif'%jj)


