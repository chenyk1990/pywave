## This DEMO is a 3D acoustic wave simulation using pseudo-spectral method
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin
#
# This script reproduces the synthetic examples in the following paper
# Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021, A compact program for 3D passive seismic source-location imaging, Seismological Research Letters, 92, 3187–3201.
# 
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


## Simulate data
# data=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],jsnap=0);

## Simulate data and wavefields
[data,wfd]=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],ifsnaps=1,jsnap=4);
np.save('data',data)
# data=np.load('data.npy')

[n1,n2,n3]=data.shape
# data_mask0=data
ng=4
datas=np.zeros([n1,n2,n3,ng])
dg=int(nx/ng)
print('dg=',dg)
for ii in range(ng):
	print("Group ",ii)
	inds=np.linspace(ii*dg+0,ii*dg+dg-1,dg,dtype='int')
	datas[:,inds,:,ii]=np.transpose(data[:,inds,:],(1, 0, 2))#why?, (forgot)
	[img,wfd]=aps3d(vel,nt,dt,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],ifsnaps=1,jsnap=4,dat=datas[:,:,:,ii],tri=True);
	print('wfd max,min',wfd.max(),wfd.min())
	np.save('wfd-tri-%d'%ii,wfd)

imag=1
for ii in range(ng):
	print("Group ",ii)
	tmp=np.load('wfd-tri-%d.npy'%ii)
	imag=imag*tmp*tmp
imag=np.sum(imag,axis=3)
imag=imag/imag.max()

## plot 3D Location Imag
plot3d(imag,frames=[30,30,30],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False,vmin=0,vmax=0.05)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('3D source-location image')
plt.savefig(fname='imag3d-1.png',format='png',dpi=300)
plt.show()

plot3d(imag,frames=[40,40,40],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False,vmin=0,vmax=0.5)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('3D source-location image')
plt.savefig(fname='imag3d-2.png',format='png',dpi=300)
plt.show()

plot3d(imag,frames=[50,50,50],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False,vmin=0,vmax=0.1)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
plt.title('3D source-location image')
plt.savefig(fname='imag3d-3.png',format='png',dpi=300)
plt.show()

## plot 3D data
data=np.load('data.npy')
plot3d(data,z=np.arange(nt)*dt,x=np.arange(nx)*dz,y=np.arange(nz)*dz,showf=False,close=False)
plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
plt.title('3D synthetic data')
plt.savefig(fname='data3d.png',format='png',dpi=300)
plt.show()

## plot grouped 3D data
data=np.load('data.npy')
[n1,n2,n3]=data.shape
ng=4
datas=np.zeros([n1,n2,n3,ng])
dg=int(n2/ng)
print('dg=',dg)
for ii in range(ng):
	print("Group ",ii)
	inds=np.linspace(ii*dg+0,ii*dg+dg-1,dg,dtype='int')
	datas[:,inds,:,ii]=np.transpose(data[:,inds,:],(1, 0, 2))
	plot3d(datas[:,:,:,ii],z=np.arange(nt)*dt,x=np.arange(nx)*dz,y=np.arange(nz)*dz,showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
	plt.title('3D synthetic data')
	plt.savefig(fname='data3d-mask-%d.png'%ii,format='png',dpi=300)
	plt.show()

## plot 3D wavefields
jj=3
wfd=np.load('wfd-tri-%d.npy'%jj)

fignames=[]
for ii in range(375,0,-20):
	print(ii)
	plot3d(wfd[:,:,:,ii],vmin=-100,vmax=100,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('3D wavefield at %g s'%(1.5-(375-ii)*dt*4))
	figname='wfd3d-%d.png'%ii;fignames.append(figname);
	plt.savefig(fname=figname,format='png',dpi=300)
	
## plot 3D wavefield animation in GIF
from pyseistr import gengif #pip install git+https://github.com/aaspip/pyseistr
gengif(fignames,'wfd3ds-tri-%d.gif'%jj)


    
    
    
