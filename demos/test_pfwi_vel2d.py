## This is to perform velocity inversion using passive seismic data

from pywave import pfwi
import numpy as np
import matplotlib.pyplot as plt
import os
from pyseistr import binread,smooth #pip install git+https://github.com/aaspip/pyseistr
# DataPath
#https://github.com/chenyk1990/cykd2/blob/master/various/cyksmall/marmvel.bin

## load velocity
# marmvel=binread(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/marmvel.bin',n1=751,n2=2301)

marmvel=binread('marmvel.bin',n1=751,n2=2301)

marmvel=marmvel*0.001;
marmvel=marmvel[0:240,1250:2251];
marmvel=marmvel[::2,::3]*1.8-1.2;
vel=np.zeros([130,334]);vel[10:,:]=marmvel;vel[0:10,:]=np.multiply(np.ones([10,1]),marmvel[0,:]);

## obtain start velocity
vel_s=np.concatenate([vel[0:14],1./smooth(1/vel[14:,:],[20,20,1])],axis=0)
q=np.ones(vel.shape)*10000;
plt.subplot(2,1,1);
plt.imshow(vel);
plt.subplot(2,1,2);
plt.imshow(vel_s);
plt.show()

## wavelet
from pyseistr import ricker
trace=np.zeros(3001);trace[99]=1000000;dt=0.001;
wav,tw=ricker(10,dt,0.2)
wavelet=np.convolve(trace,wav,mode='same');
plt.plot(wavelet);plt.show()

## src
[nz,nx]=vel.shape
nt=len(wavelet)
src=np.zeros([nz,nx,nt,3]);
s1=np.array([[110,120,117,120,112,115],[100,110,165,205,220,260],[100,300,500,700,900,1100]]);
s2=np.array([[115,111,120,122,118,115],[95,120,170,200,230,255],[100,300,500,700,900,1100]]);
s3=np.array([[112,118,111,120,116,110],[90,110,145,180,200,240],[100,300,500,700,900,1100]]);
nfrac=s1.shape[1]; #number of fractions
for ii in range(nfrac):
	src[s1[0,ii],s1[1,ii],s1[2,ii],0] = 1e6;
	src[s2[0,ii],s2[1,ii],s2[2,ii],1] = 1e6;
	src[s3[0,ii],s3[1,ii],s3[2,ii],2] = 1e6;
	
for ii in range(nz):
	for jj in range(nx):
		for kk in range(3):
			src[ii,jj,:,kk]=np.convolve(src[ii,jj,:,kk],wav,mode='same');
# plt.imshow(np.sum(np.sum(src,3),2));
# plt.show()

## plot source
plt.subplot(1,2,1);
plt.plot(src[110,100,:,0]);
plt.subplot(1,2,2);
plt.plot(src[115,260,:,0]);
plt.show()

## generate data
mypar={'nz':130, 'nx':334, 'dz': 0.008, 'dx': 0.012, 'oz': 0, 'ox': 0, 'ns': 3,
		'nt': 3001, 'dt': 0.001, 'ot': 0, 'nb':60, 'coef': 0.005, 'acqui_type': 1, 
		'inv': 0, 'waterz': -30, 'onlysrc': 0, 'onlyvel': 1, 'conv_error': 0.01, 'niter': 30}

## Modeling part
data,_,_,_,_=pfwi(vel,q,wavelet,src,data=None,mode=4,media=1,inv=0,verb=1,par=mypar);
np.save('datas',data);
# plt.subplot(1,3,1);
# plt.imshow(data[:,:,0],aspect='auto');
# plt.subplot(1,3,2);
# plt.imshow(data[:,:,1],aspect='auto');
# plt.subplot(1,3,3);
# plt.imshow(data[:,:,2],aspect='auto');
# plt.show()

## Source inversion part
# mypar['inv']=True;
# mypar['onlysrc']=True;
# mypar['niter']=30;
# data=np.load('datas.npy')
# data2,vinv,grad,src,mwt=pfwi(vel,q,wavelet,src=None,data=data,mode=4,media=1,inv=0,verb=1,par=mypar);
# plt.subplot(1,2,1);
# plt.plot(src[110,100,:,0]);
# plt.subplot(1,2,2);
# plt.plot(src[115,260,:,0]);
# plt.title('Niter=30')
# plt.show()

## velocity inversion part
mypar['inv']=True;
mypar['onlysrc']=False;
mypar['onlyvel']=True;
mypar['niter']=20;
# data=np.load('datas.npy')
print(vel_s.shape,q.shape,wavelet.shape,src.shape,data.shape)
data2,vinv,grad,src2,mwt=pfwi(vel_s,q,wavelet,src=src,data=data,mode=4,media=1,inv=0,verb=1,par=mypar);
print('Vinv.shape:',vinv.shape)

## visualization 
par=mypar;
# source=np.sum(src[:,:,:,0],2)*(-50)+np.sum(src[:,:,:,1],2)*(2.0)+np.sum(src[:,:,:,2],2)*(50)
source=np.sum(src[:,:,:,0],2)*(50)	#only showing one source
plt.figure(figsize=(8, 10))
plt.subplot(4,1,1);
plt.imshow(vel+source,cmap=plt.jet(),aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Ground truth"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,2);
plt.imshow(vel_s,aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Initial model"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,3);
plt.imshow(vinv[:,:,int(par['niter']/2)],aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%int(par['niter']/2));plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,1,4);
plt.imshow(vinv[:,:,par['niter']-1],aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%par['niter']); plt.ylabel("Depth (km)"); plt.xlabel("Lateral (km)"); 
plt.savefig(fname='test_pfwi_vel2d_vel.png',format='png',dpi=300)
plt.show()

## generate synthetic data
mypar['inv']=False;
mypar['onlysrc']=False;
mypar['onlyvel']=False;
data2,vinv2,grad2,src2,mwt2=pfwi(vel_s,q,wavelet,src=src,data=data,mode=4,media=1,inv=0,verb=1,par=mypar);
plt.subplot(1,3,1);
plt.imshow(data[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Observed data"); plt.ylabel("Time (s)"); plt.xlabel("Receiver (m)"); 
plt.subplot(1,3,2);
plt.imshow(data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Simulated data"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 
plt.subplot(1,3,3);
plt.imshow(data[:,:,0]-data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Difference"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 
plt.savefig(fname='test_pfwi_vel2d_data.png',format='png',dpi=300)
plt.show()


##A better visualization
plt.figure(figsize=(12, 8))
plt.subplot(1,6,1);
plt.imshow(data[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Observed data"); plt.ylabel("Time (s)"); plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,2);
plt.imshow(data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Simulated data"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 
plt.subplot(1,6,3);
plt.imshow(data[:,:,0]-data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Difference"); plt.gca().set_yticks([]); plt.xlabel("Receiver (km)"); 
# plt.savefig(fname='test_pfwi_vel2d_data.png',format='png',dpi=300)

plt.subplot(4,2,2);
plt.imshow(vel+source,cmap=plt.jet(),aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Ground truth"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,2,4);
plt.imshow(vel_s,aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
plt.title("Initial model"); plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,2,6);
plt.imshow(vinv[:,:,int(par['niter']/2)],aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%int(par['niter']/2));plt.ylabel("Depth (km)"); plt.gca().set_xticks([]);
plt.subplot(4,2,8);
plt.imshow(vinv[:,:,par['niter']-1],aspect='auto',clim=(1.5, 3.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]);
plt.title("%d Iterations"%par['niter']); plt.ylabel("Depth (km)"); plt.xlabel("Lateral (km)"); 
plt.savefig(fname='test_pfwi_vel2d_one.png',format='png',dpi=300)
plt.show()




