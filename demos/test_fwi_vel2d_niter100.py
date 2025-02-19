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

## generate data
mypar={'nz':130, 'nx':334, 'dz': 0.008, 'dx': 0.012, 'oz': 0, 'ox': 0, 'ns': 20, 'ds': 0.2,
		'nt': 3001, 'dt': 0.001, 'ot': 0, 'nb':60, 'coef': 0.005, 'acqui_type': 1, 
		'inv': 0, 'waterz': -30, 'onlysrc': 0, 'onlyvel': 1, 'conv_error': 0.01, 'niter': 30}

## Modeling part
# data,vinv,grad,src,mwt=pfwi(vel,q,wavelet,src,data=None,mode=4,media=1,inv=0,verb=1,par=mypar);
# np.save('datas',data);
# plt.subplot(1,3,1);
# plt.imshow(data[:,:,0],aspect='auto');
# plt.subplot(1,3,2);
# plt.imshow(data[:,:,1],aspect='auto');
# plt.subplot(1,3,3);
# plt.imshow(data[:,:,2],aspect='auto');
# plt.show()

## Source inversion part
mypar['inv']=True;
mypar['onlysrc']=True;
mypar['niter']=100;
data,vinv,grad,src,mwt=pfwi(vel,q,wavelet,src=None,data=None,mode=1,media=1,inv=0,verb=1,par=mypar);
np.save('fwi-datas-%d.npy'%mypar['ns'],data)

data=np.load('fwi-datas-%d.npy'%mypar['ns'])
data2,vinv,grad,src,mwt=pfwi(vel_s,q,wavelet,src=None,data=data,mode=2,media=1,inv=0,verb=0,par=mypar);

# data=data2;
# plt.subplot(1,3,1);
# plt.imshow(data[:,:,0],aspect='auto');
# plt.subplot(1,3,2);
# plt.imshow(data[:,:,1],aspect='auto');
# plt.subplot(1,3,3);
# plt.imshow(data[:,:,2],aspect='auto');
# plt.show()
# 
par=mypar;
plt.figure(figsize=(8, 10))
plt.subplot(4,1,1);
plt.imshow(vel,cmap=plt.jet(),aspect='auto',clim=(1.5, 4.5), extent=[0,par['dx']*(par['nx']-1),par['dz']*(par['nz']-1),0]); 
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
plt.savefig(fname='test_fwi_vel2d_vel-%d.png'%mypar['niter'],format='png',dpi=300)
# plt.show()

np.save('fwi-vels-%d.npy'%mypar['niter'],vinv)

################################################################################################
### Synthetic again for data misfit comparison
################################################################################################
data2,vinv2,grad2,src2,mwt2=pfwi(vel_s,q,wavelet,src=src,data=data,mode=1,media=1,inv=0,verb=1,par=mypar);
np.save('fwi-data2-0.npy',data2)

data2,vinv2,grad2,src2,mwt2=pfwi(vinv[:,:,-1],q,wavelet,src=src,data=data,mode=1,media=1,inv=0,verb=1,par=mypar);
np.save('fwi-data2-%d.npy'%mypar['niter'],data2)

data2=np.load('fwi-data2-0.npy')
data3=np.load('fwi-data2-%d.npy'%mypar['niter'])
data=np.load('fwi-datas-%d.npy'%mypar['ns'])
par=mypar
plt.figure(figsize=(10, 10))
plt.subplot(2,3,1);
plt.imshow(data[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Observed data"); plt.ylabel("Time (s)"); plt.xlabel("Receiver (m)"); 
plt.subplot(2,3,2);
plt.imshow(data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Simulated data - Initial Model"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 
plt.subplot(2,3,3);
plt.imshow(data[:,:,0]-data2[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Difference - Initial Model"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 


plt.subplot(2,3,5);
plt.imshow(data3[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Simulated data - 100 iterations"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 
plt.subplot(2,3,6);
plt.imshow(data[:,:,0]-data3[:,:,0],aspect='auto',clim=(-0.5, 0.5),extent=[0,par['dx']*(par['nx']-1),par['dt']*(par['nt']-1),0]);
plt.title("Difference - 100 iterations"); plt.gca().set_yticks([]); plt.xlabel("Receiver (m)"); 

plt.savefig(fname='test_fwi_vel2d_datacomp.png',format='png',dpi=300)
plt.show()









