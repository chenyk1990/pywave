from pyntfa import ntfa1d
from pynpre import stft1d,st1d
#pip install git+https://github.com/chenyk1990/pyntfa
#pip install git+https://github.com/chenyk1990/pynpre
#There is a story why ST and STFT are written under the Pynpre package

import numpy as np
import matplotlib.pyplot as plt

# Download the data from https://github.com/aaspip/data/blob/main/cchirps.bin
fid=open("cchirps.bin","rb");
din = np.fromfile(fid, dtype = np.float32, count = 512).reshape([512,1],order='F')
# plt.plot(din)
# plt.show();

n1=din.shape[0]
dt=0.004;

## STFT
dout,w0,dw,nw=stft1d(din,dt=dt,inv=0,opt=1,sym=0,ntw=31,ot=0,wind=0,verb=1)
dout=dout.reshape([n1,nw,2],order='F');
print("dout.shape is ",dout.shape)
## Inverse transform
trace=stft1d(dout,dt=0.004,inv=1,opt=1,sym=0,ntw=31,ot=0,wind=0,verb=1)

## Visualization of STFT
t=np.linspace(0,(n1-1)*dt,n1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(3,2,1)
plt.plot(t,din,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Input');
plt.subplot(3,2,3)
plt.plot(t,trace,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Reconstruction');
plt.subplot(3,2,5)
plt.plot(t,din.flatten()-trace,'k',linewidth=1);plt.ylim(-3,3);plt.ylabel('Amplitude');plt.xlabel('Time (s)');plt.title('Reconstruction Error');

plt.subplot(1,2,2)#clim=(0, 10)
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum (STFT)')
plt.savefig('test_pyntfa_syn1d_stltft_stft.png',format='png',dpi=300)
plt.show();

## ST
dout,w0,dw,nw=st1d(din,dt=dt,inv=0,flo=0,fhi=0.5,verb=1)
dout=dout.reshape([n1,nw,2],order='F');
print("dout.shape is ",dout.shape)
## Inverse transform
trace=st1d(dout,dt=dt,inv=1,flo=0,fhi=0.5,verb=1)

## Visualization of ST
t=np.linspace(0,(n1-1)*dt,n1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(3,2,1)
plt.plot(t,din,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Input');
plt.subplot(3,2,3)
plt.plot(t,trace,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Reconstruction');
plt.subplot(3,2,5)
plt.plot(t,din.flatten()-trace,'k',linewidth=1);plt.ylim(-3,3);plt.ylabel('Amplitude');plt.xlabel('Time (s)');plt.title('Reconstruction Error');

plt.subplot(1,2,2)#clim=(0, 10)
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum (ST)')
plt.savefig('test_pyntfa_syn1d_stltft_st.png',format='png',dpi=300)
plt.show();


## NTFA
##### Below is for non-stationary data regularizaton
dout,w0,dw,nw = ntfa1d(din,dt=dt,niter=10,rect=7,ifb=0,inv=0,ifn=0)
dout=dout.reshape([n1,nw,2],order='F');

print('First version:',dout[:,:,0].max(),dout[:,:,0].min()) ##should be [0.0864,-0.0876]
## NOTE: dout[:,:,0] is imaginary and dout[:,:,1] is real

## Visualization of NTFA
t=np.linspace(0,(n1-1)*dt,n1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(3,2,1)
plt.plot(t,din,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Input');
plt.subplot(3,2,3)
plt.plot(t,trace,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Reconstruction');
plt.subplot(3,2,5)
plt.plot(t,din.flatten()-trace,'k',linewidth=1);plt.ylim(-3,3);plt.ylabel('Amplitude');plt.xlabel('Time (s)');plt.title('Reconstruction Error');

plt.subplot(1,2,2)#clim=(0, 10)
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum (NTFA)')
plt.savefig('test_pyntfa_syn1d_stltft_ntfa.png',format='png',dpi=300)
plt.show();


##### Below is for non-stationary data+model regularizaton
# create radius
r_ref=7 #reference radius
r_max=14 #reference radius

tmp=dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1]
tmp=tmp/tmp.max()
tmp[tmp<0.05]=0
tmp[tmp>=0.05]=1
rectn0=tmp*(r_ref-r_max)+r_max     #first axis radius     (time)
rectn1=tmp*(-2)+3                #second axis radius    (frequency)

fig = plt.figure(figsize=(16, 8))
plt.subplot(1,2,1)
plt.imshow(rectn0,aspect='auto');
plt.colorbar(orientation='horizontal',shrink=0.6,label='Traveltime (s)');
plt.title('First Axis (Time) smoothing');

plt.subplot(1,2,2)
plt.imshow(rectn1,aspect='auto');
plt.colorbar(orientation='horizontal',shrink=0.6,label='Traveltime (s)');
plt.title('Second Axis (Frequency) smoothing');
plt.show()

n1=din.shape[0]
doutn,w0,dw,nw = ntfa1d(din,dt=dt,niter=10,rect=7,ifb=0,inv=0,ifn=1,rect1=rectn0,rect2=rectn1)
doutn=doutn.reshape([n1,nw,2],order='F');
print('Second version:',doutn[:,:,0].max(),doutn[:,:,0].min()) ##should be [0.08177045 -0.081738934]
## Reconstruct trace
tracen=ntfa1d(doutn,dt=1,niter=10,rect=7,ifb=0,inv=1,ifn=1,rect1=rectn0,rect2=rectn1);

## Visualization of NTFA2
t=np.linspace(0,(n1-1)*dt,n1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(3,2,1)
plt.plot(t,din,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Input');
plt.subplot(3,2,3)
plt.plot(t,tracen,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Reconstruction');
plt.subplot(3,2,5)
plt.plot(t,din.flatten()-tracen,'k',linewidth=1);plt.ylim(-3,3);plt.ylabel('Amplitude');plt.xlabel('Time (s)');plt.title('Reconstruction Error');

plt.subplot(1,2,2)#clim=(0, 10)
plt.imshow(doutn[:,:,0]*doutn[:,:,0]+doutn[:,:,1]*doutn[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum (NTFA)')
plt.savefig('test_pyntfa_syn1d_stltft_ntfa2.png',format='png',dpi=300)
plt.show();


