## This DEMO is a 3D example [x,y,z] with constant velocity and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2024, The University of Texas at Austin

from pywave import aps3d
import numpy as np

vel=np.ones([101*101*101,1],dtype='float32');
data=aps3d(vel,nt=1501,dt=0.001,ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101]);


# time=t.reshape(101,101,101,order='F'); #[x,y,z]

# velx=3.80395*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
# # velx=3.09354*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
# 
# eta=0.340859*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
# eta=0.340859*np.ones([101*101*101,1],dtype='float32');
# t=fmm.eikonalvti(velx,vel,eta,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=1);
# time2=t.reshape(101,101,101,order='F');#first axis (vertical) is x, second is z


## Verify
print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
print(['Correct result:',0.4845428, 0.0, 0.08635751, 0.00745762])




