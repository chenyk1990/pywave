from ftfacfun import *

def st1d(din,dt=0.004,inv=0,flo=0,fhi=0.5,verb=1):
    """
    st1d: S transform for 1D signals
    
    INPUT
    din: input trace (1D, n1/nt)
    dt: time interval
    inv: flag of inverse transform (0: forward; 1: inverse; default: 0)
    flo: default (0)
    fhi: default (0.5 - Nyquist)
    verb: verbosity
    
    OUTPUT
    dout: [forward] output Time-frequency spectrum (first axis: Time; second axis: Frequency; third axis: real/imag)
    NOTE: 1) dout is of size n1*nw*2 (e.g., dout.reshape([n1,nw,2],order='F'))
          2) The first component in 3rd axis is Real; and the second component is Imaginary (Remember it when constructing a complex number)
          3) The third axis real/imag sequence of stft1d is opposite to the case in ntfa1d (pyntfa)
          w0,dw,nw: intuitive frequency axis
          [inverse] reconstructed trace
    
    EXAMPLE
    demos/test_pynpre_syn1d_stltft.py
    
    HISTORY
    Original version by Yangkang Chen, Oct 30, 2024
    
    """
    import numpy as np
    
    din=np.float32(din)
    n1=din.shape[0];
    
    print(n1)
    
    dtmp=st1dc(din.flatten(order='F'),n1,verb,inv,flo,fhi,dt)
    
    if inv==0:
        nw=np.int32((dtmp.size-3)/n1/2);
        dout=dtmp[0:n1*nw*2]
        w0=dtmp[n1*nw*2]    #different from PYntfa
        dw=dtmp[n1*nw*2+1]  #different from PYntfa
        nw2=np.int32(dtmp[n1*nw*2+2])
        print(dtmp.size,w0,dw,nw2)
        if nw2 != nw:
            print('nw=',nw,'nw2=',nw2,'dimension discrepancy')
        else:
            print('dimension consistent')
        return dout,w0,dw,nw
    else:
        return dtmp
    
    

def stft1d(din,dt=0.004,inv=0,opt=1,sym=0,ntw=7,ot=0,wind=0,verb=1):
    """
    stft1d: Short-Time Fourier transform for 1D signals
    
    INPUT
    din: input trace (1D, n1/nt)
    dt: time interval
    inv: flag of inverse transform (0: forward; 1: inverse; default: 0)
    opt: optimal FFT size
    sym: apply symmetric Fourier transform
    ntw: size of short-time window
    ot: starting time
    wind: if applying Hanning window
    verb: verbosity
    
    OUTPUT
    dout: [forward] output Time-frequency spectrum (first axis: Time; second axis: Frequency; third axis: real/imag)
    NOTE: 1) dout is of size n1*nw*2 (e.g., dout.reshape([n1,nw,2],order='F'))
          2) The first component in 3rd axis is Real; and the second component is Imaginary (Remember it when constructing a complex number)
          3) The third axis real/imag sequence of stft1d is opposite to the case in ntfa1d (pyntfa)
          w0,dw,nw: intuitive frequency axis
          [inverse] reconstructed trace
    
    EXAMPLE
    demos/test_pynpre_syn1d_stltft.py
    
    HISTORY
    Original version by Yangkang Chen, Oct 30, 2024
    
    """
    import numpy as np

    n1=din.shape[0];
    din=np.float32(din)

    
    print(n1)
    
    dtmp=stft1dc(din.flatten(order='F'),n1,verb,wind,inv,sym,opt,ntw,dt,ot)
    
    
    if inv==0:
        nw=np.int32((dtmp.size-3)/n1/2);
        dout=dtmp[0:n1*nw*2]
        w0=dtmp[n1*nw*2]    #different from PYntfa
        dw=dtmp[n1*nw*2+1]  #different from PYntfa
        nw2=np.int32(dtmp[n1*nw*2+2])
        print(dtmp.size,w0,dw,nw2)
        if nw2 != nw:
            print('nw=',nw,'nw2=',nw2,'dimension discrepancy')
        else:
            print('dimension consistent')
        return dout,w0,dw,nw
    else:
        return dtmp
    


