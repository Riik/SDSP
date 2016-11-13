# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:03:54 2016

@author: Rik van der Vlist
"""
import numpy as np
from scipy.io import wavfile
import math

def wsola_analysis(y,fs,alpha,nleng,nshift,deltamax):
# wsola analysis of speech file
#
# Inputs:
#   y=input speech (normalized to 32767)
#   fs=input speech sampling rate
#   alpha=speed up/slow down factor (0.5 <= alpha <= 3.0)
#   nleng=len of analysis frame in samples
#   nshift=shift of analysis frame in samples
#   wtype=window type (1=Hamming, 0=rectangular, 2=triangular)
#   deltamax=maximum number of samples to search for best alignment
#   ipause=plotting option for debug
#
# Outputs:
#   youts=time scaled signal, scaled to 32767
#   youtn=time scaled signal, scaled to 1

# define the windowing function
    win = np.hanning(nleng)
    
# create search region based on deltamax parameter
    deltas=round(deltamax*fs/1000)
    
    print('nleng:{}, nshift:{}, deltas:{}, fs:{} \n'.format(nleng,nshift,deltas,fs))
    
# initialize overlap add with first frame
    nideal=0+nshift
    nalpha=0
    nlin=0
    nsamp=len(y)
    yout=np.zeros((int(nsamp/alpha+nleng+0.5)))
    yout[:nleng]=y[:nleng]*win
    
# full wsola processing
    while (nideal+nleng <= nsamp and nalpha+nleng+deltas+alpha*nshift <= nsamp):
        xideal=y[nideal:nideal+nleng]
        nalpha=nalpha+round(alpha*nshift)
        indexl=max(nalpha-deltas,1)
        xreal=y[indexl:nalpha+nleng+deltas]
        nlin=nlin+nshift
        
# correlate pair of frames and find optimal matching point
        c=np.correlate(xreal,xideal, 'full')
         
# compensate for xcorr shift by len of xreal find peak of len compensated cs        
        lxreal=len(xreal)
        cs = c[lxreal:lxreal+2*deltas]
        try:
            maxind= np.argmax(cs)
        except:
            maxind=1        
# overlap add the best matchnshift
        xadd=y[nalpha-deltas+maxind:nalpha-deltas+maxind+nleng]*win
        yout[nlin:nlin+nleng]=yout[nlin:nlin+nleng]+xadd              
# update indices
        nideal=nalpha-deltas+maxind+nshift
    
# play out time-altered sound
    youtn=yout/float(max(max(yout),-min(yout)))
    # sound(youtn,fs)
    
# save output file
    youts=youtn*32700
    return [youts, youtn, yout]

scale_factor=0.5
filename_in='clean.wav'

[fs, wavdata] = wavfile.read(filename_in)
# Now we convert from int16 to int32 to avoid compatibility issues in Python
# and to ensure compatibility with files that are formatted differently
wavdata = np.ndarray.astype(wavdata, 'int32')

#define filter order p
p = 10;
# window length in ms
WINDOW_LEN_MS = 20
# overlap percentage
overlap = 0.5
# calculate frame length in samples from length in ms
N = fs * (WINDOW_LEN_MS) // 1000
# calculate step size for the frames
step = math.floor(N*(1-overlap))
# calculate number of needed frames
nFrames = (len(wavdata)-N) // step
#wavdata = sg.resample(wavdata, len(wavdata)*fs/new_fs)
deltamax_ms = 5 

[youts, youtn, yout] = wsola_analysis(wavdata,fs,scale_factor,N,step,deltamax_ms)
yout = np.ndarray.astype(yout, 'int16')
wavfile.write('wsola.wav', fs, yout)



