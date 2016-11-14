# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:03:54 2016

@author: Rik van der Vlist
"""
import numpy as np
from scipy.io import wavfile
import scipy
import math
debug = []
debugc = []

axreal = []
axideal = []

def findfilter(frame):
    # initialize the autocorrelation function with zeros
    acf = np.array([0]*(p+1))
    # calculate first term by x*x_H (sum of elementwise squares)
    # divide by N, a biased estimate is obtained
    acf[0] = np.dot(frame, frame) / N
    for k in range(1,p+1):
        # create k_th term by creating to vectors with offset k and length N-k
        acf[k] = int((frame[k:] * frame[:-k]).sum() / N)
    # make the autocorrelation symmetric from -p to p and convert from N array to Nx1 array
    # this is the estimated autocorrelation [-rx(p) ... rx(0) ... rx(p)]
    acf = np.concatenate([acf[p:0:-1], acf])
    acf = acf.reshape(-1,1)
    # Scipy has a built-in solve_toeplitz function which uses Levinson-Durbin recursion
    # which is faster than numpy.linalg.solve (which uses Gaußian elimination)
    # R is upper row from Toeplitx Rx, [rx(0) ... rx(p-1)]
    R = acf[-p-1:-1]
    # r is right hand side of Rx*a = -r, [rx(1) ... rx(p)].T
    r = acf[-p:].reshape(-1)
    try:
        a = scipy.linalg.solve_toeplitz(R, -r)
        a = np.hstack((np.array([1]), a))
    except np.linalg.linalg.LinAlgError:
        a = np.hstack((np.array([1]), np.zeros(p)))
    # when we take the inverse of the filter H^-1, (AR->MA) we find the error sequence
    # beacuse the inverse is a MA filter we can find the error sequence by a simple convolution
    e = np.convolve(frame, a, 'same');
    # the variance of the error can be used for gerneation of the noise source for decoding
    g = np.var(e);
    return [a, e, g]

def wsola_analysis(y,fs,alpha,nleng,nshift,deltamax):
    i = 0

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
   
# create search region based on deltamax parameter
    deltas=round(deltamax*fs/1000)
# define the windowing functions
    win = np.hanning(nleng)

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
# increase cursor by step size multiplied with scale factor
        nalpha=nalpha+round(alpha*nshift)
        indexl=max(nalpha-deltas,0)
# obtain values from desired range extended by search area at beginning and end
        xreal=y[indexl:nalpha+nleng+deltas]
# move the output cursor
        nlin=nlin+nshift
        
# apply the window and find the error domain signal for both xreal and xideal
        [a1, error_xreal, g] = findfilter(xreal*np.hanning(len(xreal)))
        axreal.append(a1)
        [a2, error_xideal, g] = findfilter(xideal*win)
        axideal.append(a2)
# correlate the error sequences 
        c=np.correlate(error_xreal,error_xideal, 'full')
#        c = np.concatenate((np.zeros(len(xreal)-len(xideal)), c))
         
# find the maximum value of the correlation between the maximum search bound
        middle = len(c) // 2
        cs = c[middle-deltas:middle+deltas]
# it is obvious that we should add 3 here
        maxind= int(np.argmax(cs))
        debug.append(maxind)
        debugc.append(cs)

# overlap add the best matchnshift
        xadd=y[nalpha-deltas+maxind:nalpha-deltas+maxind+nleng]*win
        yout[nlin:nlin+nleng]=yout[nlin:nlin+nleng]+xadd              
# update the cursor for the next frame to realized starting point + step size 
        nideal=nalpha-deltas+maxind+nshift
        i = i + 1
    
# play out time-altered sound
    youtn=yout/float(max(max(yout),-min(yout)))
    # sound(youtn,fs)
    
# save output file
    youts=youtn*32700
    print(i)
    return [youts, youtn, yout]

scale_factor=2.0
filename_in='male.wav'

[fs, wavdata] = wavfile.read(filename_in)
test = wavdata
# Now we convert from int16 to int32 to avoid compatibility issues in Python
# and to ensure compatibility with files that are formatted differently
wavdata = np.ndarray.astype(wavdata, 'double')
#wavdata = wavdata / (max(max(wavdata), -min(wavdata)))

#define filter order p
p = 8;
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
#yout *= 32700
yout = np.ndarray.astype(yout, 'int16')
wavfile.write('wsola.wav', fs, yout)






