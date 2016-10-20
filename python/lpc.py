# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
import numpy as np
import math

[fs, wavdata] = wavfile.read('clean.wav')
# Now we convert from int16 to int32 to avoid compatibility issues in Python
# and to ensure compatibility with files that are formatted differently
wavdata = np.ndarray.astype(wavdata, 'int32')

#define filter order p
p = 12;
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

# define the windowing function
window = np.hanning(N)

def findfilter(frame):
    # initialize the autocorrelation function with zeros
    acf = np.array([0]*(p+1))
    # calculate first term by x*x_H (sum of elementwise squares)
    # divide by N, a biased estimate is obtained
    acf[0] = np.dot(frame, frame) / N
    for k in range(1,p+1):
        # create k_th term by creating to vectors with offset k and length N-k
        acf[k] = (frame[k:] * frame[:-k]).sum() / N
    # make the autocorrelation symmetric from -p to p and convert from N array to Nx1 array
    acf = np.concatenate([acf[p:0:-1], acf])
    acf = acf.reshape(-1,1)
    # compose autocorrelation matrix from the autocorrelation sequence
    # a shifted version of the function is added to the matrix column per column
    R = acf[-p-1:-1]
    for i in range(1,p):
        R = np.hstack((R, acf[-p-i-1:-i-1]))
    # create array with r[1] - r[p]    
    r = acf[-p:].reshape(-1)
    try:
        a = np.linalg.solve(R, r)
    except np.linalg.linalg.LinAlgError:
        a = np.zeros(p)
    return a

framedata = []
for i in range(nFrames):
    frame = wavdata[i*step:i*step+N]
    # window the frame
    frame = frame * window
    # add frame to list of frames
    framedata.append(frame)
#TODO: we will be losing some samples in the end because the length of data is
#       an integer multiple of the frame length 

# next step is estimating the autocorrelation for each frame
filtercoeffs = []
for frame in framedata:
    filtercoeffs.append(findfilter(frame))
    
# synthesize new frames by 
synthframes = []
for coeffs in filtercoeffs:
    #generate a noise sequence
    noise = np.random.randint(-32767, 32767, size=N)
    synthframe = np.convolve(noise, coeffs, 'same')
    synthframes.append(synthframe)

#sum all synthesized frames back together
wavdata_hat = np.zeros(len(wavdata))
for i, frame in enumerate(synthframes):
    wavdata_hat[i*step:i*step+N] += frame

#print('Autocorrelation function from -p to p')
#print(acf)
#print('Autocorrelation matrix')
#print(R);
#[eig, eigvect] = np.linalg.eig(R)
#print('Eigenvalues autocorrelation matrix')
#print(eig)
#print('A coefficients');
#print(a)

wavdata_hat = np.ndarray.astype(wavdata_hat, 'int16')
wavfile.write('synthesized.wav', fs, wavdata_hat)


