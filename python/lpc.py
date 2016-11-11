# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sg

[fs, wavdata] = wavfile.read('clean.wav')
# Now we convert from int16 to int32 to avoid compatibility issues in Python
# and to ensure compatibility with files that are formatted differently
wavdata = np.ndarray.astype(wavdata, 'int32')
plt.plot(wavdata)
plt.show()
#define filter order p
p = 10;
# window length in ms
WINDOW_LEN_MS = 20
# overlap percentage
overlap = 0.5
new_fs = 8000
new_fs = fs

# calculate frame length in samples from length in ms
N = fs * (WINDOW_LEN_MS) // 1000
# calculate step size for the frames
step = math.floor(N*(1-overlap))
# calculate number of needed frames
nFrames = (len(wavdata)-N) // step
#wavdata = sg.resample(wavdata, len(wavdata)*fs/new_fs)

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
        acf[k] = int((frame[k:] * frame[:-k]).sum() / N)
    # make the autocorrelation symmetric from -p to p and convert from N array to Nx1 array
    # this is the estimated autocorrelation [-rx(p) ... rx(0) ... rx(p)]
    acf = np.concatenate([acf[p:0:-1], acf])
    acf = acf.reshape(-1,1)
    # compose autocorrelation matrix from the autocorrelation sequence
    # a shifted version of the function is added to the matrix column per column
    R = acf[-p-1:-1]
    for i in range(1,p):
        R = np.hstack((R, acf[-p-i-1:-i-1]))
    # create array with [r(1) ... r(p)] 
    r = acf[-p:].reshape(-1)
    # solve Rx*a = -r
    # try/catch block is needed because signal can become zero and makes Rx not full rank
    try:
        a = np.linalg.solve(R, -r)
        # add 1 as first filter coefficient
        a = np.hstack((np.array([1]), a))
    except np.linalg.linalg.LinAlgError:
        a = np.hstack((np.array([1]), np.zeros(p)))
#    if(max(a) > 50):
#        print(a)
#        print(acf)
#        print(R)
    # when we take the inverse of the filter H^-1, (AR->MA) we find the error sequence
    # beacuse the inverse is a MA filter we can find the error sequence by a simple convolution
    e = np.convolve(frame, a, 'same');
    # the variance of the error can be used for gerneation of the noise source for decoding
    g = np.var(e);
    return [a, e, g]

framedata = []
for i in range(nFrames):
    frame = wavdata[i*step:i*step+N]
    # window the frame
    frame = frame * window
    # add frame to list of frames
    framedata.append(frame)
#TODO: we will be losing some samples in the end because the length of data is
#       an integer multiple of the frame length 

filtercoeffs = []
errors = []
gains = []
# we start by estimating the filter coefficients for each frame
# this is the encoding part
for frame in framedata:
    [coeff, error, gain] = findfilter(frame)
    filtercoeffs.append(coeff)
    errors.append(error)
    gains.append(gain)

# From the coeficcients we can generate synthesized speech using a noise source 
# or using the original signal
synthframes = []
for i, coeffs in enumerate(filtercoeffs):
    #generate a noise sequence
    noise = np.random.randn(N)*np.sqrt(gains[i])
    # alternatively, the error sequence can be used to obtain the original signal
    #noise = errors[i]
    synthframe = sg.lfilter(np.array([1]), coeffs, noise)
    synthframes.append(synthframe)

#sum all synthesized frames back together
wavdata_hat = np.zeros(len(wavdata))
for i, frame in enumerate(synthframes):
    wavdata_hat[i*step:i*step+N] += frame
#sum all error sequences back together
wavdata_noise = np.zeros(len(wavdata))
for i, frame in enumerate(errors):
    wavdata_noise[i*step:i*step+N] += frame

# make some plots 
plt.plot(wavdata_hat)
plt.show()
plt.plot(wavdata_noise)
plt.show()

# write back to files
wavdata_hat = np.ndarray.astype(wavdata_hat, 'int16')
wavfile.write('synthesized.wav', new_fs, wavdata_hat)
wavdata_noise = np.ndarray.astype(wavdata_noise, 'int16')
wavfile.write('noisefromfilter.wav', new_fs, wavdata_noise)


