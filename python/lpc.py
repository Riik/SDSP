# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist

Script to obtain the LPC (linear prediction) coefficients from an audio file. 
"""
from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy
import wsola

scale_factor=2
filename_in='clean.wav'

[fs, wavdata] = wavfile.read(filename_in)
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

# given a frame of data, this function returns the LPC coefficients, error sequence and error variance 
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

framedata = []
for i in range(int(nFrames)):
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

    # as a reference, also summ error sequences together 
wavdata_noise = np.zeros(len(wavdata))
for i, frame in enumerate(errors):
    wavdata_noise[i*step:i*step+N] += frame

noise=wsola.wsola(wavdata_noise, fs, scale_factor, 'xcorr')
noise_framedata=[]
for i in range(int(nFrames)):
    noise_frame = noise[i*step:i*step+N*scale_factor]
    # window the frame
    window = np.hanning(N*scale_factor)
    noise_frame = noise_frame * window 
    # add frame to list of frames
    noise_framedata.append(noise_frame)

# From the coeficcients we can generate synthesized speech using a noise source 
# or using the original signal
synthframes = []
for i, coeffs in enumerate(filtercoeffs):
    #generate a noise sequence
    
    noise = np.random.randn(N)*np.sqrt(gains[i])
    noise = noise_framedata[i]
    
    # alternatively, the error sequence can be used to obtain the original signal. 
    # this is what we might use for the WSOLA extension 
    #noise = errors[i]
    synthframe = sg.lfilter(np.array([1]), coeffs,  noise)
    synthframes.append(synthframe)

#sum all synthesized frames back together
wavdata_hat = np.zeros(len(wavdata)*scale_factor)
for i, frame in enumerate(synthframes):
    wavdata_hat[i*N:i*N+N*scale_factor] += frame

# make some plots 
#plt.plot(wavdata_hat)
#plt.show()
#plt.plot(wavdata_noise)
#plt.show()


#scaled=np.int16(scaled/max(abs(scaled)) * 32767)
#filename_out = '%s withScale %.2f.wav' %(filename_in,scale_factor)
#scipy.io.wavfile.write(filename_out,new_fs,scaled)
# write back to files, convert back to int16s
wavdata_hat = np.ndarray.astype(wavdata_hat, 'int16')
wavfile.write('synthesized.wav', new_fs, wavdata_hat)
wavdata_noise = np.ndarray.astype(wavdata_noise, 'int16')
wavfile.write('noisefromfilter.wav', new_fs, wavdata_noise)


