# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
import numpy as np

[fs, wavdata] = wavfile.read('clean.wav')

#define filter order p
p = 5;

# calculate frame length in samples from length in ms
WINDOW_LEN_MS = 20
N = fs * (WINDOW_LEN_MS) // 1000

# frame offset
Lk = 200000
# get a frame of length N with starting point Lk from data 
frame = wavdata[Lk:Lk+N]
# At this part some windowing is probably needed
# Now we convert from int16 to int32 to avoid compatibility issues in Python
frame = np.ndarray.astype(frame, 'int32')
# next step is estimating the autocorrelation
# initialize the autocorrelation function with zeros
acf = np.array([0]*p)
# calculate first term by x*x_H (sum of elementwise squares)
acf[0] = np.dot(frame, frame)
for k in range(1,p):
    # create k_th term by creating to vectors with offset k and length N-k
    acf[k] = (frame[k:] * frame[:-k]).sum() 
# divide by N, a biased estimate is obtained
acf = acf / N
# make the autocorrelation symmetric from -p to p and convert from N array to Nx1 array
acf = np.concatenate([acf[p:0:-1], acf])
acf = acf.reshape(-1,1)
# compose autocorrelation matrix from the autocorrelation sequence
# a shifted version of the function is added to the matrix column per column
R = acf[-p:]
for i in range(1,p):
    R = np.hstack((R, acf[-p-i:-i]))

print('Autocorrelation function from -p to p')
print(acf)
print('Autocorrelation matrix')
print(R);
[eig, eigvect] = np.linalg.eig(R)
print('Eigenvalues autocorrelation matrix')
print(eig)

#wavfile.write('cleanloud.wav', fs, frame)


