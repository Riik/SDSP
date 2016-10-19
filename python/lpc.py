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

WINDOW_LEN_MS = 20
N = fs * (WINDOW_LEN_MS) // 1000

Lk = 200000
frame = wavdata[Lk:Lk+N]
frame = np.ndarray.astype(frame, 'int32')
######
## Estimate autocorrelation
######
acf = np.array([0]*p)
# calculate first term by sum of x*x_H
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
R = acf[-p:]
for i in range(1,p):
    R = np.hstack((R, acf[-p-i:-i]))

#compare methods 
print('Autocorrelation function from -p to p')
print(acf)
print('Autocorrelation matrix')
print(R);
[eig, eigvect] = np.linalg.eig(R)
print('Eigenvalues autocorrelation matrix')
print(eig)

wavfile.write('cleanloud.wav', fs, frame)


