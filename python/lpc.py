# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt

def acf(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data) ** 2) 
    print("C0")
    print(c0)
    print(np.dot(frame, frame))

    def r(h):
        acf_lag = ((data[:n - h] ) * (data[h:] )).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs


[fs, wavdata] = wavfile.read('clean.wav')
wavfile.write('cleanloud.wav', fs, wavdata*3)
#define filter order p
p = 5;

WINDOW_LEN_MS = 1
N = fs * (WINDOW_LEN_MS) // 1000

Lk = 200000
frame = wavdata[Lk:Lk+N]
#####
## METHOD 0
#####
coeff = acf(frame)



######
## METHOD 1
######
acf = np.array([0]*p)
acf[0] = np.dot(frame, frame)
for k in range(1,p):
    acf[k] = (frame[k:] * frame[:-k]).sum() 
acf = acf / N

######
## METHOD 2
######
# create autocorrelation array from r[-N/2] to r[N/2]. r[0] is in the center
autocorrelation = np.correlate(frame, frame, "full");
mid = math.floor(len(autocorrelation)/2)

#transfrom to multidimensional array
autocorrelation = autocorrelation.reshape(-1,1)
# normalize by N
#autocorrelation = autocorrelation / window_len
plt.plot(autocorrelation)
#plt.show()
R = autocorrelation[mid:mid+p]
for i in range(1,p):
    R = np.hstack((R, autocorrelation[mid-i:mid-i+p]))

wavfile.write('cleanloud.wav', fs, frame)
#frame = np.reshape(frame, (-1,1))
#print(R)
#print("eigenvalues:")
#print(np.linalg.eig(R))

#compare methods 
print(autocorrelation[mid:mid+p])
print(acf)
print([x for x in coeff])
