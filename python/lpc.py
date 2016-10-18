# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
from scipy import signal as sg
import math
import numpy as np

[fs, wavdata] = wavfile.read('clean.wav')
wavfile.write('cleanloud.wav', fs, wavdata*3)
#define filter order p
p = 5;

WINDOW_LEN_MS = 20
window_len = fs * (WINDOW_LEN_MS) // 1000

Lk = 100000
frame = wavdata[Lk:Lk+window_len]

# create autocorrelation array from r[-N/2] to r[N/2]. r[0] is in the center
autocorrelation = sg.correlate(frame, frame, "full");
mid = math.floor(len(autocorrelation)/2)

#transfrom to multidimensional array
autocorrelation = autocorrelation.reshape(-1,1)
# normalize by N
autocorrelation = autocorrelation / window_len

R = autocorrelation[mid:mid+p]
for i in range(1,p):
    R = np.hstack((R, autocorrelation[mid-i:mid-i+p]))

wavfile.write('cleanloud.wav', fs, frame)
#frame = np.reshape(frame, (-1,1))
print(R)
print("eigenvalues:")
print(np.linalg.eig(R))
