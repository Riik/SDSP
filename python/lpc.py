# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:01:14 2016

@author: Rik van der Vlist
"""
from scipy.io import wavfile
import numpy as np

[fs, wavdata] = wavfile.read('clean.wav')
wavfile.write('cleanloud.wav', fs, wavdata*3)

WINDOW_LEN_MS = 20
window_len = fs * (WINDOW_LEN_MS) // 1000

Lk = 100000
frame = wavdata[Lk:Lk+window_len]
#frame = np.reshape(frame, (-1,1))
print(frame)
