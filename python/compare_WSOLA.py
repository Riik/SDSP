# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:45:07 2016

@author: Rik van der Vlist
"""

from scipy.io import wavfile
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np


filename_ref='clean.wav'
filename_wsola_lpc='wsola speed1 errordomain.wav'
filename_wsola='wsola speed1 WSOLA.wav'

[fs_r, reference] = wavfile.read(filename_ref)
#reference = np.ndarray.astype(reference, 'double')
[fs_l, wav_wsola] = wavfile.read(filename_wsola)
#wav_wsola = np.ndarray.astype(wav_wsola, 'double')
[fs_w, wav_lpc] = wavfile.read(filename_wsola_lpc)
#wav_lpc = np.ndarray.astype(wav_lpc, 'double')

print(fs_r)
print(fs_l)
print(fs_w)
print(reference.dtype)
print(wav_wsola.dtype)
print(wav_lpc.dtype)

wav_wsola = wav_wsola[:len(reference)]
wav_lpc = wav_lpc[:len(reference)]

wav_wsola_error = reference - wav_wsola
wav_lpc_error = reference - wav_lpc

error_wsola = sum(wav_wsola_error*wav_wsola_error)
error_lpc = sum(wav_lpc_error*wav_lpc_error)
print("MSE wsola and reference at 1x speed")
print(error_wsola)
print("MSE wsola_lpc and reference at 1x speed")
print(error_lpc)

noerror = reference - reference
noerror = sum(noerror*noerror)
print(noerror)

f, t, Sxx_wsola= sg.spectrogram(wav_wsola, fs=16000)
f, t, Sxx_lpc = sg.spectrogram(wav_lpc, fs=16000)
f, t, Sxx_reference = sg.spectrogram(reference, fs=16000)

error_2d_wsola = Sxx_wsola - Sxx_reference
MSE_2D_err_wsola = sum(sum(error_2d_wsola*error_2d_wsola))
error_2d_lpc = Sxx_lpc - Sxx_reference
MSE_2D_err_lpc = sum(sum(error_2d_lpc*error_2d_lpc))
print("2D MSE error WSOLA")
print(MSE_2D_err_wsola)
print("2D MSE error LPC")
print(MSE_2D_err_lpc)