# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:40:15 2016

@author: Mert
"""

from __future__ import division
from math import ceil,floor
from numpy import hanning, correlate,divide,array,arange,zeros,multiply,argmax,spacing,ndarray,rint
import scipy.io.wavfile


# % sound(scaled(1:length(scaled)/10),fs);

def wsola(sig_in, fs, scale_factor, simil_method):
#    nargin=wsola.func_code.co_argcount
#    if nargin<4:
#        simil_method = 'xcorr'
#    sf_split_margin = 3
    win_time = 0.020 #seconds
    overlap_ratio = 0.5
    sig_in=array(sig_in, dtype=float)
    max_err = min(0.005, divide(win_time*overlap_ratio,scale_factor/2))
    win_len=ceil(win_time*fs) #320
    max_err_length=ceil(max_err*fs) #around 80
    step_len=floor(overlap_ratio*win_len)
    
    win = hanning(win_len)
    new_scale = arange(1,round(len(sig_in)*scale_factor*2))
    sig_out = zeros(len(new_scale))
    win_out = zeros(len(new_scale))
    
    cursor_in = 1
    cursor_out = 1
    while cursor_in < (len(sig_in)-win_len-max(step_len,(step_len/scale_factor))-2*max_err_length) \
            and cursor_out<(len(sig_out)-win_len):
#         input segments
        new_seg=multiply(sig_in[cursor_in:cursor_in+win_len],win)
#        print(new_seg)
        new_seg_neighbour = multiply(sig_in[(cursor_in+step_len):(cursor_in+step_len+win_len)],win)
#        print(new_seg_neighbour)
#        overlapp add
        sig_out[cursor_out:(cursor_out+win_len)]+=new_seg
        #overlap add window normalization vec
        win_out[cursor_out:(cursor_out+win_len)]+=win
        cursor_out += step_len
        cursor_in +=  round(step_len/scale_factor)      
        new_seg_cand = multiply(sig_in[cursor_in:cursor_in+win_len],win)
        shift = max_xcorr_similarity(new_seg_neighbour, new_seg_cand, max_err_length)
        cursor_in -= shift;
    print(sig_out)
    sig_out=sig_out[1:cursor_out]
    win_out=win_out[1:cursor_out]
    return divide(sig_out,(win_out+spacing(1)),sig_out); #%normalize to remove possible modulations

def max_xcorr_similarity(seg1, seg2, max_lag):
    corrArray=correlate(seg1, seg2, 'full')
    middle=max(len(seg1),len(seg2))
    corrArray=corrArray[middle-max_lag:middle+max_lag]
    max_i = argmax(corrArray)
    return max_i - max_lag        
filename_in='clean.wav'
rate, data = scipy.io.wavfile.read(filename_in)
scale_factor=0.5
#data=data[8000:-1]
scaled=wsola(data, rate, scale_factor, 'xcorr')
#scaled=rint(scaled)
#scaled=scaled.astype(int)
filename_out = '%s withScale %.2f.wav' %(filename_in,scale_factor)
scipy.io.wavfile.write(filename_out,rate,scaled)