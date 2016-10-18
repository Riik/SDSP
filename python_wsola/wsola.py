from math import *

from numpy import hanning, correlate
from numpy import *
import scipy.signal

def wsola(sig_in, fs, scale_factor, simil_method):
#  sig_in - input signal
#  fs - sampling frequency
#  scale_factor - by what ratio to strech/compress the audio.
#  simil_method - optional parameter either 'xcorr' or 'amdf' (default)

    nargin=wsola.func_code.co_argcount
#    nargin=len(sys.argv)
    if nargin<4:
        simil_method = 'xcorr'
    print(sig_in)
    sf_split_margin = 3
    orig_len = len(sig_in)

    if (scale_factor > sf_split_margin) | (scale_factor < 1/sf_split_margin):
        # recursively treat scale factors too big or too small
        num_recur = ceil(divide(abs((log(scale_factor),log(sf_split_margin)))))

        scale_factor_recur = power(scale_factor , ((num_recur-1)/num_recur))

        sig_in = wsola(sig_in, fs, scale_factor_recur, simil_method)

        scale_factor = divide(scale_factor(divide(len(sig_in,orig_len))))
#    print(sig_in)
    # constants
    win_time = 0.020 #seconds
    overlap_ratio = 0.5
    max_err = min(0.005, divide(win_time*overlap_ratio,scale_factor/2))

    # lengths
    win_len = ceil(multiply(win_time,fs))
    max_err_len = ceil(multiply(max_err,fs))
    step_len = floor(multiply(overlap_ratio,win_len))

    #  vectors
    win = hanning(win_len)
    
    # % orig_scale = 1:length(sig_in)
    new_scale = arange(round((multiply(len(sig_in),scale_factor*2))))
    sig_out = zeros(len(new_scale))
    win_out = zeros(len(new_scale))
#    print(sig_in)
    cursor_in = int(1)
    cursor_out = int(1)
    print(cursor_in<(len(sig_in)-win_len-max(step_len, divide(step_len,scale_factor))-2*max_err_len))
    print(cursor_out<(len(sig_out)-win_len))
    while (cursor_in<(len(sig_in)-win_len-max(step_len, divide(step_len,scale_factor))-2*max_err_len))\
 and (cursor_out<(len(sig_out)-win_len)):
        # % input segments
        
        new_seg = multiply(sig_in[cursor_in:(cursor_in+win_len)],win)
#        print(sig_in)
        new_seg_neighbour = multiply(sig_in[(cursor_in+step_len):(cursor_in+step_len+win_len)],win)
        
#        print (len(new_seg))
#        print(len(sig_out[cursor_out:(cursor_out+win_len)]))
        # % overlap add
#        print(new_seg)
        print(cursor_out)
#        print(sig_out[cursor_out:(cursor_out+win_len)])
        sig_out[cursor_out:(cursor_out+win_len)] += new_seg
#        print(sig_out[cursor_out:(cursor_out+win_len)])
        # % overlap add window normalization vec
        win_out[cursor_out:(cursor_out+win_len)] +=win
        # % move cursors
        cursor_out += step_len
#        print(cursor_out)
        cursor_in = cursor_in + round(divide(step_len,scale_factor))
#        print(cursor_in)
        # % new candidate
#        print(sig_in)
#        print(sig_in[(cursor_in):(cursor_in+win_len)])
        new_seg_cand = multiply(sig_in[(cursor_in):(cursor_in+win_len)],win)
        # % similarity calc. Note: matlab apears to be using FFT for xcorr, and
        # % so computes the whole thing (instead of just the center).
        # % Writing own version of xcorr would probably make it faster.
        #
        # % adjust cursor place
#        print(new_seg_cand)
#        print(new_seg_neighbour)
        if (simil_method== 'xcorr'):
            shift = max_xcorr_similarity(new_seg_neighbour, new_seg_cand, max_err_len)
#        elif (simil_method =='amdf'):
#            shift = min_amdf_similarity(new_seg_neighbour, new_seg_cand, max_err_len)
#        else:
#            return
#        int(print(shift))
        cursor_in = cursor_in-int(shift)


    # % remove slack
    delete(sig_out,s_[int(cursor_out)::]) 
    delete(win_out,s_[int(cursor_out)::])
    
    return divide(sig_out,(win_out))
    # %normalize to remove possible modulations

def max_xcorr_similarity(seg1, seg2, max_lag):
    corrArray=scipy.signal.correlate(seg1, seg2, 'same')
    maxValue = max(corrArray)
    
    max_i=[i for i, j in enumerate(corrArray)
           if j == maxValue
           ]
    
    return max_i - max_lag

#def min_amdf_similarity(seg1, seg2, max_lag):
#        n = len(seg1)
#        amdf = ones(1,2*max_lag-1)
#        for lag in range(int(-max_lag),int(max_lag)):
#            print lag
#            amdf[lag+max_lag+1] = sum(abs(seg2[max(1,(-lag+1)):min(n,(n-lag))]-\
#                           seg1[max(1,(lag+1)):min(n,(n+lag))] ))/n
#        minValue = min(amdf)
#        max_i=[i for i, j in enumerate(amdf)
#           if j == minValue
#           ]
#        return minValue - max_lag;


