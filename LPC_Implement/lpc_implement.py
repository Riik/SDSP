# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:47:58 2016

@author: Mert
"""

from scipy.io import wavfile
#import numpy as np
from numpy import hanning, correlate,divide,array,arange,zeros,multiply,argmax, \
spacing,append,rint,int16,abs,concatenate,roll,linalg,dot,var,shape

from math import ceil,floor
import scipy.io.wavfile
import scipy.signal.lfilter


def pressStack(X):
#%
#% Renders an overlap-add stack into the original signal.
#% It assumes the stacked signals are already windowed.
#%
#% X - a stacked overlap-add
#%
#% x - the rendered signal
#%
    [nw, count] = shape(X);
    step = floor(nw*0.5);
    n = (count-1)*step+nw;
    
    x = zeros(n, 1);
    for i in  range(count):
       x[ step*(i):step*(i-1+nw)  ] = x[  step*(i):step*(i-1+nw)  ] + X[:, i]
    
    return x


def lpcDecode(A, GFE, w, lowcut):
#%
#% Decodes the LPC coefficients into
#%
#% A - the LPC filter coefficients
#% GFE - the signal power(G) or the signal power with fundamental frequency(GF) 
#%       or the full source signal(E) of each windowed segment.
#% w - the window function
#% lowcut - the cutoff frequency in normalized frequencies for a lowcut
#%          filter.
#%
    nargin=lpcDecode.func_code.co_argcount
    if nargin < 4:
        lowcut = 0


[ne, n] = shape(GFE)
nw = len(w);

#% synthesize estimates for each chunk
Xhat = zeros(nw, n)

if ne < 2: #% GFE is only the signal power:
    
    for i in range(n):
        src = randn(nw, 1) # % noise
        Xhat[:,i] = w * lfilter( 1, [-1; A[:,i]], sqrt(GFE[i])*src);
    
    
#    % render down to signal
    xhat = pressStack(Xhat)
    
elseif ne < 3, % GFE is the pitch fequency and signal power
    F = GFE(2,:); % pitch frequency
    G = GFE(1,:); % power
    offset = 0;
    
    nw2 = round(nw/2);
    xhat = zeros(nw2*n, 1);
    
    for i = 1:n,
        
        % create source
        if F(i) > 0, % pitched
            src = zeros(nw2,1);
            
            step = round(1/F(i));
            pts = (offset+1):step:nw2;
            
            if ~isempty(pts),
                offset = step + pts(end) - nw2;
                src(pts) = sqrt(step); % impulse train, compensate power
            end
            
        else
            src = randn(nw2, 1); % noise
            offset = 0;
        end
        
        % filter
        xhat( nw2*i + (1:nw2) ) = filter( 1, [-1; A(:,i)], sqrt(G(i))*src);
    end
    
else % GFE is the error signal
    for i = 1:n,
        Xhat(:,i) = w .* filter( 1, [-1; A(:,i)], GFE(:,i));
    end

    % render down to signal
    xhat = pressStack(Xhat);
end
    
% dc blocker hack
if lowcut > 0,
    [b,a] = butter(10, lowcut, 'high'); 
    xhat = filter(b,a,xhat);
end    

return xhat

#%
#% Stacks a signal into overlap-add chunks.
#%
#% x - a single channel signal
#% w - the window function
#%
#% X - the overlap-add stack
#%
def stackOLA(x, w):
    n = len(x)
    nw = len(w)
    step = floor(nw*0.5)
    count = floor((n-nw)/step) + 1
    X = zeros(nw, count)
    for i in xrange(count):
        X[:, i] = w * x[(i-1)*step+1 : (i-1)*step+1+nw ]
    return X



def myLPC(x, p):
#% x - a single channel audio signal
#% p - the polynomial order of the all-pole filter

#% a - the coefficients to the all-pole filter
#% g - the variance(power) of the source
#% e - the full error signal
    N = len(x)
    
#    % form matrices
    b = x[1:N]
    xz=concatenate((x,zeros(p,1)), axis=0)
#    xz = array[x; zeros(p,1)]
    
    A = zeros(N-1, p)
    for i in range(p):
        temp = roll(xz, i)
        A[:, i] = temp[1:(N-1)]
    Ap = linalg.pinv(A)
#    % solve for a
    a = dot(Ap,b)
    
#    % calculate variance of errors
    e = b - dot(A,a)
    g = var(e)
    return [a,g,e]

def lpcEncode(x, p, w):
    X = stackOLA(x, w)
    [nw, n] = shape(X)
    A = zeros(p, n)
    G = zeros(1, n)
    E = zeros(nw, n)
    for i in range(n):
        [a, g, e] = myLPC(X[:,i], p)
        A[:, i] = a
        G[i] = g
        E[2:nw, i] = e

    return A,G
    
def lpc(data, rate, scale_factor):
    win_time = 0.020 #seconds
    overlap_ratio = 0.5
    
    max_err = min(0.005, divide(win_time*overlap_ratio,scale_factor/2))
    win_len=ceil(win_time*fs) #320
    max_err_length=ceil(max_err*fs) #around 80
    step_len=floor(overlap_ratio*win_len)
    win = hanning(win_len)
    p=6
    [A, G]=lpcEncode(x,p,win)
    xhat = lpcDecode(A, G, w);
filename_in='clean.wav'
rate, data = scipy.io.wavfile.read(filename_in)
scale_factor=1
scaled=lpc(data, rate, scale_factor, 'xcorr')

scaled=int16(scaled/max(abs(scaled)) * 32767)
filename_out = '%s withScale %.2f.wav' %(filename_in,scale_factor)
scipy.io.wavfile.write(filename_out,rate,scaled)