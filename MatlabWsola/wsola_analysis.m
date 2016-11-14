
function [yout,youtn]=wsola_analysis(y,fs,alpha,win_len,step_len,wtype,deltamax)

% wsola analysis of speech file
%
% Inputs:
%   y=input speech (normalized to 32767)
%   fs=input speech sampling rate
%   alpha=speed up/slow down factor (0.5 <= alpha <= 3.0)
%   win_len=length of analysis frame in samples
%   step_len=shift of analysis frame in samples
%   wtype=window type (1=Hamming, 0=rectangular, 2=triangular)
%   deltamax=maximum number of samples to search for best alignment
%   ipause=plotting option for debug
%
% Outputs:
%   youts=time scaled signal, scaled to 32767
%   youtn=time scaled signal, scaled to 1

% create appropriate window based on wtype parameter
if (wtype == 1)
    win=hamming(win_len);
elseif (wtype == 0)
    win=ones(win_len,1);
elseif (wtype == 2)
    win(1:win_len/2)=(.5:1:win_len/2)/(win_len/2);
    win(win_len/2+1:win_len)=(win_len/2-.5:-1:0)/(win_len/2);
    win=win';
end

% create search region based on deltamax parameter
deltas=round(deltamax*fs/1000);

fprintf('win_len:%d, step_len:%d, deltas:%d, fs:%d \n',win_len,step_len,deltas,fs);

% initialize overlap add with first frame
nideal=1+step_len;
nalpha=1;
cursor_out=1;
nsamp=length(y);
yout=zeros(floor(nsamp/alpha+win_len+0.5),1);
yout(1:win_len)=y(1:win_len).*win(1:win_len);
fno=2; % frame number
indices=[];
% full wsola processing
while (nideal+win_len <= nsamp & nalpha+win_len+deltas+alpha*step_len <= nsamp)
    xideal=y(nideal:nideal+win_len-1);
    nalpha=nalpha+round(alpha*step_len);
    indexl=max(nalpha-deltas,1);
    xreal=y(indexl:nalpha+win_len-1+deltas);
    cursor_out=cursor_out+step_len;
    
    % correlate pair of frames and find optimal matching point
    c=xcorr(xreal,xideal);
    length(c)
    % compensate for xcorr shift by length of xreal; find peak of length compensated cs
    lxreal=length(xreal)
    cs(1:2*deltas+1)=c(lxreal:lxreal+2*deltas);
    maxind=find(cs == max(cs));
    
    indices=[indices;maxind];
   
    % overlap add the best match
    xadd=y(nalpha-deltas+maxind-1:nalpha-deltas+maxind+win_len-2).*win(1:win_len);
    yout(cursor_out:cursor_out+win_len-1)=yout(cursor_out:cursor_out+win_len-1)+xadd;

    
    % update indices
    nideal=nalpha-deltas+maxind-1+step_len;
end

% play out time-altered sound
youtn=yout/max(max(yout),-min(yout));
% sound(youtn,fs);

% save output file
youts=youtn*32700;

save('indices.mat','indices')
end