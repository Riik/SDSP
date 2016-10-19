[sig_in,fs]=audioread('clean.wav'); 
scale_factor=3;
sig_in=sig_in(8000:end);
scaled=wsola_time_scaling(sig_in, fs, scale_factor, 'xcorr');
% sound(scaled(1:length(scaled)/10),fs);
filename = sprintf('withScale %.02f .wav',scale_factor);
audiowrite(filename,scaled,fs);