[y,fs]=audioread('clean.wav'); 
scale_factor=0.5;
nleng=320;
nshift=160;
wtype=1;
deltamax=5;
[yout,youtn]=wsola_analysis(y,fs,scale_factor,nleng,nshift,wtype,deltamax);
filename = sprintf('withScale %.02f .wav',scale_factor);
audiowrite(filename,youtn,fs);