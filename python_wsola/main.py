import wave
import scipy.io.wavfile
import sys
import wsola

rate, data = scipy.io.wavfile.read('clean.wav')
scipy.io.wavfile.write('clean2.wav',rate,data )
# file=wave.open('clean.wav','r')
# fs=data.getframerate()
scale_factor=2
data=data[8000:-1]
scaled=wsola.wsola(data, rate, scale_factor, 'xcorr')
# % sound(scaled(1:length(scaled)/10),fs);
filename = 'withScale %2f.wav' %scale_factor
# audiowrite(filename,scaled,fs);
# data=open(filename, 'w')
scipy.io.wavfile.write(filename,rate,scaled)
