import wave
import struct
import math
import random

SAMPLE_LEN= 100000

noise_output = wave.open('noise.wav', 'w')
noise_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))

for i in range(0, SAMPLE_LEN):
    ## for sine use code below 
#    frequency = 1000
#    value = int(15000*math.sin(i*2*math.pi*frequency/44100))

    ## for noise use this code
    value = random.randint(-32767, 32767)
    
    
    packed_value = struct.pack('h', value)
    noise_output.writeframes(packed_value)
    noise_output.writeframes(packed_value)

noise_output.close()