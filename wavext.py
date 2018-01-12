import scipy.io.wavfile
import math

rate, data = scipy.io.wavfile.read('/home/pietro/Desktop/babble recognizer/AUDIO/ATT/ATT000804_AB_01.wav')
i = 0
outfile = open("wave.dat","w")
while i < 1000: #len(data):
	outfile.write (str(data[i][0])+"\t"+str(data[i][1])+"\n")
	i += 1
print (str(data.shape))
print (str(rate))
print (str(len(data)))

