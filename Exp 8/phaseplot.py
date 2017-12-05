import numpy as np
import matplotlib.pyplot as plt


def plot_data_and_fft(t,ch1,f,fft,depth,freq):
	plt.figure()

	plt.subplot(211)
	plt.plot(t, ch1)
	plt.xlabel("Time [s]")
	plt.ylabel("Voltage [V]")
	plt.xlim(-4e-5,4e-5)

	#x = np.linspace(-3e-5,3e-5,num=10000)
	#y = amplitude_modulation(depth,0.025,625e3,freq,x)
	#plt.plot(x,y,'r')

	ax = plt.subplot(212)
	plt.plot(f, fft)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("FFT [dBm]")
	#for n in range(0,6):
	#	freq1 = 625e3 + n*50e3 
	#	freq2 = 625e3 - n*50e3 
	#	plt.axvline(x=freq1,linestyle ='--',color='#000000')
	#	plt.axvline(x=freq2,linestyle ='--',color='#000000')
	plt.xlim(125e3,1125e3)
	
	#theoretical_fft(t,ch1)

#first part of the experiment
t, ch1 = np.loadtxt('data/DATA26.CSV', delimiter = ',', skiprows = 1, unpack = True)
f, fft = np.loadtxt('data/DATA27.CSV', delimiter = ',', skiprows = 1, unpack = True)
plot_data_and_fft(t,ch1,f,fft,1,1)
plt.show()