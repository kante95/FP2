# Not a documentation
import numpy as np
from scipy.signal import argrelextrema
from scipy.fftpack import fft as fft_ciao
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

def amplitude_modulation(mu,A,fc,fm,t):
	return (1+mu*np.cos(2*np.pi*fm*t))*A*np.cos(2*np.pi*fc*t)

def carrier(t,A,fc):
	return A*np.cos(2*np.pi*fc*t)

def message(t,h,fm):
	return h*np.cos(2*np.pi*fm*t)


def plot_data_and_fft(t,ch1,f,fft,depth,freq):
	plt.figure()

	plt.subplot(311)
	t = np.arange(-2.0000E-04,2.0005E-04,0.0005E-4)
	carr = carrier(t,0.025,625e3)
	mess = message(t,0.025,50e3)
	plt.plot(t,carr,label = "carrier")
	plt.plot(t,mess,label = "message")
	plt.xlim(-3e-5,3e-5)
	plt.ylabel("Voltage [V]")

	plt.subplot(312)
	plt.plot(t, ch1)
	plt.xlabel("Time [s]")
	plt.ylabel("Voltage [V]")
	plt.xlim(-3e-5,3e-5)

	ax = plt.subplot(313)
	plt.plot(f, fft)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("FFT [dBm]")
	plt.axvline(x=625e3,linestyle ='--',color='#000000')
	plt.axvline(x=575e3,linestyle ='--',color='#000000')
	plt.axvline(x=675e3,linestyle ='--',color='#000000')
	plt.xlim(325e3,925e3)
	#display_peak(f,fft,ax)
	#theoretical_fft(t,ch1)

t = np.arange(-2.0000E-04,2.0005E-04,0.0005E-4)
v = amplitude_modulation(1,0.025,625e3,50e3,t)

N = len(t)
T = 0.0005e-4

yf = fft_ciao(v)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

dbm = np.abs(yf[0:int(N/2)])

plot_data_and_fft(t,v,xf,dbm,1,1)

plt.show()