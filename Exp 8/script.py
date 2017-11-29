# Not a documentation
import numpy as np
from scipy.signal import argrelextrema
from scipy.fftpack import fft as fft_ciao
import matplotlib.pyplot as plt

from matplotlib.ticker import EngFormatter

def amplitude_modulation(mu,A,fc,fm,t):
	return (1+mu*np.cos(2*np.pi*fm*t))*A*np.cos(2*np.pi*fc*t)


def display_peak(f,ch1,ax):
	maxima = argrelextrema(ch1, np.greater)
	# print(maxima)

	filter_val = -60
	peaks1 = maxima[0][ch1[maxima[0]] > filter_val]

	#text = r''
	#for i in ch1[peaks1]:
	#	text += str(i) + '\\'
	plt.scatter(f[peaks1], ch1[peaks1], marker = 'x', color = 'r')
	print( ch1[peaks1])
	#ax.text(335e3, -30, text, style='italic',
    #    bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
	#plt.plot(f, ch1, color = 'k')

def theoretical_fft(t,v):
	N = len(t)
	T = 0.0005e-4

	yf = fft_ciao(np.hanning(len(v))*v)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	
	#plt.plot(xf, 30+10*np.log10(np.abs(yf[0:int(N/2)])**2/50)) #evil hack, wiki says it should be +60!!?


def plot_data_and_fft(t,ch1,f,fft,depth,freq):
	plt.figure()

	plt.subplot(211)
	plt.plot(t, ch1)
	plt.xlabel("Time [s]")
	plt.ylabel("Voltage [V]")
	plt.xlim(-3e-5,3e-5)

	x = np.linspace(-3e-5,3e-5,num=10000)
	y = amplitude_modulation(depth,0.025,625e3,freq,x)
	#plt.plot(x,y,'r')

	ax = plt.subplot(212)
	plt.plot(f, fft)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("FFT [dBm]")
	plt.axvline(x=625e3,linestyle ='--',color='#000000')
	plt.axvline(x=575e3,linestyle ='--',color='#000000')
	plt.axvline(x=675e3,linestyle ='--',color='#000000')
	plt.xlim(325e3,925e3)
	display_peak(f,fft,ax)
	#theoretical_fft(t,ch1)

#first part of the experiment
freq = [50e3,50e3,50e3,30e3]
AM_depth = [1,0.5,0.1,1]
for i in range(1,7,2):
	t, ch1 = np.loadtxt('data/DATA'+str(i).zfill(2)+'.CSV', delimiter = ',', skiprows = 1, unpack = True)
	f, fft = np.loadtxt('data/DATA'+str(i+1).zfill(2)+'.CSV', delimiter = ',', skiprows = 1, unpack = True)
	plot_data_and_fft(t,ch1,f,fft,AM_depth[int(np.floor(i/2))],freq[int(np.floor(i/2))])
#plt.show()



#Bode
f, vin_min,vin_max,vout_min,vout_max,phase_min,phase_max = np.loadtxt('bode.csv', delimiter = ',', skiprows = 0, unpack = True)

vin = (vin_max+vin_min)/2
dvin = (vin_max-vin_min)/2
vout = (vout_max+vout_min)/2
dvout = (vout_max-vout_min)/2
phase = (phase_max+phase_min)/2
dphase = (phase_max-phase_min)/2

fix = lambda x: x if x>0 else 360+x
for i in range(len(phase)):
	phase[i] = fix(phase[i])


H = vout/vin
dH = H*np.sqrt((dvin/vin)**2+(dvout/vout)**2)

plt.figure()

ax = plt.subplot(211)
plt.grid(True)
plt.errorbar(f,H,yerr=dH,fmt='r.', markersize=5, ecolor='k')    # Bode magnitude plot
plt.ylabel("Magnitude")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
ax.set_xscale("log", nonposx='clip')


ax2 = plt.subplot(212)
plt.errorbar(f, phase,yerr=dphase,fmt='r.', markersize=5, ecolor='k')  # Bode phase plot
plt.xlabel("Frequency")
plt.ylabel("Phase [°]")
plt.grid(True)
ax2.set_xscale("log", nonposx='clip')
formatter0 = EngFormatter(unit='Hz')
ax2.xaxis.set_major_formatter(formatter0)


plt.show()
