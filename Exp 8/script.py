# Not a documentation
import numpy as np
from scipy.signal import argrelextrema
from scipy.fftpack import fft as fft_ciao
import matplotlib.pyplot as plt

from matplotlib.ticker import EngFormatter

from scipy.special import jn
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
	#plt.scatter(f[peaks1], ch1[peaks1], marker = 'x', color = 'r')
	print( ch1[peaks1])
	#print(f[peaks1])
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

#4th amplitude
t, ch1 = np.loadtxt('data/DATA07.CSV', delimiter = ',', skiprows = 1, unpack = True)
f, fft = np.loadtxt('data/DATA08.CSV', delimiter = ',', skiprows = 1, unpack = True)

plt.figure()

plt.subplot(211)
plt.plot(t, ch1)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.xlim(-5e-5,5e-5)

ax = plt.subplot(212)
plt.plot(f, fft)
plt.xlabel("Frequency [Hz]")
plt.ylabel("FFT [dBm]")
plt.axvline(x=625e3,linestyle ='--',color='#000000')
plt.axvline(x=595e3,linestyle ='--',color='#000000')
plt.axvline(x=655e3,linestyle ='--',color='#000000')
plt.xlim(325e3,925e3)

display_peak(f,fft,ax)


#frequency modulation
t, ch1 = np.loadtxt('data/DATA09.CSV', delimiter = ',', skiprows = 1, unpack = True)
f, fft = np.loadtxt('data/DATA10.CSV', delimiter = ',', skiprows = 1, unpack = True)

plt.figure()

plt.subplot(211)
plt.plot(t, ch1)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.xlim(-4e-5,4e-5)
plt.grid(True)

ax = plt.subplot(212)
plt.plot(f, fft)
plt.xlabel("Frequency [Hz]")
plt.ylabel("FFT [dBm]")
plt.axvline(x=625e3,linestyle ='--',color='#000000')

for n in range(1,4):
	freq1 = 625e3 + n*50e3 
	freq2 = 625e3 - n*50e3 
	plt.axvline(x=freq1,linestyle ='--',color='#000000')
	plt.axvline(x=freq2,linestyle ='--',color='#000000')
plt.xlim(325e3,925e3)

display_peak(f,fft,ax)


t, ch1 = np.loadtxt('data/DATA12.CSV', delimiter = ',', skiprows = 1, unpack = True)
f, fft = np.loadtxt('data/DATA13.CSV', delimiter = ',', skiprows = 1, unpack = True)

plt.figure()

plt.subplot(211)
plt.plot(t, ch1)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.xlim(-4e-5,4e-5)
plt.grid(True)

ax = plt.subplot(212)
plt.plot(f, fft)
plt.xlabel("Frequency [Hz]")
plt.ylabel("FFT [dBm]")
plt.axvline(x=625e3,linestyle ='--',color='#000000')

for n in range(1,4):
	freq1 = 625e3 + n*50e3 
	freq2 = 625e3 - n*50e3 
	plt.axvline(x=freq1,linestyle ='--',color='#000000')
	plt.axvline(x=freq2,linestyle ='--',color='#000000')
plt.xlim(325e3,925e3)

display_peak(f,fft,ax)



fig, ax1 = plt.subplots()
for n in range(0,4):
	x = np.arange(0,4.5,0.01)
	y = jn(n,x)
	ax1.plot(x,y,label= "n = "+str(n))
	#plt.grid()
	plt.legend()
mu = np.array([5,10,20,30,40,50,60,75,90,110,130,150,175,200])/50
for i in mu:
	ax1.axvline(x=i,linestyle =':',color='#000000')
	for n in range(0,4):
		ax1.plot(i, jn(n,i), 'r.',markersize=7)
mustr = (mu*50)
#plt.xticks(mu,mustr)
ax1.set_xlabel(r"$\mu$")
ax1.set_ylabel(r"$J_n(\mu)$")
ax1.set_xlim([0,4.3])

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(mu)
ax2.set_xticklabels(mustr)
ax2.set_xlabel(r"Frequency deviation $f_\Delta$ [kHz]")



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



time_delay = phase/(2*np.pi*f)
print(time_delay)
plt.figure()
plt.plot(f,time_delay,'.')
plt.show()
