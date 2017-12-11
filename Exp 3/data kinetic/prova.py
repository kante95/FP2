import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.integrate import simps

import matplotlib.pyplot as plt

def find_maximum(f,intensity):
	maxima = argrelextrema(intensity, np.greater)
	# print(maxima)

	filter_val = 3000
	peaks1 = maxima[0][intensity[maxima[0]] > filter_val]

	#text = r''
	#for i in ch1[peaks1]:
	#	text += str(i) + '\\'
	#plt.scatter(f[peaks1], ch1[peaks1], marker = 'x', color = 'r')
	#print( ch1[peaks1])
	#print(f[peaks1])
	#ax.text(335e3, -30, text, style='italic',
    #    bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
	#plt.plot(f, ch1, color = 'k')
	print(intensity[peaks1])
	return intensity[peaks1]


def conv(x):
    return x.replace(',', '.').encode()

def read_data(file, numcols=11):
    data = np.genfromtxt((conv(x) for x in open(file)), delimiter=";",
                         usecols=range(numcols), skip_header=100)
    return data[:, 1],data[:,6]

#first part of the experiment
degrees = [23,30,35]

t1 = np.arange(6,23,1)*60

plt.figure()
wavelength,data = read_data("Water.txt")
plt.plot(wavelength,data,'*',label="reference")
i0 = np.amax(data)   #simps(data[(wavelength>510 )& (wavelength <530)],wavelength[(wavelength>510) & (wavelength <530)]) 

conc = np.zeros(len(t1))
for i in range(68,85):
	wavelength,data = read_data('35degrees/SP_'+str(i)+'.txt')
	plt.plot(wavelength,data,label = "t = "+str(t1[i-68])+" min")
	max_i  = np.amax(data) #simps(data[(wavelength>510 )& (wavelength <530)],wavelength[(wavelength>510) & (wavelength <530)])  
	conc[i-68] = i0/max_i 
plt.xlim([490,550])
plt.legend()


def line(x,m,q):
	return m*x+q

popt, pcov = curve_fit(line,t1,np.log10(conc))

print(popt)

plt.figure()
plt.plot(t1,np.log(np.log10(conc)),'o--')
plt.xlabel("t [s]")
plt.ylabel("log10(I0/I)")
plt.grid(True)
plt.plot(t1, line(t1, *popt), 'r-',label='fit: m=%5.3f, 1=%5.3f' % tuple(popt))




plt.show()

