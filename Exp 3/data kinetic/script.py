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


def line(x,m,q):
	return m*x+q

def conv(x):
    return x.replace(',', '.').encode()

def read_data(file, numcols=11):
    data = np.genfromtxt((conv(x) for x in open(file)), delimiter=";",
                         usecols=range(numcols), skip_header=100)
    return data[:, 1],data[:,6]

#first part of the experiment
degrees = np.array([23,30,35])

def t1():
	return np.arange(1,21,1)*60,[24,44]

def t2():
	return np.arange(3,19,1)*60,[48,64]

def t3():
	return np.arange(6,24,1)*60,[68,86]

options = {23 :t1,
           30 : t2,
           35 : t3,
}


def ind(temp):
	if temp == 23: return 0
	elif temp== 30: return 1
	else: return 2

k = np.zeros(3)
dk = np.zeros(3)
for temp in degrees:

	t1,file =  options[temp]()

	plt.figure()
	wavelength,data = read_data("Water.txt")
	plt.plot(wavelength,data,'*',label="Reference H20")
	i0 = np.amax(data)-1450   #simps(data[(wavelength>510 )& (wavelength <530)],wavelength[(wavelength>510) & (wavelength <530)]) 

	conc = np.zeros(len(t1))
	for i in range(file[0],file[1]):
		wavelength,data = read_data(str(temp)+'degrees/SP_'+str(i)+'.txt')
		plt.plot(wavelength,data,label = "t = "+str(t1[i-file[0]])+" sec")
		max_i  = np.amax(data) #simps(data[(wavelength>510 )& (wavelength <530)],wavelength[(wavelength>510) & (wavelength <530)])  
		conc[i-file[0]] = i0/(max_i-1450) 
	plt.xlim([490,550])
	plt.legend()
	#plt.title("Temp = "+str(temp))
	plt.xlabel("Wavelength [nm]")
	plt.ylabel("Intensity a.u.")

	#plt.savefig("spectrumtemp"+str(temp)+".png",dpi=500)

	y = np.log(np.log10(conc)) 
	t1 = t1[~np.isnan(y)]
	y = y[~np.isnan(y)]
	
	popt, pcov = curve_fit(line,t1,y)
	perr = np.sqrt(np.diag(pcov))
	k[ind(temp)] = popt[0]
	dk[ind(temp)] = perr[0]
	print(popt)

	plt.figure("fit")
	plt.plot(t1,y,'+--',markersize=10,label="temp = "+str(temp)+ "°")
	plt.xlabel("t [s]")
	plt.ylabel("ln(A)")
	plt.grid(True)
	plt.plot(t1, line(t1, *popt), 'r-',label='fit y = mx+q')
	plt.legend()
	#plt.savefig("fit.png",bbox_inches='tight',dpi=500)




plt.figure()
#dk[2]=0.000000000000000000001
plt.errorbar(1./degrees,k,yerr=dk/k,fmt='.')

popt, pcov = curve_fit(line,1./degrees,k)
perr = np.sqrt(np.diag(pcov))
print(popt)
print(perr)

plt.plot(1./degrees, line(1./degrees, *popt), 'r-',label='fit y = mx+q')
plt.grid(True)
plt.legend()
plt.xlabel("1/T [1/°]")
plt.ylabel("ln(k")

plt.show()

#plt.savefig("test.png",bbox_inches='tight')