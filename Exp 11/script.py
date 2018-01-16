import numpy as np
import cv2 as cv
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt


def constant(x,a):
	return a

#parameters
pixelsize = 4.65e-6 # micrometers
f = 200e-3  #millimeters
wavelenght = 633e-9 #nanometers 


errordistance = 2*pixelsize
#distance between peaks for intesity1,2 and 3 after background is substracted
distance = np.array([436-299,467-381,439-382])
distance = pixelsize*distance
print(distance*1e6,errordistance)
gratingperiod = np.array([25,40,60])

alpha = np.arctan(distance/f)

d = wavelenght/np.sin(alpha) #grating period in meters
errord = np.absolute(f*wavelenght/(distance**2*np.sqrt(distance**2/f +1)))*errordistance
pixelsize = (d/gratingperiod)*10**6  #in micrometers
errorpixelsize = (errord/gratingperiod)*10**6
print(pixelsize,errorpixelsize)

num = range(1,4)

popt, pcov = curve_fit(constant, num, pixelsize,sigma=errorpixelsize)
perr = np.sqrt(np.diag(pcov))

print(popt,perr)

plt.plot(num,pixelsize,'o')
plt.xlabel("Measurement #")
plt.ylabel(r"Pixel size [$\mu m$]")


plt.show()

#chi2 = sum((pixelsize-7.94148148)**2/(7.94148148))
#print(chi2)