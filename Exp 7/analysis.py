import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.optimize import curve_fit

#function for reading our data
def read_data(file, numcols=3):
    data = np.genfromtxt( file, delimiter=",",
                         usecols=range(numcols), skip_header=1)
    return data[:,0],data[:,1],data[:,2]

def read_oscilloscope_data(file):
    data = np.genfromtxt( file, delimiter=",",
                         usecols=range(3,5), skip_header=0)
    return data[:,0],data[:,1]

def cos_fit(x, pmax):
	return pmax*np.cos(np.pi*x/180)**2

#Input power
angle,power,errors = read_data("power_polarizer.csv")
xerr = np.full_like(errors, 2/np.sqrt(12))
plt.figure()
plt.errorbar(angle,power,yerr=errors*10**-3,xerr=xerr,fmt='.',label="Experimental points")
plt.xlabel("Angle [°]")
plt.ylabel("Power [mW]")

toterr = np.sqrt(errors*10**-6 + (xerr*7.1*np.sin(xerr) )**2 )
params, pcov = curve_fit(cos_fit, angle, power,sigma=toterr)
perr = np.sqrt(np.diag(pcov))
print(params)
print(perr)
plt.plot(np.linspace(0,350,1300),cos_fit(np.linspace(0,350,1300),*params),label = r"fit $f(\theta) = P_{max}\cos^2\theta$")
plt.legend()

#SHG power
#function of theta
file = np.arange(0,36)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.arange(0,360,10)

plt.figure()
plt.errorbar(angles,mean,yerr=error,xerr = np.full_like(error, 2/np.sqrt(12)),fmt='.-',label="Experimental points" )
plt.xlabel(r"Angle $\phi$ [°]")
plt.ylabel("Voltage [V]")
plt.legend()


#close up of a maxium
file = np.arange(36,53)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.arange(302,268,-2)

plt.figure()
plt.errorbar(angles,mean,yerr=error,xerr = np.full_like(error,2/np.sqrt(12)),fmt='.' )
plt.xlabel("Angle [°]")
plt.ylabel("Voltage [V]")



def adjust_angle(x):
	return 0.8666666666666667*x-15

#as a function of phi
file = np.arange(53,62)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.concatenate([np.arange(35,65,5),np.arange(25,10,-5)])
angles = adjust_angle(angles)

#evil sorting
indexes = np.argsort(angles,kind = "mergesort")
print(indexes)
print(angles)
a = np.zeros_like(angles)
b = np.zeros_like(angles)
c =  np.zeros_like(angles)
j = 0
for i in indexes:
	a[j] = angles[i]
	b[j] = mean[i]
	c[j] = error[i]
	j+=1
plt.figure()
plt.errorbar(a,b,yerr=c,xerr = np.full_like(error,1/np.sqrt(12)),fmt='.-',label="Experimental points" )
#plt.errorbar(adjust_angle(angles),mean,yerr=error,xerr = np.full_like(error,1/np.sqrt(12)),fmt='.-',label="Experimental points" )
plt.xlabel(r"Angle $\theta$ [°]")
plt.ylabel("Voltage [V]")
plt.legend(loc = 3)





file = np.arange(62,73)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.arange(130,240,10)
plt.figure()
plt.errorbar(angles,mean,yerr=error,xerr = np.full_like(error,2/np.sqrt(12)),fmt='.' )
plt.xlabel("Angle [°]")
plt.ylabel("Voltage [V]")

plt.show()

#params, pcov = curve_fit(fit_sine, angle, data, sigma=dataerr)
#perr = np.sqrt(np.diag(pcov))
#plt.plot(np.linspace(0,200,300),fit_sine(np.linspace(0,200,300),*params),label = "fit")



