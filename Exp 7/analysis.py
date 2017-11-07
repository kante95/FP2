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

#Input power
angle,power,errors = read_data("power_polarizer.csv")
plt.figure()
plt.errorbar(angle,power,yerr=errors*10**-3,xerr=np.full_like(errors, 2/np.sqrt(12)),fmt='.')
plt.xlabel("Angle [°]")
plt.ylabel("Power [mW]")


#SHG power
file = np.arange(0,36)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.arange(0,360,10)

plt.figure()
plt.errorbar(angles,mean,yerr=error,xerr = np.full_like(error, 2/np.sqrt(12)),fmt='.-' )
plt.xlabel("Angle [°]")
plt.ylabel("Voltage [V]")



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




file = np.arange(53,62)
mean = np.zeros(len(file))
error = np.zeros(len(file))
for i in range(len(file)):
	t,v = read_oscilloscope_data("data/TEK00"+str(file[i]).zfill(2)+".CSV")
	mean[i] = np.mean(v)
	error[i] = np.std(v)/np.sqrt(len(v))

angles = np.concatenate([np.arange(35,65,5),np.arange(25,10,-5)])

plt.figure()
plt.errorbar(angles,mean,yerr=error,xerr = np.full_like(error,2/np.sqrt(12)),fmt='.' )
plt.xlabel("Angle [°]")
plt.ylabel("Voltage [V]")
plt.show()

#params, pcov = curve_fit(fit_sine, angle, data, sigma=dataerr)
#perr = np.sqrt(np.diag(pcov))
#plt.plot(np.linspace(0,200,300),fit_sine(np.linspace(0,200,300),*params),label = "fit")



