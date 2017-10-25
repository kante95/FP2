import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.optimize import curve_fit

#function for reading our data
def read_data(file, numcols=4):
    data = np.genfromtxt( file, delimiter="\t",
                         usecols=range(numcols), skip_header=5)
    return data[:, -1]


def fit_sine(x, a, omega, phi,b):
	return a * np.sin(omega * np.pi*x/180 + phi) + b

def visibility(file):
	data = np.zeros(4)
	dataerr = np.zeros(4)
	for i in range(4):
		temp_data = read_data("data/vis_N_"+file[i])
		data[i] = np.mean(temp_data)
		dataerr[i] = np.std(temp_data)/(np.sqrt(len(temp_data)))
		print(file[i]+ " "+ str(data[i]) + " "+ str(dataerr[i]))
	num = data[3]+data[2]-data[1]-data[0]
	den = np.sum(data)
	V = num/(den)
	Verr = V*(np.sqrt(sum(dataerr**2/(den)**2)+sum(dataerr**2/(num)**2)))
	return V,Verr



#Visibility
##HV base
file = ["HH","VV","HV","VH"]
print(visibility(file))

##HV base
file = ["DD","AA","AD","DA"]
print(visibility(file))



## Correlation plot
A=["0","45"]
plt.figure()
for i in A:
	angle = np.linspace(0,200,51)
	data = np.zeros(51)
	dataerr = np.zeros(51)
	for j in range(len(angle)):
		temp_data = read_data("data/Aangle"+str(i) + "/angle"+str(int(angle[j])))
		data[j] = np.mean(temp_data)
		dataerr[j] = np.std(temp_data)/(np.sqrt(len(temp_data)))
	plt.errorbar(angle,data,yerr=dataerr,xerr= np.full_like(dataerr, 2/np.sqrt(12)) ,fmt='.',label = "A fixed at "+i+"°" )
	params, pcov = curve_fit(fit_sine, angle, data, sigma=dataerr)
	perr = np.sqrt(np.diag(pcov))
	print(params,perr)
	plt.plot(np.linspace(0,200,300),fit_sine(np.linspace(0,200,300),*params),label = "fit")

plt.xlabel("Angle [°]")
plt.ylabel("Coincidences")
plt.legend()
#plt.show()


#Bell measurements
num = np.linspace(1,16,16)
bell = np.zeros(len(num))
error = np.zeros(len(num))
for i in num:
	temp_data = read_data("data/Bell/bell"+str(int(i)))
	bell[int(i-1)] = np.mean(temp_data)
	error[int(i-1)] = np.std(temp_data)/(np.sqrt(len(temp_data)))
print("Bell numbers: " + str(bell) + "+/-" + str(error))
#Bell's parameter
E1 = (bell[10]+bell[0]-bell[2]-bell[9])/(bell[10]+bell[0]+bell[2]+bell[9])
E2 = (bell[8]+bell[1]-bell[11]-bell[3])/(bell[8]+bell[1]+bell[11]+bell[3])
E3 = (bell[4]+bell[14]-bell[6]-bell[13])/(bell[4]+bell[14]+bell[6]+bell[13])
E4 = (bell[5]+bell[15]-bell[12]-bell[7])/(bell[5]+bell[15]+bell[12]+bell[7])
S = np.absolute(E1-E2)+np.absolute(E3+E4)
print(S)

