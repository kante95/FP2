import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.optimize import curve_fit

#function for reading our data
def read_data(file, numcols=4):
    data = np.genfromtxt( file, delimiter="\t",
                         usecols=range(numcols), skip_header=5)
    return data[:, -1]

def visibility(file):
	data = np.zeros(4)
	dataerr = np.zeros(4)
	for i in range(4):
		temp_data = read_data("data/vis_N_"+file[i])
		data[i] = np.mean(temp_data)
		dataerr[i] = np.std(temp_data)/(np.sqrt(len(temp_data)))
	V = (data[3]+data[2]-data[1]-data[0])/(np.sum(data))
	return V*100



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
	plt.errorbar(angle,data,yerr=dataerr,xerr= np.full_like(dataerr, 1/np.sqrt(12)) ,fmt='.',label = "A fixed at "+i )

plt.xlabel("Angle [°]")
plt.ylabel("Coincidences")
plt.legend()
plt.show()


#Bell measurements
num = np.linspace(1,16,16)
for i in num:
	temp_data = read_data("data/Bell/bell"+str(int(i)))
	mean = np.mean(temp_data)
	error = np.std(temp_data)/(np.sqrt(len(temp_data)))
	print("Bell number "+str(int(i)) + ": " + str(mean) + "+/-" + str(error))



#Bell's parameter
#E = 
#S = 

