import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.optimize import curve_fit

#function for reading our data
def read_data(file, numcols=2):
    data = np.genfromtxt( file, delimiter=" ",
                         usecols=range(numcols), skip_header=0)
    return data[:, 0]


def lorentz(x,m,s):
	return (1/np.pi)*(s/((x-m)**2+s**2))



#Histogram of Z decays

masses = read_data("Zboson.csv")

plt.figure(1)
bins = np.logspace(1, 3.17, num=(1500-10)/10) # limits 10 -1500 and bin of 5
#bins = np.linspace(50,1500,num = (1500-50)/2)
plt.hist(masses, bins = bins)
plt.title("Histogram of invariant masses of possible Z boson candidates")
plt.xlabel("Mass [GeV]")
plt.ylabel("Number of events ")
plt.gca().set_xscale("log")
plt.yticks(np.arange(1, 20, 1,dtype=np.int32)) 


Zmasses = masses[masses<500]
mass = np.mean(Zmasses)
error = np.std(Zmasses)/m.sqrt(len(Zmasses))
print("Z boson mass: " + str(mass) +"+/-" + str(error))

Zprimemasses = masses[masses>500]
mass = np.mean(Zprimemasses)
error = np.std(Zprimemasses)/m.sqrt(len(Zprimemasses))
print("Z' boson  mass: " + str(mass) +"+/-" + str(error))

#fit
bins = np.linspace(60,130,num = (130-60)/3)
y,b = np.histogram(Zmasses, bins=bins,density=True)
for i in range(len(y)-1):
	b[i] = (b[i+1]+b[i])/2
b = b[:-1]
parameters, roba_inutile = curve_fit(lorentz,b,y)
print(parameters)
weights = np.ones_like(Zmasses)/(len(Zmasses))
plt.figure(3)
plt.hist(Zmasses,bins = bins,density=True)
x = np.linspace(60,130,num=(130-60)/0.1)
y = np.zeros_like(x)
for i in range(len(x)):
	y[i] = lorentz(x[i],*parameters)
plt.plot(x,y,label="fit")
plt.legend()

#Histogram of Higgs boson candidates

masses = read_data("Higgscandidate.csv")
#statistical analysis
mass = np.mean(masses)
error = np.std(masses)/m.sqrt(len(masses))
print("Higgs boson mass: " + str(mass) +"+/-" + str(error))


plt.figure(2)
bins = np.linspace(100, 170, num=(170-100)/2) #limits 100-170 bin of 2
plt.hist(masses, bins = bins)
plt.title("Histogram of invariant masses of possible Higgs boson candidates")
plt.xlabel("Mass [GeV]")
plt.ylabel("Number of events ")
plt.yticks(np.arange(1, 5, 1,dtype=np.int32)) #just for nice label
#plt.grid(True)
plt.show()