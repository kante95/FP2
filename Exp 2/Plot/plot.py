import matplotlib.pyplot as plt
import numpy as np


#function for reading our data
def read_data(file, numcols=2):
    data = np.genfromtxt( file, delimiter=" ",
                         usecols=range(numcols), skip_header=0)
    return data[:, 0]


#Histogram of Z decays

masses = read_data("Zboson.csv")

plt.figure(1)
bins = np.logspace(1, 3.17, num=(1500-10)/10) # limits 10 -1500 and bin of 5
plt.hist(masses, bins = bins)
plt.title("Histogram of invariant masses of possible Z boson candidates")
plt.xlabel("Mass [GeV]")
plt.ylabel("Number of events ")
plt.gca().set_xscale("log")
plt.yticks(np.arange(1, 20, 1,dtype=np.int32)) 

mass = np.mean(masses[masses<500])
error = np.std(masses[masses<500])
print("Z boson mass: " + str(mass) +"+/-" + str(error))

mass = np.mean(masses[masses>500])
error = np.std(masses[masses>500])
print("Z' boson  mass: " + str(mass) +"+/-" + str(error))


#Histogram of Higgs boson candidates

masses = read_data("Higgscandidate.csv")
#statistical analysis
mass = np.mean(masses)
error = np.std(masses)
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