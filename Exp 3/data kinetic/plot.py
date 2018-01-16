import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 16


def a(t):
	return np.exp(-0.5*t)

def b(t):
	return (0.1/0.5)*(1-np.exp(-0.5*t))

def c(t):
	return (0.3/0.5)*(1-np.exp(-0.5*t))

t = np.arange(0,12,0.01)

plt.figure()

plt.plot(t,a(t),label="[A]") 
plt.plot(t,b(t),label="[B]") 
plt.plot(t,c(t),label="[C]")

plt.legend()
plt.xlabel("Time") 
plt.ylabel("Concentration")

plt.show()