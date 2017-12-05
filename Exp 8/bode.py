import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


f, vin_min,vin_max,vout_min,vout_max,phase_min,phase_max = np.loadtxt('bode.csv', delimiter = ',', skiprows = 0, unpack = True)

phase = (phase_max+phase_min)/2
dphase = (phase_max-phase_min)/2

fix = lambda x: x if x>0 else 360+x
for i in range(len(phase)):
	phase[i] = fix(phase[i])


#H = vout/vin
#dH = H*np.sqrt((dvin/vin)**2+(dvout/vout)**2)

plt.figure()

#phase[10:] = phase[10:] + 360
plt.errorbar(f, phase,yerr=dphase,fmt='r.', markersize=5, ecolor='k')  # Bode phase plot
plt.xlabel("Frequency")
plt.ylabel("Phase [°]")
plt.grid(True)





time_delay = phase/(360*f)
dt = dphase/(360*f)

print(time_delay)
plt.figure()
plt.grid(True)
plt.errorbar(f,time_delay,yerr=dt,fmt='r.',ecolor='k')
plt.ylabel("Time [s]")
plt.xlabel("Frequency [Hz]")
plt.show()