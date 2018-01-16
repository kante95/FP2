import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16
directory = "data_rubidium/"

def multi_lorentz_peak(w, Voff, Vlorentz, width, wresonance,
                        Vlorentz2, width2, wresonance2,
                        Vlorentz3, width3, wresonance3,
                        Vlorentz4, width4, wresonance4,
                        Vlorentz5, width5, wresonance5,
                        Vlorentz6, width6, wresonance6):
    return (Voff+Vlorentz*width/((w-wresonance)**2+(width)**2)+
                Vlorentz2*width2/((w-wresonance2)**2+(width2)**2)+
                Vlorentz3*width3/((w-wresonance3)**2+(width3)**2)+
                Vlorentz4*width4/((w-wresonance4)**2+(width4)**2)+
                Vlorentz5*width5/((w-wresonance5)**2+(width5)**2)+
                Vlorentz6*width6/((w-wresonance6)**2+(width6)**2))

def multi_lorentz_peak2(w, Voff, Vlorentz, width, wresonance,
                        Vlorentz2, width2, wresonance2,
                        Vlorentz3, width3, wresonance3,
                        Vlorentz4, width4, wresonance4,
                        Vlorentz5, width5, wresonance5,
                        Vlorentz6, width6, wresonance6,
                        Vlorentz7, width7, wresonance7):
    return (Voff+Vlorentz*width/((w-wresonance)**2+(width)**2)+
                Vlorentz2*width2/((w-wresonance2)**2+(width2)**2)+
                Vlorentz3*width3/((w-wresonance3)**2+(width3)**2)+
                Vlorentz4*width4/((w-wresonance4)**2+(width4)**2)+
                Vlorentz5*width5/((w-wresonance5)**2+(width5)**2)+
                Vlorentz6*width6/((w-wresonance6)**2+(width6)**2)+
                Vlorentz7*width7/((w-wresonance7)**2+(width7)**2))


def lorentz(x,A,m,s):
	return A*(s/((x-m)**2+s**2))

def lorentz_with_offset(x,A,m,s,offset):
	return A*(s/((x-m)**2+s**2)) + offset

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2)) 

def read_oscilloscope_data(file):
    data = np.genfromtxt( file, delimiter=",",
                         usecols=range(3,5), skip_header=0)
    return data[:,0],data[:,1]


#fabryperot
t,v = read_oscilloscope_data(directory+"ALL0106/F0106CH1.CSV")
plt.errorbar(t,v,yerr = (1.6/256)*np.ones(len(t)))
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")


def center_of_peak(t,v,low,high):
	#first peak
	v = v[(t> low) & (t< high)]
	t = t[(t> low) & (t< high)]
	plt.figure()
	plt.errorbar(t,v,yerr = (1.6/256)*np.ones(len(t)),fmt='.',markersize=5,label="Experimental data")
	#popt, pcov = curve_fit(gaussian, t, v,bounds = ([-np.inf,-0.0010,0],[np.inf,-0.0004,0.00025]))
	#perr = np.sqrt(np.diag(pcov))

	popt, pcov = curve_fit(lorentz, t, v,sigma = (1.6/256)*np.ones(len(t)),bounds = ([0,low,0],[np.inf,high,0.0004]))
	perr = np.sqrt(np.diag(pcov))
	print("Lorentzian peak fit: A,t0,gamma")
	print(popt,perr)
	print("Reduced chi2")
	chi_squared = np.sum( ((lorentz(t, *popt)-v)/(1.6/256))**2 )
	reduced_chi_squared = chi_squared / (len(t) - len(popt))
	print(reduced_chi_squared)
	t1 = np.arange(low,high,0.0000001)
	plt.plot(t1, lorentz(t1, *popt), 'r-', label='Lorentzian fit')

	popt, pcov = curve_fit(gaussian, t, v,sigma = (1.6/256)*np.ones(len(t)),bounds = ([0,low,0],[np.inf,high,0.0004]))
	perr = np.sqrt(np.diag(pcov))
	print("Gaussia peak fit: B,t0,sigma")
	print(popt,perr)
	print("Reduced chi2")
	chi_squared = np.sum( ((gaussian(t, *popt)-v)/(1.6/256))**2 )
	reduced_chi_squared = chi_squared / (len(t) - len(popt))
	print(reduced_chi_squared)

	t = np.arange(low,high,0.000001)
	plt.plot(t, gaussian(t, *popt), 'g-', label='Gaussian fit')
	plt.legend()
	plt.xlabel("Time [s]")
	plt.ylabel("Voltage [V]")
	return popt[1],perr[1]


center1,dcenter1 = center_of_peak(t,v,-0.000750,-0.000640)
center2,dcenter2 = center_of_peak(t,v,0.00294,0.00304)

c = 299792458
L = 20e-2
real_fsr = c/L
time_fsr = -center1+center2
dtime_fsr = np.sqrt(dcenter1**2 + dcenter2**2)

print("Free spectral range: %f +/- %f",time_fsr,dtime_fsr)

def t2freq(t):
	return (real_fsr/time_fsr)*t*1e-6 #in MHz

#hyperfine structure

t,v = read_oscilloscope_data(directory+"ALL0102/F0102CH3.CSV")
t = t2freq(t)
tb,vb = read_oscilloscope_data(directory+"ALL0103/F0103CH3.CSV")
tb= t2freq(tb)
plt.figure()
plt.errorbar(t,v,yerr = (0.08/(256*np.sqrt(12)))*np.ones(len(t)), label="Hyperfine spectrum",fmt='.',markersize=5)
plt.errorbar(tb,vb,yerr = (0.08/(256*np.sqrt(12)))*np.ones(len(tb)),label="Background",fmt='.',markersize=5)
plt.xlabel("Detuning [MHz]")
plt.ylabel("Voltage [V]")
plt.legend()
diff = v-vb

# high = 450
# low = -150
# diff = diff[(t> low) & (t< high)]
# t = t[(t> low) & (t< high)]

plt.figure()
#plt.errorbar(t,diff,yerr = (0.08/256)*np.ones(len(t)),fmt="." )
plt.errorbar(t,diff,yerr = (2*0.08/(256*np.sqrt(12)))*np.ones(len(tb)),fmt='.',label = "Experimental data",markersize=5)
p= [-5.10681072e-03 ,3.76653515e-02,  1.58333476e+01, -7.77073287e+01,1.53816471e-01 ,  1.45447758e+01,   6.85368665e+00  , 1.06667251e-01, 1.31304916e+01 ,  9.13358709e+1  ,7.13693568e-1,   1.69622715e1, 2.37608123e+02 ,3.57212145e-1 ,  1.22766961e1 ,  1.52825804e+02,1.76598802e-3  , 1.62151338e1,378,1,1,1]


# def lorentz_peaks(t,v,low,high):
# 	#first peak
# 	v = v[(t> low) & (t< high)]
# 	t = t[(t> low) & (t< high)]

# 	popt, pcov = curve_fit(lorentz_with_offset, t, v,sigma = (0.04/256)*np.ones(len(t)),bounds = ([0,low,10,0.005],[np.inf,high,50,np.inf]))
# 	perr = np.sqrt(np.diag(pcov))
# 	print(popt,perr)
# 	t1 = np.arange(low,high,1)
# 	plt.plot(t1, lorentz_with_offset(t1, *popt), 'r-', label='fit: A=%f,  xc=%f, s=%f offset=%f' % tuple(popt))
# 	plt.legend()

# 	return popt[1]

# lorentz_peaks(t,v,190,270)

popt, pcov = curve_fit(multi_lorentz_peak2, t, diff,sigma = (0.08/256)*np.ones(len(t)),p0=p,method="lm")
perr = np.sqrt(np.diag(pcov))
print("Multi lorentzian fit")
print(popt)
print(perr)

#popt= [-5.07632489e-03, -1.80209252e-02 , -9.04256941e+00  ,-7.82219782e+01,1.794366e-01,   1.63950133e+01 ,  7.25825710e+00 ,  1.43503961e-01,1.59195514e+01 ,  9.13954496e+01 ,  7.98698005e-01 ,  1.84465199e+01,2.37651304e+02  , 4.03636692e-01 ,  1.33479294e+01,   1.52940906e+02 ,1.09375260e-02  ,7 , 378]
plt.plot(t, multi_lorentz_peak2(t, *popt), 'r-',label="Multi lorentzian fit", zorder=20)
plt.xlim([-150,450])
plt.xlabel("Detuning [MHz]")
plt.ylabel("Voltage [V]")
plt.grid(True)
plt.legend()

peak1 = popt[3]
peak2 = popt[9]
peak3 = popt[18]
dpeak1 = perr[3]
dpeak2 = perr[9]
dpeak3 = perr[18]
print("Picco F=1: "+str(popt[3])+ "+/-" + str(dpeak1))
print("Picco F=2: "+str(popt[9])+ "+/-" + str(dpeak2))
print("Picco F=3: "+str(popt[18])+ "+/-" + str(dpeak3))
print("Altri picchi:" +str(popt[6])+" "+str(popt[12])+" "+str(popt[15])+" "+str(popt[21]))

peak12 = peak2-peak1
peak13 = peak3-peak1
peak23 = peak3-peak2

dpeak12 = np.sqrt(dpeak1**2 + dpeak2**2)
dpeak13 = np.sqrt(dpeak3**2 + dpeak1**2)
dpeak23 = np.sqrt(dpeak2**2 + dpeak3**2)

print("Distanza picco F 1->2:" +str(peak12) + "+/-" + str(dpeak12))
print("Distanza picco F 1->3:" +str(peak13) + "+/-" + str(dpeak13))
print("Distanza picco F 2->3:" +str(peak23) + "+/-" + str(dpeak23))

A = (peak23+peak12)/5
B = (2*peak23-3*peak12)/5
dA = (1/5)*np.sqrt(dpeak23**2 + dpeak12**2)
dB = (1/5)*np.sqrt((2*dpeak23)**2 + (3*dpeak12)**2)

print("Magnetic dipole constant: h"+str(A)+ "+/-" + str(dA))
print("Electric quadrupole constant: h"+str(B)+ "+/-" + str(dB))


plt.show()

