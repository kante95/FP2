import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

plt.rcParams['font.size'] = 16

def line(x,a,b):
	return a*x + b

def lasing_threshold(i,p,powererr,lower,cut1,cut2,upper,color):
	current_left = i[(i>lower)&(i<cut1)]
	power_left = p[(i>lower)&(i<cut1)]
	current_right = i[(i>cut2)&(i<upper)]
	power_right = p[(i>cut2)&(i<upper)]
	popt, pcov = curve_fit(line,current_right, power_right,sigma = powererr[(i>cut2)&(i<upper)])
	perr = np.sqrt(np.diag(pcov))

	cr = np.arange(np.min(current_right)-0.5,np.max(current_right),0.1)
	plt.plot(cr, line(cr, *popt), color, label='fit')

	a_right = popt[0]
	b_right = popt[1]
	da_right = perr[0]
	db_right = perr[1]

	popt, pcov = curve_fit(line,current_left, power_left,sigma = powererr[(i>lower)&(i<cut1)])
	perr = np.sqrt(np.diag(pcov))

	cr = np.arange(np.min(current_left),np.max(current_left)+0.5,0.1)
	plt.plot(cr, line(cr, *popt), color, label='fit')

	a_left = popt[0]
	b_left = popt[1]
	da_left = perr[0]
	db_left = perr[1]

	num = a_left - a_right
	den = b_right - b_left
	lasing_x = den/num
	dlasing_x = np.absolute(lasing_x)*np.sqrt( (da_left/den)**2 + (da_right/den)**2+(da_left/num)**2 + (da_right/num)**2  )
	return lasing_x,dlasing_x

plt.figure()

#Characterization of the diode laser
input_current = np.array([3.06, 4.98, 6.97,8.97,11.06,12.96,15.03,17.09,19.07,20.99,23.03,25.08,27.01,16.02,18.04,20.03,19.50,18.55,20.48,19.82,21.97  ]) #unit mA
power = np.array([10.15, 13.78, 16.43,19.56,23.40,27.74,34.48,48.55,315.5,969.1,1506,1917,2201, 39.23,71.45,673.3, 486.3,140.1,824.7,600.7,1270       ])*1e-3	#unit uW
error_power = np.array([0.48, 0.02, 0.05, 0.006,0.006,0.01,0.015,0.026,0.42,0.71,0.674,0.859,0.585,0.075,0.053,0.5, 0.333,0.285,0.549 ,0.69,1.25      ])*1e-3 # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Experimental points")


lasing, dlasing = lasing_threshold(input_current,power,error_power,3,18.1,18.3,22,'r')
print(lasing,dlasing)
plt.xlabel("Input current [mA]")
plt.ylabel("Laser power [mW]")

plt.legend()

#Littrow configuration
plt.figure()
input_current = np.array([18.07,4.95,8.03,11.9,15.02,17.09,18.5,19,19.5,20,20.54,21.5,22,22.99, 24.04,25.03,26.08,27.03   ]) #unit mA
power = np.array([147.8,2.53,4.75,8.47,13.51,21.96,231.6,332.1,401.7,473.8,586.9,664.6,775.8,888.9, 1001,1111,1179,1200          ])*1e-3	#unit uW
error_power = np.array([0.26,450e-6,930e-6,1.37e-3,2.31e-3,13e-3,6.74,0.31,0.521,1.095,1.344,1.134,2.185,1.256,5.089,0.49,1.065,0.408     ])*1e-3 # unit uW

lasing, dlasing = lasing_threshold(input_current,power,error_power,3,17.2,16.8,22,'g')
print(lasing,dlasing)

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Experimental points")
plt.xlabel("Input current [mA]")
plt.ylabel("Laser power [mW]")

plt.legend()

#littman configuration
plt.figure()

#with feedback
input_current = np.array([9.99,12.28,13.86,15.86, 17.97, 19.06,20.08,21,23.02,25.25,18.51 ,19.58    ]) #unit mA
power = np.array([6.766,9.297,11.59,16.20 ,34.18,200.6, 403.3,561.2,872.3,1109,83.21,308.8           ])	#unit uW
error_power = np.array([2.99e-3,1.32e-3,4.4e-3, 8.81e-3,25.8e-3,0.21,0.42,0.3,0.341,0.31,0.34,0.71 ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="With feddback")

lasing, dlasing = lasing_threshold(input_current,power,error_power,3,18.3,18.3,22,'r')
print(lasing,dlasing)
#without feedback

input_current = np.array([10.07,12,14.08,16.18, 17.99,20.04,19.023,23.10,24.96,22.06,20.48,18.54,19.5 ,21.55]) #unit mA
power = np.array([7, 9.05,12.05,16.89, 34.34,370.9, 167.2  ,852.5,1050,711.3,455.5,76.86,268,637.6    ])	#unit uW
error_power = np.array([162e-6,2.3e-3,1.8e-3,3.2e-3,38e-3,300e-3,330e-3,571e-3,469e-3,378e-3,241e-3,181e-3,294e-3,452e-3    ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Whithout feddback")
plt.xlabel("Input current [mA]")
plt.ylabel(r"Laser power [$\mu$W]")

lasing, dlasing = lasing_threshold(input_current,power,error_power,3,18.3,18.3,22,'g')
print(lasing,dlasing)

plt.legend()


#geometric calculations
print("Geometric calculations")
#littrow configuration
a = 4.5 #cm
b = 6 #cm
c = 6.5 #cm
dgrating = (1/1200)*1e6 #mm -> nm

wavelength = 2*dgrating*np.sin(0.5*np.arccos((-a**2 + b**2 + c**2)/(2*b*c)))
print(wavelength)

#error analysis
da = (a*dgrating*(1/np.sin(0.5*np.arccos((-a**2 + b**2 + c**2)/(2*b*c)))))/(2*b*c)
db = -(dgrating*(a**2 + b**2 - c**2)*(1/np.sin(0.5*np.arccos((-a**2 + b**2 + c**2)/(2*b*c)))))/(4*b**2*c)
dc = -(dgrating*(a**2 - b**2 + c**2)*(1/np.sin(0.5*np.arccos((-a**2 + b**2 + c**2)/(2*b*c)))))/(4*b**2*c)
errormeter = 0.1 #cm
dwavelenght = np.sqrt((da*errormeter)**2+(dc*errormeter)**2+(db*errormeter)**2)
print(dwavelenght)

#littman configuration

def error_arccos(a1,b1,c1):
	errormeter = 0.1 #cm
	da = -(a1**2 - b1**2 + c1**2)/(2*a1**2 *b1*np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	db = -(-a1**2 + b1**2 + c1**2)/(2*a1*b1**2 *np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	dc =  c1/(a1*b1*np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	return np.sqrt((da*errormeter)**2+(dc*errormeter)**2+(db*errormeter)**2)

#crap i apologize
def error_arccos_modified(a1,b1,c1,dc1):
	errormeter = 0.1 #cm
	da = -(a1**2 - b1**2 + c1**2)/(2*a1**2 *b1*np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	db = -(-a1**2 + b1**2 + c1**2)/(2*a1*b1**2 *np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	dc =  1/(2*a1*b1*np.sqrt(1 - (a1**2 + b1**2 - c1**2)**2/(4*a1**2*b1**2)))
	return np.sqrt((da*errormeter)**2+(dc*dc1)**2+(db*errormeter)**2)

a = 8 #cm
b = 9.5 #cm
c = 16.5 #cm
d = 17 #cm
e = 23 #cm
errormeter = 0.1

print("\n")
gamma = np.arccos((c**2+e**2-d**2)/(2*c*e))
dgamma = error_arccos(c,e,d)
print("gamma: "+str(np.rad2deg(gamma))+" +/- "+str(np.rad2deg(dgamma)))
theta = np.arccos((a**2+c**2-b**2)/(2*a*c))
dtheta = error_arccos(a,c,b)
print("theta: "+str(np.rad2deg(theta))+" +/- "+str(np.rad2deg(dtheta)))
Gamma = np.arccos((c**2+d**2-e**2)/(2*c*d))
dGamma = error_arccos(c,d,e)
print("Gamma: "+str(np.rad2deg(Gamma))+" +/- "+str(np.rad2deg(dGamma)))

l2 = a**2+d**2-2*a*d*np.cos(theta+Gamma)

dl2 = np.sqrt( (errormeter*(2*a-2*d*np.cos(theta+Gamma)) )**2 + (errormeter*(2*d-2*a*np.cos(theta+Gamma)) )**2+(dtheta*(2*a*d*np.sin(theta+Gamma)) )**2+(dGamma*(2*a*d*np.sin(theta+Gamma)))**2)
print("j2: "+str(l2)+" +/- "+str(dl2) )
alpha = 0.5*np.arccos((b**2+e**2-l2)/(2*b*e))
dalpha = 0.5*error_arccos_modified(b,e,np.sqrt(l2),dl2) 
print("alpha: "+str(np.rad2deg(alpha))+" +/- "+str(np.rad2deg(dalpha)) )
beta = -alpha + gamma 
dbeta = np.sqrt(dalpha**2 +dgamma**2)
print("beta: "+str(np.rad2deg(beta))+" +/- "+str(np.rad2deg(dbeta)))
wavelength = dgrating*(np.sin(alpha)+np.sin(beta))
dwavelenght = np.sqrt((d*np.cos(alpha)*dalpha)**2+(d*np.cos(beta)*dbeta)**2 )
print("wavelength: " + str(wavelength)+" +/- "+str(np.rad2deg(dbeta)))





#Last part

def read_oscilloscope_data(file):
    data = np.genfromtxt( file, delimiter=",",
                         usecols=range(3,5), skip_header=0)
    return data[:,0],data[:,1]

def multi_lorentz_peak(w, Voff, Vlorentz, width, wresonance,Vlorentz2, width2, wresonance2):
    return (Voff+Vlorentz*width/((w-wresonance)**2+(width)**2)+Vlorentz2*width2/((w-wresonance2)**2+(width2)**2))

def multi_gaussian(x, Voff, height, width, center,height2, width2, center2):
    return Voff+height*np.exp(-(x - center)**2/(2*width**2))+height2*np.exp(-(x - center2)**2/(2*width2**2))  

t,channel1 = read_oscilloscope_data("data/ALL0003/F0003CH1.CSV") 
t,channel2 = read_oscilloscope_data("data/ALL0003/F0003CH2.CSV") 

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t,channel1)
ax2.plot(t,channel2,'orange')
ax1.set_xlabel("Time [s]")
ax2.set_ylabel("Voltage [V]")
ax1.set_ylabel("Voltage [V]")

xlower = 0.008
xupper = 0.013


#isolation and fit
v = channel1[(t>xlower)&(t<xupper)]
t1 = t[(t>xlower)&(t<xupper)]


p=[0,0.015,0.001,0.01,0.005,0.005,0.011];
popt, pcov = curve_fit(multi_lorentz_peak, t1, v,sigma = (0.04/256)*np.ones(len(t1)),p0=p,method="lm")
perr = np.sqrt(np.diag(pcov))
print("Multi lorentzian fit")
print(popt)
print(perr)

plt.figure()
plt.plot(t1,v,label="Experimental data")
plt.plot(t1, multi_lorentz_peak(t1, *popt), 'r-',label="Multi lorentzian fit")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()

center1 = popt[3]
center2 = popt[6]
dcenter1 = perr[3]
#calcolo finale
h = 209 #cm
f = 10 #cm
g = 7.8 #cm
deltaT = center2-center1
ddeltaT = np.sqrt(perr[3]**2 + perr[6]**2)
deltax = 2.0 #cm
ddeltax = 0.3 #cm
dtheta = deltax/(f+h)
angolomagico = alpha + np.arccos((f**2+b**2-g**2)/(2*f*b))
dangolomagico =np.sqrt(dalpha**2 + error_arccos(f,b,g)**2)

dfrequenecy = (299792458/((wavelength*1e-9)**2))*((dgrating*1e-9)/2)*(np.cos(angolomagico))*dtheta #fatto di conversione a quanto pare

#error analysis
light = 299792458
wavelength = wavelength*1e-9
dgrating = dgrating*1e-9
deltax = deltax * 1e-2
ddeltax = ddeltax * 1e-2
dwavelenght = dwavelenght*1e-9
dofx =  ddeltax*(light*dgrating*np.cos(angolomagico))/(wavelength**2*2*(f + h))
dofangolomagico = dangolomagico*(-(deltax*light*dgrating*np.sin(angolomagico))/(wavelength**2*2*(f + h)))
dofwalength = dwavelenght*(-(deltax*light*dgrating*np.cos(angolomagico))/(wavelength**3*(f + h)))
ddfrequnecy = np.sqrt(dofx**2 + dofangolomagico**2+dofwalength**2)

print("Frequency shift = "+str(dfrequenecy*1e-9) + " +/- " + str(ddfrequnecy*1e-9)) #gigahertz!
#altro picco
#isolation and fit
xlower = 0.0135
xupper = 0.0185

v = channel1[(t>xlower)&(t<xupper)]
t2 = t[(t>xlower)&(t<xupper)]


p=[-8.17563735e-03,1.38289037e-05,5.53220058e-04,0.0155,3.46658707e-06,3.19087018e-04,0.0165];
popt, pcov = curve_fit(multi_lorentz_peak, t2, v,sigma = (0.04/256)*np.ones(len(t2)),p0=p,method="lm",maxfev=5000)
perr = np.sqrt(np.diag(pcov))
print("Multi lorentzian fit")
print(popt)
print(perr)

plt.figure()
plt.plot(t2,v,label="Experimental data")
plt.plot(t2, multi_lorentz_peak(t2, *popt), 'r-',label="Multi lorentzian fit")

plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()
FPRpeak1 = center1
FPRpeak2 = popt[3]

dFSR = np.sqrt(dcenter1**2 + perr[3]**2)

frequency_fsr = dfrequenecy
time_fsr = deltaT
def t2freq(t):
	return (frequency_fsr/time_fsr)*t*1e-9 #gigaherz

FSR = t2freq(FPRpeak2-FPRpeak1)
dFSR_REAL = FSR*np.sqrt( (ddeltaT/deltaT)**2 + (ddfrequnecy/dfrequenecy)**2 + (dFSR/(FPRpeak2-FPRpeak1))**2  )
print(str(FSR)+"+/-" + str(dFSR_REAL))

plt.show() 
