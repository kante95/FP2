import numpy as np 
import matplotlib.pyplot as plt 


#Characterization of the diode laser
input_current = np.array([3.06, 4.98, 6.97,8.97,11.06,12.96,15.03,17.09,19.07,20.99,23.03,25.08,27.01,16.02,18.04,20.03,19.50,18.55,20.48,19.82,21.97  ]) #unit mA
power = np.array([10.15, 13.78, 16.43,19.56,23.40,27.74,34.48,48.55,315.5,969.1,1506,1917,2201, 39.23,71.45,673.3, 486.3,140.1,824.7,600.7,1270       ])	#unit uW
error_power = np.array([0.48, 0.02, 0.05, 0.006,0.006,0.01,0.015,0.026,0.42,0.71,0.674,0.859,0.585,0.075,0.053,0.5, 0.333,0.285,0.549 ,0.69,1.25      ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Without feedback")
plt.xlabel("Input current [mA]")
plt.ylabel(r"Laser power [$\mu$W]")


#Littrow configuration
input_current = np.array([18.07,4.95,8.03,11.9,15.02,17.09,18.5,19,19.5,20,20.54,21.5,22,22.99, 24.04,25.03,26.08,27.03   ]) #unit mA
power = np.array([147.8,2.53,4.75,8.47,13.51,21.96,231.6,332.1,401.7,473.8,586.9,664.6,775.8,888.9, 1001,1111,1179,1200          ])	#unit uW
error_power = np.array([0.26,450e-6,930e-6,1.37e-3,2.31e-3,13e-3,6.74,0.31,0.521,1.095,1.344,1.134,2.185,1.256,5.089,0.49,1.065,0.408     ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Littrow configuration")
plt.xlabel("Input current [mA]")
plt.ylabel(r"Laser power [$\mu$W]")

plt.legend()

#littman configuration
plt.figure()

#with feedback
input_current = np.array([9.99,12.28,13.86,15.86, 17.97, 19.06,20.08,21,23.02,25.25,18.51 ,19.58    ]) #unit mA
power = np.array([6.766,9.297,11.59,16.20 ,34.18,200.6, 403.3,561.2,872.3,1109,83.21,308.8           ])	#unit uW
error_power = np.array([2.99e-3,1.32e-3,4.4e-3, 8.81e-3,25.8e-3,0.21,0.42,0.3,0.341,0.31,0.34,0.71 ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="With feddback")
plt.xlabel("Input current [mA]")
plt.ylabel(r"Laser power [$\mu$W]")

#without feedback

input_current = np.array([10.07,12,14.08,16.18, 17.99,20.04,19.023,23.10,24.96,22.06,20.48,18.54,19.5 ,21.55]) #unit mA
power = np.array([7, 9.05,12.05,16.89, 34.34,370.9, 167.2  ,852.5,1050,711.3,455.5,76.86,268,637.6    ])	#unit uW
error_power = np.array([162e-6,2.3e-3,1.8e-3,3.2e-3,38e-3,300e-3,330e-3,571e-3,469e-3,378e-3,241e-3,181e-3,294e-3,452e-3    ]) # unit uW

plt.errorbar(input_current,power,yerr = error_power,fmt='.',label="Whithout feddback")
plt.xlabel("Input current [mA]")
plt.ylabel(r"Laser power [$\mu$W]")
plt.legend()



plt.show() 
