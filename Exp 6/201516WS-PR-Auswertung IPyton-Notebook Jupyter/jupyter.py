
# coding: utf-8

# # including some libraries

# In[1]:


import os
import sys
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
import pandas as pd
import csv
import scipy.constants as consts

from bokeh.plotting import figure, output_file, output_notebook, show

from plotly.graph_objs import *
from scipy.optimize import curve_fit,fsolve
from scipy import loadtxt
from scipy.special import jv
from uncertainties import ufloat
from uncertainties.umath import *
from math import *
#%matplotlib inline


# # Fit  Functions

# In[2]:


def func_gaus(x, FWHM, h, x0, const):
    return const + h/(1 + 4*((x-x0)/FWHM)**2)
def func_lorentz_peak (w, Voff, Vlorentz, width, wresonance,
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


# # Functions to improve plot and fit quality

# In[3]:


def trim_data(xValues,yValues,x_min,x_max):
    #determine range indices - only works if the data is sorted ascending!!!
    #shift the maximum in the x-axis to 0
    
    i_min=0
    i_max=len(xValues)
    
    for i in range(len(xValues)):
        if xValues[i]<x_min:
            i_min=i
        if xValues[i]<x_max:
            i_max=i
    xValues=xValues[i_min:i_max]
    yValues=yValues[i_min:i_max]
    return xValues, yValues


def trim_data_yerr(xValues,yValues,yerr,x_min,x_max):
    #determine range indices - only works if the data is sorted ascending!!!
    #shift the maximum in the x-axis to 0
    
    i_min=0
    i_max=len(xValues)
    
    for i in range(len(xValues)):
        if xValues[i]<x_min:
            i_min=i
        if xValues[i]<x_max:
            i_max=i
    xValues=xValues[i_min:i_max]
    yValues=yValues[i_min:i_max]
    yerr=yerr[i_min:i_max]
    return xValues, yValues, yerr


def shift_ymax_to_x_zero(xValues,yValues):
    #find the maximum in y
    m_index=yValues.index(max(yValues))
    time_from_max=xValues[m_index]
    for i in range(len(xValues)):
        xValues[i]-=time_from_max
    return xValues, yValues

def shift_ymin_to_x_zero(xValues,yValues):
    #find the maximum in y
    m_index=yValues.index(min(yValues))
    time_from_max=xValues[m_index]
    for i in range(len(xValues)):
        xValues[i]-=time_from_max
    return xValues, yValues


# #Free Spectral Range (FSR) der Cavity

# In[4]:


L = ufloat(0.1,0.0001) #resonator_length
FSR_hertz=consts.c/(2*L) #FSR der Cavity in Hertz
print(FSR_hertz)


# #Übersicht der D2 Linie mit Feinstruktur plotten

# ##Daten einlesen

# In[25]:


mess_ordner=os.getcwd()+"\\mess\\"
messung="ALL0007 übersicht ohne sättigung\\"
piezo = mess_ordner + messung + "F0007CH1.csv"
probe = mess_ordner + messung + "F0007CH3.csv"
FPI = mess_ordner + messung + "F0007CH4.csv"

#read the files
a1, y1 = np.genfromtxt(piezo, delimiter=",", usecols=(3,4), unpack=True)
t2, y2 = np.genfromtxt(probe, delimiter=",", usecols=(3,4), unpack=True)
t3, y3 = np.genfromtxt(FPI, delimiter=",", usecols=(3,4), unpack=True)


#convert the returned nd-arrays to standard python lists
t1=a1.tolist()
y1=y2.tolist()
y2=y2.tolist()
y3=y3.tolist()

messung="ALL0008 übersicht mit sättigung\\"
piezo = mess_ordner + messung + "F0008CH1.csv"
probe = mess_ordner + messung + "F0008CH3.csv"
FPI = mess_ordner + messung + "F0008CH4.csv"

#read the files
tt1, yy1 = np.genfromtxt(piezo, delimiter=",", usecols=(3,4), unpack=True)
tt2, yy2 = np.genfromtxt(probe, delimiter=",", usecols=(3,4), unpack=True)
tt3, yy3 = np.genfromtxt(FPI, delimiter=",", usecols=(3,4), unpack=True)

#convert the returned nd-arrays to standard python lists
tt1=tt1.tolist()
yy1=yy2.tolist()
yy2=yy2.tolist()
yy3=yy3.tolist()


# ##Zeitachse zu Frequenz umrechnen
# 1. die am weitesten auseinander befindlichen Peaks im FPI spektrum mit Gauskurven fitten um deren Positoin zu bestimmen
# 2. die Zeit pro FSR berechnen
# 3. über die Länge des FPI die FSR in Hertz berechnen
# 4. über die FSR in Hertz die Zeitachse zu einer Frequenzachse umrechnen

# In[26]:


t_l,y_l=trim_data(t1,y3,x_min=-0.00096,x_max=-0.00088)
t_r,y_r=trim_data(t1,y3,x_min=0.00326,x_max=0.00332)

if False:
    #calculate guess curve
    xValues_guess=np.linspace(min(t1), max(t1), num=1000)
    yValues_guess=func_gaus(xValues_guess, *[0.00001, 4,-0.00092,0])
    plt.plot(xValues_guess, yValues_guess)
    plt.plot(t1,y3)
    plt.show()

############################################################################################################################                        
#fit the data                   p0=[FWHM, h, x0, const]
popt_l, pcov_l = curve_fit(func_gaus, t_l, y_l, p0=[0.00001, 4,-0.00092,0])
perr_l = np.sqrt(np.diag(pcov_l))
popt_r, pcov_r = curve_fit(func_gaus, t_r, y_r, p0=[0.00001, 4, 0.00328,0])
perr_r = np.sqrt(np.diag(pcov_r))
#popt : array
#       Optimal values for the parameters so that the sum of the squared error
#       of f(xdata, *popt) - ydata is minimized
#pcov : 2d array
#       The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
#       To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))


#fit the data
t_l=ufloat(popt_l[2],perr_l[2])
t_r=ufloat(popt_r[2],perr_r[2])

FSR_seconds=(t_r-t_l)/6
                
print("zeit pro FSR = ", FSR_seconds)  # durch 6 teilen, weil zwischen dem Linken und dem rechten Peak 6FSR liegen


# In[27]:


umrechnungsfaktor=FSR_hertz/FSR_seconds
f1=np.multiply(t1,umrechnungsfaktor.nominal_value) #nach dieser umrechnung erhält man t in der einheit von delta_nu.
Hyperfeinstruktur=np.subtract(yy2,y1)

#daten in die Zwischenablage kopieren, zum Einfügen in ein anderes Plot Programm
d = {'verbreitert': y1, 'dopplerfrei': yy2,'FPI':y3, 'Hyperfeinstruktur':np.subtract(yy2,y1)}
df = pd.DataFrame(data=d, index=f1)
df.to_clipboard()


# In[28]:


plt.plot(t1,y1, label='doppler verbreitert')
plt.plot(t1,yy2, label='doppler frei')
#plt.plot(t1,y3, label='FPI')
plt.plot(t1,Hyperfeinstruktur, label='Hyperfeinstruktur')
plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid=True
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()


# In[29]:


plt.plot(f1,y1, label='doppler verbreitert')
plt.plot(f1,yy2, label='doppler frei')
plt.plot(f1,y3, label='FPI')
plt.plot(f1,Hyperfeinstruktur, label='Hyperfeinstruktur')
plt.xlabel('frequency (Hz)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid=True
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()


# #Hyperfeinstruktur plotten

# In[30]:


messung="ALL0009 erste res ohne sätt\\"
piezo = mess_ordner + messung + "F0009CH1.csv"
probe = mess_ordner + messung + "F0009CH3.csv"
FPI = mess_ordner + messung + "F0009CH4.csv"

#read the files
t1, y1 = np.genfromtxt(piezo, delimiter=",", usecols=(3,4), unpack=True)
t2, y2 = np.genfromtxt(probe, delimiter=",", usecols=(3,4), unpack=True)
t3, y3 = np.genfromtxt(FPI, delimiter=",", usecols=(3,4), unpack=True)

#convert the returned nd-arrays to standard python lists
t1=t1.tolist()
y1=y2.tolist()
y2=y2.tolist()
y3=y3.tolist()



messung="ALL0010 erste res mit sätt\\"
piezo = mess_ordner + messung + "F0010CH1.csv"
probe = mess_ordner + messung + "F0010CH3.csv"
FPI = mess_ordner + messung + "F0010CH4.csv"

#read the files
tt1, yy1 = np.genfromtxt(piezo, delimiter=",", usecols=(3,4), unpack=True)
tt2, yy2 = np.genfromtxt(probe, delimiter=",", usecols=(3,4), unpack=True)
tt3, yy3 = np.genfromtxt(FPI, delimiter=",", usecols=(3,4), unpack=True)

#convert the returned nd-arrays to standard python lists
tt1=tt1.tolist()
yy1=yy2.tolist()
yy2=yy2.tolist()
yy3=yy3.tolist()




messung="ALL0011 erste res kalib\\"
piezo = mess_ordner + messung + "F0011CH1.csv"
probe = mess_ordner + messung + "F0011CH3.csv"
FPI = mess_ordner + messung + "F0011CH4.csv"

#read the files
ttt1, yyy1 = np.genfromtxt(piezo, delimiter=",", usecols=(3,4), unpack=True)
ttt2, yyy2 = np.genfromtxt(probe, delimiter=",", usecols=(3,4), unpack=True)
ttt3, yyy3 = np.genfromtxt(FPI, delimiter=",", usecols=(3,4), unpack=True)

#convert the returned nd-arrays to standard python lists
ttt1=ttt1.tolist()
yyy1=yyy2.tolist()
yyy2=yyy2.tolist()
yyy3=yyy3.tolist()


# In[31]:


t_l,y_l=trim_data(ttt1,yyy3,x_min=-0.00037,x_max=-0.00005)
t_r,y_r=trim_data(ttt1,yyy3,x_min=0.00037,x_max=0.00057)

if False:
    #calculate guess curve
    xValues_guess=np.linspace(min(ttt1), max(ttt1), num=1000)
    yValues_guess=func_gaus(xValues_guess, *[0.00001, 4, 0.000537,0])
    plt.plot(xValues_guess, yValues_guess)
    plt.plot(ttt1,yyy3)
    plt.show()

############################################################################################################################                        
#fit the data                   p0=[FWHM, h, x0, const]
popt_l, pcov_l = curve_fit(func_gaus, t_l, y_l, p0=[0.00001, 4,-0.0002,0])
perr_l = np.sqrt(np.diag(pcov_l))
popt_r, pcov_r = curve_fit(func_gaus, t_r, y_r, p0=[0.00001, 4, 0.000537,0])
perr_r = np.sqrt(np.diag(pcov_r))
#popt : array
#       Optimal values for the parameters so that the sum of the squared error
#       of f(xdata, *popt) - ydata is minimized
#pcov : 2d array
#       The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
#       To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))


#fit the data
t_l=ufloat(popt_l[2],perr_l[2])
t_r=ufloat(popt_r[2],perr_r[2])

FSR_seconds=(t_r-t_l)
                
print("zeit pro FSR = ", FSR_seconds)  # durch 6 teilen, weil zwischen dem Linken und dem rechten Peak 6FSR liegen


# ###Zeitachse skalieren

# In[32]:


umrechnungsfaktor=FSR_hertz/FSR_seconds
f1=np.multiply(t1,umrechnungsfaktor.nominal_value) #nach dieser umrechnung erhält man t in der einheit von delta_nu.


# ##Hintergrund von Spektrum entfernen

# In[33]:


spektrum=np.subtract(y1,yy2)


# ##Spektrum Plotten

# In[34]:


plt.plot(t1,y1, label='doppler verbreitert')
plt.plot(t1,yy2, label='doppler frei')
#plt.plot(t1,y3, label='FPI')
plt.plot(t1,spektrum, label='Hyperfeinstruktur')
plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid=True
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()


# In[35]:


plt.plot(f1,y1, label='doppler verbreitert')
plt.plot(f1,yy2, label='doppler frei')
#plt.plot(f1,y3, label='FPI')
plt.plot(f1,spektrum, label='Hyperfeinstruktur')
plt.xlabel('frequency (Hz)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid=True
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()


# ##peak fitting

# In[36]:


#t_l,y_l=trim_data(t1,y3,x_min=-0.00096,x_max=-0.00088)
#t_r,y_r=trim_data(t1,y3,x_min=0.00326,x_max=0.00332)

#func_lorentz_peak (Voff, Vlorentz, width, wresonance)
'''p=[0,1,0.00002,-0.00040,
    1,0.00002,-0.00036,
    1,0.00002,-0.00032,
    1,0.00002,-0.00029,
    1,0.00002,-0.00025,
    1,0.00002,-0.00019]'''

p=[-0.01,0.0000002,0.00001,-0.00040,
    0.0000006,0.00001,-0.00036,
    0.0000004,0.00001,-0.00032,
    0.0000010,0.00001,-0.00029,
    0.0000016,0.00001,-0.00025,
    0.0000002,0.00001,-0.00019]

t1,spektrum=trim_data(t1,spektrum,min(t1),-0.00015)
    
if False:
    #calculate guess curve
    xValues_guess=np.linspace(min(t1), max(t1), num=1000)
    yValues_guess=func_lorentz_peak(xValues_guess, *p)
    plt.plot(xValues_guess, yValues_guess)
    plt.plot(t1,spektrum)
    plt.show()


# In[37]:


############################################################################################################################                        
#fit the data
popt, pcov = curve_fit(func_lorentz_peak, t1, spektrum, p0=p)
perr = np.sqrt(np.diag(pcov))
#popt : array
#       Optimal values for the parameters so that the sum of the squared error
#       of f(xdata, *popt) - ydata is minimized
#pcov : 2d array
#       The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
#       To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))


# In[38]:


#Derived Person's Chi Squared Value For This Model --> https://en.wikipedia.org/wiki/Goodness_of_fit?oldformat=true#Pearson.27s_chi-squared_test
chi_squared = np.sum(((func_lorentz_peak(t1, *popt) - spektrum) ** 2)/spektrum) #funktioniert in diesem Fall nicht, weil die Variable Spektrum = 0 enthält und bei der division durch 0 "inf" auftritt

reduced_chi_squared = chi_squared / (len(t1) - len(popt)-1)

print(chi_squared, reduced_chi_squared)


# In[39]:


#calculate fitted curve
xValues_guess=np.linspace(min(t1), max(t1), num=len(spektrum))
yValues_guess=func_lorentz_peak(xValues_guess, *popt)
#daten in die Zwischenablage kopieren, zum Einfügen in ein anderes Plot Programm
d = {'fitx': xValues_guess*umrechnungsfaktor.nominal_value,
     'fity': yValues_guess,
     'frequ':np.multiply(t1,umrechnungsfaktor.nominal_value),
     'daten': spektrum}
df = pd.DataFrame(data=d)
df.to_clipboard()
if False:
    plt.plot(xValues_guess*umrechnungsfaktor.nominal_value, yValues_guess)
    plt.plot(np.multiply(t1,umrechnungsfaktor.nominal_value),spektrum)
    plt.show()
    
    
    # output to static HTML file
    output_file("plot_spektrum_fitted.html")
    # create a new plot
    p = figure()    
    # add some renderers    
    p.line(np.multiply(t1,umrechnungsfaktor.nominal_value), spektrum, legend="data")
    p.line(xValues_guess*umrechnungsfaktor.nominal_value, yValues_guess, legend="fit")
    # show the results
    show(p)
    



#fit the data
i=ufloat(popt[3],perr[3])*umrechnungsfaktor
h=ufloat(popt[6],perr[6])*umrechnungsfaktor
g=ufloat(popt[9],perr[9])*umrechnungsfaktor
f=ufloat(popt[12],perr[12])*umrechnungsfaktor
e=ufloat(popt[15],perr[15])*umrechnungsfaktor
d=ufloat(popt[18],perr[18])*umrechnungsfaktor

wi=ufloat(popt[2],perr[2])*umrechnungsfaktor/6e6*2
wh=ufloat(popt[5],perr[5])*umrechnungsfaktor/6e6*2
wg=ufloat(popt[8],perr[8])*umrechnungsfaktor/6e6*2
wf=ufloat(popt[11],perr[11])*umrechnungsfaktor/6e6*2
we=ufloat(popt[14],perr[14])*umrechnungsfaktor/6e6*2
wd=ufloat(popt[17],perr[17])*umrechnungsfaktor/6e6*2


# In[40]:


print (wi,wh,wg,wf,we,wd)


# In[41]:


delta_ig=abs(i-g)
delta_gd=abs(g-d)
delta_id=abs(i-d)
print(delta_ig,delta_gd,delta_id)


# ##Funktionen zur Berechnung der Feinstrukturaufspaltung
# Formeln für die Feinstrukturkonstanten von Rb87 (Kernspin I=3/2) im 5^2P_{3/2}  Zustand (J=3/2 .. gekennzeichnet durch _{J})
# 
# ->I=3/2
# 
# ->J=3/2

# In[42]:


#\delta nu = A/2 * K + B * bruch

def K(F):
    J=3/2
    I=3/2
    return (F*(F+1)-J*(J+1)-(I*(I+1)))
def bruch(F):
    J=3/2
    I=3/2
    return (3/4*K(F)*(K(F)+1)-I*(I+1)*J*(J+1)) / (2*I*(2*I-1)*J*(2*J-1))

def delta_bruch(f1,f2):
    return bruch(f2)-bruch(f1)

def delta_K(f1,f2):
    return K(f2)-K(f1)


# In[43]:


#Im Fall, dass aus den Zwischen-Peakabständen einer "6er Peakgruppe"
#die Hyperfeinstrukturkopplungskonstanten bestimmt werden soll
#ist von den Quantenzahlen I und J immer constant! --> Nur F ändert sich

b_ig=delta_bruch(3,2)
a_ig=delta_K(3,2) / 2

b_id=delta_bruch(3,1)
a_id=delta_K(3,1) / 2

b_gd=delta_bruch(2,1)
a_gd=delta_K(2,1) / 2

print("dieses Gleichungsystem wird in der nächsten Zelle gelöst:")
print("G->D: A",-a_gd,"+ ",-b_gd,"B = delta_gd")
print("I->D: A",-a_id,"+ ",b_id,"B = delta_id \n")

print("I->G: A",a_ig,"+",b_ig,"B = delta_ig  #diese Gleichung wird nicht verwendet.")


# In[44]:


#das hier liefert die RICHTIGEN Ergebnisse!!!!!
#es werden zufällig die richtigen Koeffizienten verwendet,
#obwohl die Peaks F=1,2,3 nicht mit der benennung i,g,d übereinstimmt
#tatsächlich wäre
#i:F=1
#g:F=2
#d:F=3
#deswegen werden die Koeefizienten in der oberen Zelle falsch zugewiesen,
#Sie sind aber in den unteren Beiden Formeln richtig eingesetzt!!

A=delta_id/a_id
B=(delta_id-delta_gd*a_ig/a_gd)/(b_id-b_gd*a_id/a_gd)
print("Kopplungskonstanten A und B in Hz:\n", A,B)

