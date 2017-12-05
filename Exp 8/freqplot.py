import numpy as np
import matplotlib.pyplot as plt



fdev = np.array([20,30,40,50,60,75,90,110,130,150,175,200])
for i in range(14,26):
	plt.figure(fdev[i-14])
	f, fft = np.loadtxt('data/DATA'+str(i)+'.CSV', delimiter = ',', skiprows = 1, unpack = True)	
	plt.plot(f,fft)
	plt.xlim(325e3,925e3)
	#plt.title(str(fdev[i-14]))
	for n in range(0,4):
		freq1 = 625e3 + n*50e3 
		freq2 = 625e3 - n*50e3 
		plt.axvline(x=freq1,linestyle ='--',color='#000000')
		plt.axvline(x=freq2,linestyle ='--',color='#000000')
		plt.text(freq1,-130,"n = " + str(n),color='red',fontsize=15,horizontalalignment='center')
		if n!=0:
			plt.text(freq2,-130,"n = -" + str(n),color='red',fontsize=15,horizontalalignment='center')
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("FFT [dBm]")

plt.show()