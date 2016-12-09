import matplotlib.pyplot as plt
import numpy as np
import sys
data = np.loadtxt(sys.argv[1])
plt.plot(data[:,0],data[:,1],color='b',label='analysis error')
plt.plot(data[:,0],data[:,2],color='r',label='analysis spread')
plt.ylim(0,2)
plt.legend()
plt.xlabel('analysis cycles')
plt.ylabel('RMS error/spread')
plt.show()
