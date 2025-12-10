import matplotlib
matplotlib.use('QTAgg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

nc = Dataset('sqgu20_N96_6hrly.nc')
pv = nc['pv'][-1,1,...]
ny, nx = pv.shape
nobs = 1024

rsobs = np.random.RandomState(42)
indxob = np.sort(rsobs.choice(2*nx*ny,nobs,replace=False))
x = nc.variables['x'][:]
y = nc.variables['y'][:]
x, y = np.meshgrid(x, y)
xobs = nx*np.concatenate((x.ravel(),x.ravel()))[indxob]/nc.L
yobs = ny*np.concatenate((y.ravel(),y.ravel()))[indxob]/nc.L

plt.imshow(pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower')
plt.scatter(xobs, yobs, s=5, color='black')
plt.axis('off')
plt.tight_layout()
plt.savefig('obnetwork.png')
plt.show()


