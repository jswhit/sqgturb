import matplotlib
matplotlib.use('QTAgg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

nc = Dataset('sqgu20_N96_6hrly.nc')
ntimes, nlevs, ny, nx = nc['pv'].shape
nobs = 1024

rsobs = np.random.RandomState(42)
indxob = np.sort(rsobs.choice(2*nx*ny,nobs,replace=False))
x = nc.variables['x'][:]
y = nc.variables['y'][:]
x, y = np.meshgrid(x, y)
xobs = nx*np.concatenate((x.ravel(),x.ravel()))[indxob]/nc.L
yobs = ny*np.concatenate((y.ravel(),y.ravel()))[indxob]/nc.L

# just plot obs on lower boundary
pv = nc['pv'][-1,0,...] # last time, lower boundary
plt.imshow(pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower')
plt.scatter(xobs[:nobs//2], yobs[:nobs//2], s=5, color='black')

# just plot obs on upper boundary
#pv = nc['pv'][-1,1,...] # last time, upper boundary
#plt.imshow(pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower')
#plt.scatter(xobs[nobs//2:], yobs[nobs//2:], s=5, color='black')
plt.axis('off')
plt.tight_layout()
plt.savefig('obnetwork.png')
plt.show()


