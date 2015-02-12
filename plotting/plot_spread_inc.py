import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
filename = 'sqg_enkf_test.nc'
nc = Dataset(filename)
pv_a = nc.variables['pv_a'][-1]
pv_b = nc.variables['pv_b'][-1]
x_obs = nc.variables['x_obs'][-1]
y_obs = nc.variables['y_obs'][-1]
x = nc.variables['x'][:]
y = nc.variables['y'][:]
nanals,nlevs,nlats,nlons = pv_a.shape
print nanals,nlevs,nlats,nlons
analinc = (pv_a - pv_b).mean(axis=0)
sprd = np.sqrt(((pv_b-pv_b.mean(axis=0))**2).sum(axis=0)/(nanals-1))
print analinc.shape, analinc.min(), analinc.max()
print sprd.shape, sprd.min(), sprd.max()
fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.axis('off'); ax2.axis('off'); ax3.axis('off')
im1=ax1.imshow(pv_b.mean(axis=0)[0],interpolation='nearest',origin='lower',vmin=-30,vmax=30)
#levs = np.linspace(-30,30,21)
#cs1 = ax1.contourf(x,y,pv_b.mean(axis=0)[0],levs,cmap=plt.cm.jet)
im2=ax2.imshow(sprd[0],interpolation='nearest',origin='lower',cmap=plt.cm.hot_r,vmin=0,vmax=4)
#cs2 = ax2.contourf(x,y,sprd[0],21,cmap=plt.cm.hot_r)
#ax2.scatter(x_obs,y_obs,color='k',zorder=10)
im3=ax3.imshow(analinc[0],interpolation='nearest',origin='lower',vmin=-4,vmax=4)
#levs = np.linspace(-4,4,21)
#cs3 = ax3.contourf(x,y,analinc[0],levs,cmap=plt.cm.jet)
ax1.set_title('ens mean background')
ax2.set_title('spread')
ax3.set_title('increment')
plt.tight_layout()
plt.show()
nc.close()
