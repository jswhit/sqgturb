from netCDF4 import Dataset
from sqg import rfft2, irfft2, SQG
import numpy as np
nc = Dataset('data/sqg_N256_dealiased.nc')
print nc
pv = nc.variables['pv']
time = nc.variables['t'][:]/3600.
N = pv.shape[-1]
model =\
SQG(np.empty((2,N,(N/2)+1),np.complex64),U=nc.U,L=nc.L,H=nc.H,dt=nc.dt,r=nc.r,tdiab=nc.tdiab,diff_order=nc.diff_order,diff_efold=nc.diff_efold)
N = model.N
k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
l = N*np.fft.fftfreq(N)
k,l = np.meshgrid(k,l)
ktot = np.sqrt(k**2+l**2)
kespecmean = None
ncount = 0
for nt,t in enumerate(time):
    pvgrd = pv[nt]  - model.pvbar
    pvspec = rfft2(pvgrd)
    psispec = model.invert(pvspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespecmean is None:
        kespecmean = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    else:
        kespecmean = kespecmean + kespec
    ncount += 1
    print ncount, kespec.mean()
    #print t, kespec.mean(), (u**2+v**2).mean()
kespecmean = kespecmean/ncount
print kespecmean.mean(), kespecmean.shape
ktotmax = (model.N/2)+1
kespec = np.zeros(ktotmax,np.float)
for i in range(kespecmean.shape[2]):
    for j in range(kespecmean.shape[1]):
        totwavenum = ktot[j,i]
        if int(totwavenum) < ktotmax:
            kespec[int(totwavenum)] = kespec[int(totwavenum)] +\
            kespecmean[:,j,i].mean(axis=0)
wavenums = np.arange(ktotmax,dtype=np.float)
wavenums[0] = 1.
idealke1 = 2.*kespec[1]*wavenums**-3
idealke2 = 2.*kespec[1]*wavenums**(-5./3,)
nc.close()

import matplotlib.pyplot as plt
import numpy as np
plt.loglog(wavenums,kespec,color='b')
#plt.loglog(wavenums,idealke1,color='k')
plt.loglog(wavenums,idealke2,color='r')
plt.show()
