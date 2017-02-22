from netCDF4 import Dataset
from sqgturb import rfft2, irfft2, SQG
import numpy as np
levplot = None
nc = Dataset('../examples/sqg_N512_N128_3hrly.nc')
nc2 = Dataset('../examples/sqg_N128_3hrly.nc')
nc3 = Dataset('../examples/sqg_N128_3hrly_hd.nc')
pv = nc.variables['pv']
pv2 = nc2.variables['pv']
pv3 = nc3.variables['pv']
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
kespecmean2 = None
kespecmean3 = None
ncount = 0
for nt,t in enumerate(time):
    pvgrd = pv[nt]
    pvspec = rfft2(pvgrd)
    if levplot is not None:
        psispec = model.invert(pvspec)[levplot]
    else:
        psispec = model.invert(pvspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespecmean is None:
        kespecmean = kespec
    else:
        kespecmean = kespecmean + kespec
    pvgrd = pv2[nt]
    pvspec = rfft2(pvgrd)
    if levplot is not None:
        psispec = model.invert(pvspec)[levplot]
    else:
        psispec = model.invert(pvspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec2 = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespecmean2 is None:
        kespecmean2 = kespec2
    else:
        kespecmean2 = kespecmean2 + kespec2
    pvgrd = pv3[nt]
    pvspec = rfft2(pvgrd)
    if levplot is not None:
        psispec = model.invert(pvspec)[levplot]
    else:
        psispec = model.invert(pvspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec3 = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespecmean3 is None:
        kespecmean3 = kespec3
    else:
        kespecmean3 = kespecmean3 + kespec3
    ncount += 1
    print ncount, kespec.mean(), kespec2.mean()
    #print t, kespec.mean(), (u**2+v**2).mean()
kespecmean = kespecmean/ncount
kespecmean2 = kespecmean2/ncount
kespecmean3 = kespecmean3/ncount
print kespecmean.mean(), kespecmean2.mean(), kespecmean3.mean(), kespecmean.shape
ktotmax = (model.N/2)+1
kespec = np.zeros(ktotmax,np.float)
kespec2 = np.zeros(ktotmax,np.float)
kespec3 = np.zeros(ktotmax,np.float)
if levplot is None:
    kespecmean = kespecmean.mean(axis=0)
    kespecmean2 = kespecmean2.mean(axis=0)
    kespecmean3 = kespecmean3.mean(axis=0)
nk = kespecmean.shape[-1]; nl = kespecmean.shape[-2]
for i in range(nk):
    for j in range(nl):
        totwavenum = ktot[j,i]
        if int(totwavenum) < ktotmax:
            kespec[int(totwavenum)] = kespec[int(totwavenum)] +\
            kespecmean[j,i]
            kespec2[int(totwavenum)] = kespec2[int(totwavenum)] +\
            kespecmean2[j,i]
            kespec3[int(totwavenum)] = kespec3[int(totwavenum)] +\
            kespecmean3[j,i]
wavenums = np.arange(ktotmax,dtype=np.float)
wavenums[0] = 1.
idealke1 = 2.*kespec[1]*wavenums**-3
idealke2 = 2.*kespec[1]*wavenums**(-5./3,)

for n in range(ktotmax):
    print n,kespec[n],kespec2[n]

import matplotlib.pyplot as plt
import numpy as np
plt.loglog(wavenums[:-1],kespec[:-1],color='b',label='N512')
plt.loglog(wavenums[:-1],kespec2[:-1],color='r',label='N128 (low diffusion)')
plt.loglog(wavenums[:-1],kespec3[:-1],color='c',label='N128 (higher diffusion)')
#plt.loglog(wavenums,idealke1,color='k')
plt.loglog(wavenums,idealke2,color='k')
plt.xlabel('total wavenumber')
plt.ylabel('kinetic energy')
plt.title('KE Spectrum 128 x 128 (black line is -5/3)')
plt.legend()
plt.show()
