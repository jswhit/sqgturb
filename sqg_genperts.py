from sqg import SQG,rfft2,irfft2
import numpy as np
from netCDF4 import Dataset
import sys, time

filename_climo = "data/sqg_N64_symek_12000.nc"
filename_truth = "data/sqg_N256_N64_symek.nc"
filename = "data/N256_N64_symek_12000_errperts.nc"
# get model info
nc_climo = Dataset(filename_climo)
nc_truth = Dataset(filename_truth)
x = nc_climo.variables['x'][:]
y = nc_climo.variables['y'][:]
x, y = np.meshgrid(x, y)
nx = len(x); ny = len(y)
pv_climo = nc_climo.variables['pv']
pv_truth = nc_truth.variables['pv']
scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
diff_efold = nc_climo.diff_efold

model = SQG(pv_climo[0],\
    nsq=nc_climo.nsq,f=nc_climo.f,dt=nc_climo.dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold)

obtimes = nc_truth.variables['t'][:]
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/model.dt))
model.timesteps = assim_timesteps

nc = Dataset(filename, mode='w', format='NETCDF4_CLASSIC')
nc.r = model.r
nc.f = model.f
nc.U = model.U
nc.L = model.L
nc.H = model.H
nc.g = nc_climo.g; nc.theta0 = nc_climo.theta0
nc.nsq = model.nsq
nc.tdiab = model.tdiab
nc.dt = model.dt
nc.diff_efold = model.diff_efold
nc.diff_order = model.diff_order
x = nc.createDimension('x',model.N)
y = nc.createDimension('y',model.N)
z = nc.createDimension('z',2)
t = nc.createDimension('t',None)
pvvar =\
nc.createVariable('pv',np.float32,('t','z','y','x'),zlib=True)
pvvar.units = 'K'
# eady pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
xvar = nc.createVariable('x',np.float32,('x',))
xvar.units = 'meters'
yvar = nc.createVariable('y',np.float32,('y',))
yvar.units = 'meters'
zvar = nc.createVariable('z',np.float32,('z',))
zvar.units = 'meters'
tvar = nc.createVariable('t',np.float32,('t',))
tvar.units = 'seconds'
xvar[:] = np.arange(0,model.L,model.L/model.N)
yvar[:] = np.arange(0,model.L,model.L/model.N)
zvar[0] = 0; zvar[1] = model.H

ntimes = len(obtimes)
nout = 0
levplot = 1
N = model.N
k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
l = N*np.fft.fftfreq(N)
k,l = np.meshgrid(k,l)
ktot = np.sqrt(k**2+l**2)
kespecmean = 0.
for ntime in xrange(ntimes-1):
    model.t = obtimes[ntime]
    pvfcst = model.advance(pv_truth[ntime])
    pverr = pvfcst - pv_truth[ntime+1]
    pvvar[nout,:,:,:] = pverr
    tvar[nout] = model.t
    nc.sync()
    nout = nout + 1
    print scalefact*pverr.min(), scalefact*pverr.max(), pverr.shape
    pvspec = rfft2(pverr)
    psispec = model.invert(pvspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    kespecmean = kespecmean + kespec/(ntimes-1)
    #import matplotlib.pyplot as plt
    #levs = np.linspace(-2,2,21)
    #plt.contourf(xvar[:],yvar[:],scalefact*pverr[levplot],levs)
    #plt.axis('off')
    #plt.title('actual error')
    #plt.axis('off')
    #plt.show()
    #raise SystemExit
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

import matplotlib.pyplot as plt
import numpy as np
plt.loglog(wavenums,kespec,color='k')
plt.xlim(0,32)
plt.show()
