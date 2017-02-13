from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, RandomPattern, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
filenamein = sys.argv[1]
fcstlen = int(sys.argv[2])
n = int(sys.argv[3])

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][n]
N = pv.shape[-1]
dt = 600 # time step in seconds
norder = 8; diff_efold = 5400
stdev = 0.25e6
rp = RandomPattern(0.5*nc.L/N,2*dt,nc.L,pv.shape[-1],dt=dt,stdev=stdev,nsamples=2)
print 'random pattern hcorr,tcorr,stdev = ',rp.hcorr, rp.tcorr, rp.stdev
model = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,random_pattern=rp,
            dealias=True,symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=True,symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
model.timesteps = int(outputinterval/model.dt)
modeld.timesteps = int(outputinterval/model.dt)
scalefact = nc.f*nc.theta0/nc.g
ntimes = len(nc.dimensions['t'])
N = pv.shape[1]

pvfcst_d = modeld.advance(nc['pv'][n])
pvfcst_s = model.advance(nc['pv'][n])
vmin = -25; vmax = 25.
import matplotlib.pyplot as plt
im = plt.imshow(scalefact*pvfcst_d[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('deterministic')
plt.figure()
im = plt.imshow(scalefact*pvfcst_s[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('stochastic')
plt.figure()
im = plt.imshow(scalefact*nc['pv'][n+fcstlen][1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('truth')
plt.figure()
pvdiff = pvfcst_d-nc['pv'][n+fcstlen]
minmax = scalefact*max(np.abs(pvdiff.min()), pvdiff.max())
vmin = -minmax; vmax = minmax
print scalefact*pvdiff.min(), scalefact*pvdiff.max()
im = plt.imshow(scalefact*pvdiff[1],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('deterministic error')
plt.figure()
pvdiff = pvfcst_s-pvfcst_d
vmin = 0.5*vmin; vmax = 0.5*vmax
print scalefact*pvdiff.min(), scalefact*pvdiff.max()
im = plt.imshow(scalefact*pvdiff[1],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('difference')
plt.show()
