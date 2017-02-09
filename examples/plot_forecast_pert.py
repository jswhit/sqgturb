from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, SQGpert, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
filenamein = sys.argv[1]
#fcstlen = int(sys.argv[2])
n = 0
fcstlen = 4

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][n]
dt = 600 # time step in seconds
norder = 8; diff_efold = 5400
norder_pert = 2; diff_efold_pert = diff_efold; pert_amp=50; pert_shift=8.0
pert_corr = 0.0; windpert_max=1.e30
model = SQGpert(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            diff_order_pert=norder_pert,diff_efold_pert=diff_efold_pert,
            pert_shift=pert_shift,pert_amp=pert_amp,pert_corr=pert_corr,
            windpert_max=windpert_max,
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
pvdiff = pvfcst_s-pvfcst_d
print scalefact*pvdiff.min(), scalefact*pvdiff.max()
im = plt.imshow(scalefact*pvdiff[1],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('difference')
plt.show()
