from netCDF4 import Dataset
import numpy as np
import sys, os
from sqg import SQG, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
filenamein = sys.argv[1]
fcstlen = int(sys.argv[2])

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][0]
dt = 600 # time step in seconds
norder = 8; diff_efold = 5400
model = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=True,symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
model.timesteps = int(outputinterval/model.dt)
scalefact = nc.f*nc.theta0/nc.g
ntimes = len(nc.dimensions['t'])
meanerr = 0.
for n in range(ntimes-fcstlen):
    model.advance(pv=nc['pv'][n])
    pvfcst = irfft2(model.pvspec)
    pvtruth = nc['pv'][n+fcstlen]
    pverr = scalefact*(pvfcst - pvtruth)
    err = np.sqrt((pverr**2).mean())
    meanerr += err/(ntimes-fcstlen)
    print n,np.sqrt((pverr**2).mean())
print 'mean = ',meanerr

#vmin = -10; vmax = 10
#import matplotlib.pyplot as plt
#im = plt.imshow(scalefact*pverr[1],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
#plt.show()
