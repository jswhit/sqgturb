from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQGpert, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
filenamein = sys.argv[1]
#fcstlen = int(sys.argv[2])
fcstlen = 1

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][0]
dt = 600 # time step in seconds
norder = 8; diff_efold = 5400
norder_pert = 4; diff_efold_pert = diff_efold; pert_amp=200; pert_shift=8.0
pert_corr = 3.0; windpert_max=50
model = SQGpert(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            diff_order_pert=norder_pert,diff_efold_pert=diff_efold_pert,
            pert_shift=pert_shift,pert_amp=pert_amp,pert_corr=pert_corr,
            windpert_max=windpert_max,
            dealias=True,symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
model.timesteps = int(outputinterval/model.dt)
scalefact = nc.f*nc.theta0/nc.g
ntimes = len(nc.dimensions['t'])
N = pv.shape[1]
#meanerr = 0.
#for n in range(ntimes-fcstlen):
#    model.advance(pv=nc['pv'][n])
#    pvfcst = irfft2(model.pvspec)
#    pvtruth = nc['pv'][n+fcstlen]
#    pverr = scalefact*(pvfcst - pvtruth)
#    err = np.sqrt((pverr**2).mean())
#    meanerr += err/(ntimes-fcstlen)
#    print n,np.sqrt((pverr**2).mean())
#print 'mean = ',meanerr

nanals = 10
fcstlen = 1
pverrsq_mean = np.zeros((2,N,N),np.float32)
pvspread_mean = np.zeros((2,N,N),np.float32)
pvens = np.zeros((nanals,2,N,N),np.float32)
#ntimes = 10+fcstlen
for n in range(ntimes-fcstlen):
    for nanal in range(nanals):
        pvens[nanal] = model.advance(nc['pv'][n])
    pvfcstmean = pvens.mean(axis=0)
    pvtruth = nc['pv'][n+fcstlen]
    pverrsq = (scalefact*(pvfcstmean - pvtruth))**2
    pvspread = ((scalefact*(pvens-pvfcstmean))**2).sum(axis=0)/(nanals-1)
    print n,np.sqrt(pverrsq.mean()),np.sqrt(pvspread.mean())
    pvspread_mean += pvspread/(ntimes-fcstlen)
    pverrsq_mean += pverrsq/(ntimes-fcstlen)
print 'mean',np.sqrt(pverrsq_mean.mean()),np.sqrt(pvspread_mean.mean())
vmin = 0; vmax = 4
import matplotlib.pyplot as plt
im = plt.imshow(np.sqrt(pvspread_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('mean spread')
plt.figure()
im = plt.imshow(np.sqrt(pverrsq_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('mean error')
plt.show()
