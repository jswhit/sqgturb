from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, rfft2, irfft2
from scipy.ndimage import gaussian_filter

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

verbose=False
if len(sys.argv) < 1:
    print 'python run_ensemble.py filenamein'
    raise SystemExit

filenamein = sys.argv[1] # upscaled truth
# smoothing applied to truth and forecasts.
#stdev = 1.-np.exp(-1)
stdev = None # no smoothing

nc = Dataset(filenamein)
scalefact = nc.f*nc.theta0/nc.g
# initialize qg model instance
pv = nc['pv'][0]
norder = 8
N = pv.shape[-1]
if N == 64:
    diff_efold=86400.  ; dt = 1200 #N64
else:
    diff_efold=86400./2.; dt = 600  #N128
print '# diff_efold = ',diff_efold
modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
             diff_order=norder,diff_efold=diff_efold,
             dealias=True,symmetric=bool(nc.symmetric),threads=threads,
             precision='single')
fcstlenmax = 80
#fcstlenmax = 24
nfcsts = 200
fcstleninterval = 1
fcstlenspectra = [1,4,8,24,48,80]
fcsttimes = fcstlenmax/fcstleninterval
outputinterval = fcstleninterval*(nc['t'][1]-nc['t'][0])
print '# output interval = ',outputinterval
forecast_timesteps = int(outputinterval/modeld.dt)
modeld.timesteps = int(outputinterval/modeld.dt)
scalefact = nc.f*nc.theta0/nc.g
ntimes = len(nc.dimensions['t'])

N = modeld.N
pverrsqd_mean = np.zeros((fcsttimes,2,N,N),np.float)
nskip = 8
ntimes = nfcsts*nskip+fcstlenmax # for debugging
print '# ntimes,nfcsts,fcstlenmax,nskip = ',ntimes,nfcsts,fcstlenmax,nskip
ncount = len(range(0,ntimes-fcstlenmax,nskip))
print '# ',ncount,'forecasts',fcsttimes,'forecast times',forecast_timesteps,\
      'time steps for forecast interval'

for n in range(0,ntimes-fcstlenmax,nskip):

    modeld = SQG(nc['pv'][n],nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
                 diff_order=norder,diff_efold=diff_efold,
                 dealias=True,symmetric=bool(nc.symmetric),threads=threads,
                 precision='single')

    for nfcst in range(fcsttimes):
        fcstlen = (nfcst+1)*fcstleninterval
        for nt in range(forecast_timesteps):
            modeld.timestep()

        pvfcstd = irfft2(modeld.pvspec)
        pvtruth = nc['pv'][n+fcstlen]

        # smooth truth and forecasts
        if stdev is not None:
           for k in range(2):
               pvtruth[k] = gaussian_filter(pvtruth[k],stdev,output=None,
                            order=0,mode='wrap', cval=0.0)
               pvfcstd[k] = gaussian_filter(pvfcstd[k],stdev,output=None,
                            order=0,mode='wrap', cval=0.0)

        pverrsqd = (scalefact*(pvfcstd - pvtruth))**2

        if verbose: print n,fcstlen,np.sqrt(pverrsqd.mean())
        pverrsqd_mean[nfcst] += pverrsqd/ncount

for nfcst in range(fcsttimes):
    fcstlen = (nfcst+1)*fcstleninterval
    print fcstlen,np.sqrt(pverrsqd_mean[nfcst].mean())
