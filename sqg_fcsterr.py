from sqg import SQG,rfft2,irfft2
import numpy as np
from netCDF4 import Dataset
import sys, time

filename_climo = "data/sqg_N64.nc"
filename_truth = "data/sqg_N256_N64.nc"
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
#diff_efold = nc_climo.diff_efold
fcstlag = int(sys.argv[1])
diff_efold = float(sys.argv[2])

model = SQG(pv_climo[0],\
    nsq=nc_climo.nsq,f=nc_climo.f,dt=nc_climo.dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold)

obtimes = nc_truth.variables['t'][:]
assim_interval = obtimes[fcstlag]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/model.dt))
model.timesteps = assim_timesteps

ntimes = len(obtimes) - 100
pv0err_mean = 0; pv1err_mean = 0; pverr_mean = 0
nfcst = 0
for ntime in xrange(ntimes):
    pvfcst = model.advance(pv_truth[ntime])
    pverr = pvfcst - pv_truth[ntime+fcstlag]
    pverrsq = (scalefact*pverr)**2
    pv0err = np.sqrt(pverrsq[0].mean())
    pv1err = np.sqrt(pverrsq[1].mean())
    pverr = np.sqrt(pverrsq.mean())
    pv0err_mean = pv0err_mean +  pv0err
    pv1err_mean = pv1err_mean +  pv1err
    pverr_mean = pverr_mean + pverr
    nfcst += 1
    #print ntime, pv0err, pv1err
print fcstlag,diff_efold,pv0err_mean/nfcst,pv1err_mean/nfcst,pverr_mean/nfcst
