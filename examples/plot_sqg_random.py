from sqgturb import SQGrandom, SQG, RandomPattern
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, os, cPickle

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

filenamein = sys.argv[1]

n = 0
fcstlen = 3

nc = Dataset(filenamein)
# initialize qg model instance

pv = nc['pv'][n]

modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=nc.dt,
            diff_order=nc.diff_order,diff_efold=nc.diff_efold,
            dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
scalefact = nc.f*nc.theta0/nc.g


stdev = 0.25e6
#rp = RandomPattern(200.e3,3.*nc.dt,nc.L,pv.shape[-1],dt=nc.dt,stdev=stdev,nsamples=2)
#f = open('rp.pickle','wb')
#cPickle.dump(rp, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#raise SystemExit
f = open('rp.pickle','rb')
rp = cPickle.load(f)
rp.stdev = 0.25e6
f.close()
#print rp.dt, rp.L, rp.N, rp.dt, rp.hcorr, rp.tcorr
#rp = None

modelr = SQGrandom(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=nc.dt,
            diff_order=nc.diff_order,diff_efold=nc.diff_efold,
            random_pattern=rp,
            dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
            precision='single')

outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
modeld.timesteps = int(outputinterval/modeld.dt)
modelr.timesteps = int(outputinterval/modelr.dt)

pvd = modeld.advance(pv=nc['pv'][n])
pvr = modelr.advance(pv=nc['pv'][n])
pvdiff_d = scalefact*(pvd-nc['pv'][n+fcstlen])
print pvdiff_d.min(), pvdiff_d.max()
pvdiff_r = scalefact*(pvr-nc['pv'][n+fcstlen])
print pvdiff_r.min(), pvdiff_r.max()
print np.sqrt((pvdiff_r**2).mean())

vmin = -25; vmax = 25.
plt.figure()
im = plt.imshow(scalefact*pvd[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('deterministic')
plt.figure()
im = plt.imshow(scalefact*pvr[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('stochastic')
plt.figure()
im = plt.imshow(pvdiff_r[1],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('difference')
plt.show()


nc.close()
