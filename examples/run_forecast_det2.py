from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, RandomPattern, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

#filenamein = sys.argv[1]
#fcstlen = int(sys.argv[2])
#diff_efold = float(sys.argv[3])
diff_efold = 5400.
filenamein = 'sqg_N512_N128_3hrly_blockmean.nc'
fcstlen = int(sys.argv[1])
if len(sys.argv) > 2:
    ntimes = int(sys.argv[2])
    ntimes = ntimes+fcstlen
else:
    ntimes = -999

verbose = False

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][0]
dt = 600 # time step in seconds
norder = 8
model = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
model.timesteps = int(outputinterval/model.dt)
scalefact = nc.f*nc.theta0/nc.g
if ntimes < 0:
    ntimes = len(nc.dimensions['t'])

N = model.N
pverrsq_mean = np.zeros((2,N,N),np.float32)
kespec_errmean = None
for n in range(ntimes-fcstlen):
    pvfcst = model.advance(nc['pv'][n])
    pvtruth = nc['pv'][n+fcstlen]
    pverrsq = (scalefact*(pvfcst - pvtruth))**2
    if verbose: print n,np.sqrt(pverrsq.mean())
    pverrsq_mean += pverrsq/(ntimes-fcstlen)

    pverrspec = scalefact*rfft2(pvfcst - pvtruth)
    psispec = model.invert(pverrspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespec_errmean is None:
        kespec_errmean =\
        (model.ksqlsq*(psispec*np.conjugate(psispec))).real/(ntimes-fcstlen)
    else:
        kespec_errmean = kespec_errmean + kespec/(ntimes-fcstlen)

print 'fcstlen=',fcstlen,' mean error=',np.sqrt(pverrsq_mean.mean())

#vmin = 0; vmax = 4
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.figure()
#im = plt.imshow(np.sqrt(pverrsq_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
#plt.title('mean error')

k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
l = N*np.fft.fftfreq(N)
k,l = np.meshgrid(k,l)
ktot = np.sqrt(k**2+l**2)
ktotmax = (model.N/2)+1
kespec_err = np.zeros(ktotmax,np.float)
for i in range(kespec_errmean.shape[2]):
    for j in range(kespec_errmean.shape[1]):
        totwavenum = ktot[j,i]
        if int(totwavenum) < ktotmax:
            kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] +\
            kespec_errmean[:,j,i].mean(axis=0)

#plt.figure()
#wavenums = np.arange(ktotmax,dtype=np.float)
#wavenums[0] = 1.
#idealke = 2.*kespec_err[1]*wavenums**(-5./3,)
#plt.loglog(wavenums,kespec_err,color='k')
#plt.title('error spectrum for day %s' % fcstlen)
#plt.show()
