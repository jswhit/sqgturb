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
if len(sys.argv) > 3:
    ntimes = int(sys.argv[3])
    ntimes = ntimes+fcstlen
else:
    ntimes = -999

nc = Dataset(filenamein)
# initialize qg model instance
pv = nc['pv'][0]
dt = 600 # time step in seconds
norder = 8; diff_efold = 5400
stdev = 0.3e6
rp = RandomPattern(150.e3,2.*dt,nc.L,pv.shape[-1],dt=nc.dt,stdev=stdev,nsamples=2)
print 'random pattern hcorr,tcorr,stdev = ',rp.hcorr, rp.tcorr, rp.stdev
model = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            random_pattern=rp,
            dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=True,symmetric=bool(nc.symmetric),threads=threads,
            precision='single')
outputinterval = fcstlen*(nc['t'][1]-nc['t'][0])
model.timesteps = int(outputinterval/model.dt)
modeld.timesteps = int(outputinterval/model.dt)
scalefact = nc.f*nc.theta0/nc.g
if ntimes < 0:
    ntimes = len(nc.dimensions['t'])

nanals = 10
N = model.N
pverrsq_mean = np.zeros((2,N,N),np.float32)
pverrsqd_mean = np.zeros((2,N,N),np.float32)
pvspread_mean = np.zeros((2,N,N),np.float32)
pvens = np.zeros((nanals,2,N,N),np.float32)
kespec_errmean = None; kespec_sprdmean = None
for n in range(ntimes-fcstlen):
    for nanal in range(nanals):
        pvens[nanal] = model.advance(nc['pv'][n])
    pvfcstmean = pvens.mean(axis=0)
    pvfcstd = modeld.advance(nc['pv'][n])
    pvtruth = nc['pv'][n+fcstlen]
    pverrsq = (scalefact*(pvfcstmean - pvtruth))**2
    pverrsqd = (scalefact*(pvfcstd - pvtruth))**2
    pvspread = ((scalefact*(pvens-pvfcstmean))**2).sum(axis=0)/(nanals-1)
    print n,np.sqrt(pverrsq.mean()),np.sqrt(pverrsqd.mean()),np.sqrt(pvspread.mean())
    pvspread_mean += pvspread/(ntimes-fcstlen)
    pverrsq_mean += pverrsq/(ntimes-fcstlen)
    pverrsqd_mean += pverrsqd/(ntimes-fcstlen)

    pverrspec = scalefact*rfft2(pvfcstmean - pvtruth)
    psispec = model.invert(pverrspec)
    psispec = psispec/(model.N*np.sqrt(2.))
    kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
    if kespec_errmean is None:
        kespec_errmean =\
        (model.ksqlsq*(psispec*np.conjugate(psispec))).real/(ntimes-fcstlen)
    else:
        kespec_errmean = kespec_errmean + kespec/(ntimes-fcstlen)

    for nanal in range(nanals):
        pvsprdspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
        psispec = model.invert(pvsprdspec)
        psispec = psispec/(model.N*np.sqrt(2.))
        kespec = (model.ksqlsq*(psispec*np.conjugate(psispec))).real
        if kespec_sprdmean is None:
            kespec_sprdmean =\
            (model.ksqlsq*(psispec*np.conjugate(psispec))).real/(nanals*(ntimes-fcstlen))
        else:
            kespec_sprdmean = kespec_sprdmean+kespec/(nanals*(ntimes-fcstlen))

print 'mean',np.sqrt(pverrsq_mean.mean()),np.sqrt(pverrsqd_mean.mean()),np.sqrt(pvspread_mean.mean())
vmin = 0; vmax = 4
import matplotlib.pyplot as plt
im = plt.imshow(np.sqrt(pvspread_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('mean spread')
plt.figure()
im = plt.imshow(np.sqrt(pverrsq_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
plt.title('mean error')

k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
l = N*np.fft.fftfreq(N)
k,l = np.meshgrid(k,l)
ktot = np.sqrt(k**2+l**2)
ktotmax = (model.N/2)+1
kespec_err = np.zeros(ktotmax,np.float)
kespec_sprd = np.zeros(ktotmax,np.float)
for i in range(kespec_errmean.shape[2]):
    for j in range(kespec_errmean.shape[1]):
        totwavenum = ktot[j,i]
        if int(totwavenum) < ktotmax:
            kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] +\
            kespec_errmean[:,j,i].mean(axis=0)
            kespec_sprd[int(totwavenum)] = kespec_sprd[int(totwavenum)] +\
            kespec_sprdmean[:,j,i].mean(axis=0)

plt.figure()
wavenums = np.arange(ktotmax,dtype=np.float)
wavenums[0] = 1.
idealke = 2.*kespec_err[1]*wavenums**(-5./3,)
plt.loglog(wavenums,kespec_err,color='k')
plt.loglog(wavenums,kespec_sprd,color='b')
#plt.loglog(wavenums,idealke,color='r')
plt.title('error (black) and spread (blue) spectra for day %s' % fcstlen)

plt.show()
