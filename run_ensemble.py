from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, rfft2, irfft2
from scipy.ndimage import gaussian_filter

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

verbose=False
if len(sys.argv) < 2:
    print 'python run_ensemble.py filenamein amp filename_inc'
    raise SystemExit

filenamein = sys.argv[1] # upscaled truth
amp = float(sys.argv[2]) # amplitude of additive noise
#amp = 0.
filename_inc = sys.argv[3] # analysis increments
#filename_inc = None
# smoothing applied to truth and forecasts.
#stdev = 1.-np.exp(-1)
stdev = None # no smoothing

diff_efold_ens=86400.  ; dt = 1200 #N64
#diff_efold_ens=86400./2.; dt = 600  #N128
#diff_efold_ens=86400./2.; dt = 240  #N256

diff_efold_det = diff_efold_ens
nanals = 20

nc = Dataset(filenamein)
scalefact = nc.f*nc.theta0/nc.g
# initialize qg model instance
pv = nc['pv'][0]
norder = 8
N = pv.shape[-1]
modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
             diff_order=norder,diff_efold=diff_efold_det,
             dealias=True,symmetric=bool(nc.symmetric),threads=threads,
             precision='single')
print '# amp, filename_inc = ',amp,filename_inc
fcstlenmax = 80
nfcsts = 200
#fcstlenmax = 24
#nfcsts = 20
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
pverrsq_mean = np.zeros((fcsttimes,2,N,N),np.float)
pverrsqd_mean = np.zeros((fcsttimes,2,N,N),np.float)
pvspread_mean = np.zeros((fcsttimes,2,N,N),np.float)
pvens = np.zeros((nanals,2,N,N),np.float)
kespec_errmean = np.zeros((fcsttimes,2,N,N/2+1),np.float)
kespec_sprdmean = np.zeros((fcsttimes,2,N,N/2+1),np.float)
nskip = 8
ntimes = nfcsts*nskip+fcstlenmax # for debugging
print '# ntimes,nfcsts,fcstlenmax,nskip = ',ntimes,nfcsts,fcstlenmax,nskip
ncount = len(range(0,ntimes-fcstlenmax,nskip))
print '# ',ncount,'forecasts',fcsttimes,'forecast times',forecast_timesteps,\
      'time steps for forecast interval'
print '# amp = %s' % amp

for n in range(0,ntimes-fcstlenmax,nskip):

    modeld = SQG(nc['pv'][n],nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
                 diff_order=norder,diff_efold=diff_efold_det,
                 dealias=True,symmetric=bool(nc.symmetric),threads=threads,
                 precision='single')
    models = []
    for nanal in range(nanals):
        models.append( SQG(nc['pv'][n],nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
                       diff_order=norder,diff_efold=diff_efold_ens,
                       ai_amp=amp,ai_filename=filename_inc,
                       dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
                       precision='single') )

    for nfcst in range(fcsttimes):
        fcstlen = (nfcst+1)*fcstleninterval
        for nt in range(forecast_timesteps):
            for nanal in range(nanals):
                models[nanal].timestep()
            modeld.timestep()
        for nanal in range(nanals):
            pvens[nanal] = irfft2(models[nanal].pvspec)

        pvfcstd = irfft2(modeld.pvspec)
        pvfcstmean = pvens.mean(axis=0)
        pvtruth = nc['pv'][n+fcstlen]

        # smooth truth and forecasts
        if stdev is not None:
           for k in range(2):
               pvtruth[k] = gaussian_filter(pvtruth[k],stdev,output=None,
                            order=0,mode='wrap', cval=0.0)
               pvfcstd[k] = gaussian_filter(pvfcstd[k],stdev,output=None,
                            order=0,mode='wrap', cval=0.0)
               for nanal in range(nanals):
                   pvens[nanal,k,...] = gaussian_filter(pvens[nanal,k,...],stdev,output=None,
                                        order=0,mode='wrap', cval=0.0)
           pvfcstmean = pvens.mean(axis=0)

        pverrsq = (scalefact*(pvfcstmean - pvtruth))**2
        pverrsqd = (scalefact*(pvfcstd - pvtruth))**2
        pvspread = ((scalefact*(pvens-pvfcstmean))**2).sum(axis=0)/(nanals-1)

        if verbose: print n,fcstlen,np.sqrt(pverrsq.mean()),np.sqrt(pverrsqd.mean()),np.sqrt(pvspread.mean())
        #if nfcst == 8:
        #   import matplotlib.pyplot as plt
        #   x = nc['x'][:]; y = nc['y'][:]
        #   err = scalefact*(pvtruth-pvfcstd)[1]
        #   errmax = max(np.abs(err.max()), np.abs(err.min()))
        #   print err.min(), err.max(), errmax
        #   plt.figure()
        #   plt.imshow(scalefact*pvtruth[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=-25,vmax=25)
        #   plt.title('truth')
        #   plt.axis('off')
        #   plt.figure()
        #   plt.imshow(scalefact*pvfcstd[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=-25,vmax=25)
        #   plt.title('fcst')
        #   plt.axis('off')
        #   plt.figure()
        #   plt.imshow(err,cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-errmax,vmax=errmax)
        #   plt.title('error')
        #   plt.axis('off')
        #   plt.show()
        #   raise SystemExit
        pvspread_mean[nfcst] += pvspread/ncount
        pverrsq_mean[nfcst] += pverrsq/ncount
        pverrsqd_mean[nfcst] += pverrsqd/ncount

        #if fcstlen in fcstlenspectra:
        #    pverrspec = scalefact*rfft2(pvfcstmean - pvtruth)
        #    psispec = modeld.invert(pverrspec)
        #    psispec = psispec/(modeld.N*np.sqrt(2.))
        #    kespec = (modeld.ksqlsq*(psispec*np.conjugate(psispec))).real
        #    kespec_errmean[nfcst] += kespec/ncount

        #    for nanal in range(nanals):
        #        pvsprdspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
        #        psispec = modeld.invert(pvsprdspec)
        #        psispec = psispec/(modeld.N*np.sqrt(2.))
        #        kespec = (modeld.ksqlsq*(psispec*np.conjugate(psispec))).real
        #        kespec_sprdmean[nfcst] += kespec/(nanals*ncount)

#print 'fcstlen = ',fcstlen, 'mean error =',np.sqrt(pverrsq_mean.mean()),np.sqrt(pverrsqd_mean.mean()),np.sqrt(pvspread_mean.mean())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for nfcst in range(fcsttimes):
    fcstlen = (nfcst+1)*fcstleninterval
    print fcstlen,np.sqrt(pverrsq_mean[nfcst].mean()),np.sqrt(pverrsqd_mean[nfcst].mean()),np.sqrt(pvspread_mean[nfcst].mean())
    #if fcstlen in fcstlenspectra:
    #    #vmin = 0; vmax = 4
    #    #im = plt.imshow(np.sqrt(pvspread_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    #    #plt.title('mean spread')
    #    #plt.figure()
    #    #im = plt.imshow(np.sqrt(pverrsq_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    #    #plt.title('mean error')

    #    k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
    #    l = N*np.fft.fftfreq(N)
    #    k,l = np.meshgrid(k,l)
    #    ktot = np.sqrt(k**2+l**2)
    #    ktotmax = N/2+1
    #    kespec_err = np.zeros(ktotmax,np.float)
    #    kespec_sprd = np.zeros(ktotmax,np.float)
    #    for i in range(kespec_errmean[nfcst].shape[2]):
    #        for j in range(kespec_errmean[nfcst].shape[1]):
    #            totwavenum = ktot[j,i]
    #            if int(totwavenum) < ktotmax:
    #                kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] +\
    #                kespec_errmean[nfcst,:,j,i].mean(axis=0)
    #                kespec_sprd[int(totwavenum)] = kespec_sprd[int(totwavenum)] +\
    #                kespec_sprdmean[nfcst,:,j,i].mean(axis=0)
    #    plt.figure()
    #    wavenums = np.arange(ktotmax,dtype=np.float)
    #    wavenums[0] = 1.
    #    idealke = 2.*kespec_err[1]*wavenums**(-5./3,)
    #    plt.loglog(wavenums,kespec_err,color='k')
    #    plt.loglog(wavenums,kespec_sprd,color='b')
    #    #plt.loglog(wavenums,idealke,color='r')
    #    plt.title('error (black) and spread (blue) spectra for hr %s' %\
    #            int(3*fcstlen),fontsize=12)
    #    plt.savefig('%sherr_spectrum_amp%s.png' % (3*fcstlen,amp))
