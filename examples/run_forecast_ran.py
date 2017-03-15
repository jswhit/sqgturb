from netCDF4 import Dataset
import numpy as np
import sys, os
from sqgturb import SQG, RandomPattern, rfft2, irfft2

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
verbose=False

filenamein = sys.argv[1]
amp = float(sys.argv[2])
hcorr = float(sys.argv[3])
tcorr = float(sys.argv[4])
norm = sys.argv[5]

diff_efold_ens=86400./2.
diff_efold_det = diff_efold_ens
nsamples = 2
nanals = 10

nc = Dataset(filenamein)
scalefact = nc.f*nc.theta0/nc.g
# initialize qg model instance
pv = nc['pv'][0]
dt = 600. # time step in seconds
norder = 8
N = pv.shape[-1]
models = []
for nanal in range(nanals):
    if norm == 'pv':
        stdev= amp/scalefact # amp given in units of K (for psi units are m**2/s)
    elif norm == 'psi':
        stdev = amp # psi units are m**2/s
    else:
        raise ValueError('illegal random pattern norm')
    rp = RandomPattern(hcorr*nc.L/N,tcorr*dt,nc.L,N,dt,nsamples=nsamples,stdev=stdev,norm=norm)
    models.append( SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
                diff_order=norder,diff_efold=diff_efold_ens,
                random_pattern=rp.copy(seed=nanal),
                dealias=bool(nc.dealias),symmetric=bool(nc.symmetric),threads=threads,
                precision='single') )
modeld = SQG(pv,nsq=nc.nsq,f=nc.f,U=nc.U,H=nc.H,r=nc.r,tdiab=nc.tdiab,dt=dt,
             diff_order=norder,diff_efold=diff_efold_det,
             dealias=True,symmetric=bool(nc.symmetric),threads=threads,
             precision='single')
print '# random pattern amp,hcorr,tcorr,norm,nsamples = ',amp, \
hcorr,tcorr,rp.norm,rp.nsamples
fcstlenmax = 80
fcstleninterval = 4
fcstlenspectra = [4,16,40,80]
fcsttimes = fcstlenmax/fcstleninterval
outputinterval = fcstleninterval*(nc['t'][1]-nc['t'][0])
forecast_timesteps = int(outputinterval/models[nanal].dt)
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
#ntimes = 120 # for debuggin
ncount = len(range(0,ntimes-fcstlenmax,16))
print '# ',ncount,'forecasts',fcsttimes,'forecast times',forecast_timesteps,\
      'time steps for forecast interval'

for n in range(0,ntimes-fcstlenmax,16):
    pvspecic = rfft2(nc['pv'][n])
    for nanal in range(nanals):
        models[nanal].pvspec = pvspecic
    modeld.pvspec = pvspecic
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
        pverrsq = (scalefact*(pvfcstmean - pvtruth))**2
        pverrsqd = (scalefact*(pvfcstd - pvtruth))**2
        pvspread = ((scalefact*(pvens-pvfcstmean))**2).sum(axis=0)/(nanals-1)
        if verbose: print n,fcstlen,np.sqrt(pverrsq.mean()),np.sqrt(pverrsqd.mean()),np.sqrt(pvspread.mean())
        pvspread_mean[nfcst] += pvspread/ncount
        pverrsq_mean[nfcst] += pverrsq/ncount
        pverrsqd_mean[nfcst] += pverrsqd/ncount

        if fcstlen in fcstlenspectra:
            pverrspec = scalefact*rfft2(pvfcstmean - pvtruth)
            psispec = modeld.invert(pverrspec)
            psispec = psispec/(modeld.N*np.sqrt(2.))
            kespec = (modeld.ksqlsq*(psispec*np.conjugate(psispec))).real
            kespec_errmean[nfcst] += kespec/ncount

            for nanal in range(nanals):
                pvsprdspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
                psispec = modeld.invert(pvsprdspec)
                psispec = psispec/(modeld.N*np.sqrt(2.))
                kespec = (modeld.ksqlsq*(psispec*np.conjugate(psispec))).real
                kespec_sprdmean[nfcst] += kespec/(nanals*ncount)

#print 'fcstlen = ',fcstlen, 'mean error =',np.sqrt(pverrsq_mean.mean()),np.sqrt(pverrsqd_mean.mean()),np.sqrt(pvspread_mean.mean())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for nfcst in range(fcsttimes):
    fcstlen = (nfcst+1)*fcstleninterval
    print fcstlen,np.sqrt(pverrsq_mean[nfcst].mean()),np.sqrt(pverrsqd_mean[nfcst].mean()),np.sqrt(pvspread_mean[nfcst].mean())
    if fcstlen in fcstlenspectra:
        #vmin = 0; vmax = 4
        #im = plt.imshow(np.sqrt(pvspread_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        #plt.title('mean spread')
        #plt.figure()
        #im = plt.imshow(np.sqrt(pverrsq_mean[1]),cmap=plt.cm.hot_r,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        #plt.title('mean error')

        k = np.abs((N*np.fft.fftfreq(N))[0:(N/2)+1])
        l = N*np.fft.fftfreq(N)
        k,l = np.meshgrid(k,l)
        ktot = np.sqrt(k**2+l**2)
        ktotmax = N/2+1
        kespec_err = np.zeros(ktotmax,np.float)
        kespec_sprd = np.zeros(ktotmax,np.float)
        for i in range(kespec_errmean[nfcst].shape[2]):
            for j in range(kespec_errmean[nfcst].shape[1]):
                totwavenum = ktot[j,i]
                if int(totwavenum) < ktotmax:
                    kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] +\
                    kespec_errmean[nfcst,:,j,i].mean(axis=0)
                    kespec_sprd[int(totwavenum)] = kespec_sprd[int(totwavenum)] +\
                    kespec_sprdmean[nfcst,:,j,i].mean(axis=0)
        plt.figure()
        wavenums = np.arange(ktotmax,dtype=np.float)
        wavenums[0] = 1.
        idealke = 2.*kespec_err[1]*wavenums**(-5./3,)
        plt.loglog(wavenums,kespec_err,color='k')
        plt.loglog(wavenums,kespec_sprd,color='b')
        #plt.loglog(wavenums,idealke,color='r')
        plt.title('error (black) and spread (blue) spectra for hr %s' %\
                int(3*fcstlen),fontsize=12)
        plt.savefig('%sherr_spectrum_hcorr%s.png' % (3*fcstlen,hcorr))
