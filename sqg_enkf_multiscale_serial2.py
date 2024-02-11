from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist, gaspcohn
from scipy.linalg import lapack, inv

# local volume serial filter cycling for SQG turbulence model with boundary temp obs.
# Multi-scale horizontal (Z localization)  but no vertical localization.
# Relaxation to prior spread inflation.
# Random observing network.

if len(sys.argv) == 1:
   msg="""
python sqg_enkf.py hcovlocal_scales band_cutoffs covinflate 
   hcovlocal_scales: horizontal localization scales in km (specified as list, in double quotes, e.g. "[2000.e6,1000.e6]"
   band_cutoffs:  wavelength cutoffs for filter bands (dimension one less than hcovlocal_scales)
   covinflate: RTPS covinflate inflation parameter
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scales = eval(sys.argv[1])
nlscales = len(hcovlocal_scales)
band_cutoffs = eval(sys.argv[2])
nband_cutoffs = len(band_cutoffs)
covinflate = float(sys.argv[3])
if nband_cutoffs != nlscales-1:
    raise SystemExit('band_cutoffs should be one less than hcovlocal_scales')

exptname = os.getenv('exptname','sqg_enkf')
threads = int(os.getenv('OMP_NUM_THREADS','1'))

diff_efold = None # use diffusion from climo file

profile = False # turn on profiling?

read_restart = False
# if savedata not None, netcdf filename will be defined by env var 'exptname'
# if savedata = 'restart', only last time is saved (so expt can be restarted)
#savedata = True
#savedata = 'restart'
savedata = None
#nassim = 101
#nassim_spinup = 1
nassim = 200 # assimilation times to run
nassim_spinup = 100

nanals = 20 # ensemble members

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqgu20_N64_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqgu20_N64_6hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqg_N256_N96_12hrly.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState() # varying seed for initial conditions

# get model info
nc_climo = Dataset(filename_climo)
# parameter used to scale PV to temperature units.
scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
# initialize qg model instances for each ensemble member.
x = nc_climo.variables['x'][:]
y = nc_climo.variables['y'][:]
x, y = np.meshgrid(x, y)
nx = len(x); ny = len(y)
dt = nc_climo.dt
if diff_efold == None: diff_efold=nc_climo.diff_efold
pvens = np.empty((nanals,2,ny,nx),np.float32)
if not read_restart:
    pv_climo = nc_climo.variables['pv']
    indxran = rsics.choice(pv_climo.shape[0],size=nanals,replace=False)
else:
    ncinit = Dataset('%s_restart.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
    ncinit.set_auto_mask(False)
    pvens[:] = ncinit.variables['pv_b'][-1,0:nanals,...]/scalefact
    tstart = ncinit.variables['t'][-1]
    #for nanal in range(nanals):
    #    print(nanal, pvens[nanal].min(), pvens[nanal].max())
# get OMP_NUM_THREADS (threads to use) from environment.
models = []
for nanal in range(nanals):
    if not read_restart:
        pvens[nanal] = pv_climo[indxran[nanal]]
        #print(nanal, pvens[nanal].min(), pvens[nanal].max())
    models.append(\
    SQG(pvens[nanal],
    nsq=nc_climo.nsq,f=nc_climo.f,dt=dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,symmetric=nc_climo.symmetric,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold,threads=threads))
if read_restart: ncinit.close()

hcovlocal_scales_km = [lscale/1000. for lscale in hcovlocal_scales]
print("# hcovlocal=%s diff_efold=%s covinflate=%s nanals=%s" %\
     (repr(hcovlocal_scales_km),diff_efold,covinflate,nanals))
print('# band_cutoffs=%s' % repr(band_cutoffs))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
nobs = 2*nx*ny//8 # number of obs to assimilate (randomly distributed)

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)
xens = np.empty((nanals,2,nx*ny),np.float32)

# model-space localization matrix (only needed for Z localization)
n = 0
covlocal_modelspace = np.empty((nlscales,2*nx*ny,nx*ny),np.float32)
x1 = x.reshape(nx*ny); y1 = y.reshape(nx*ny)
x2 = np.concatenate((x1,x1)); y2 = np.concatenate((y1,y1))
mincovlocal = np.finfo(np.float32).eps
for nscale in range(nlscales):
   for n in range(2*nx*ny):
       dist = cartdist(x2[n],y2[n],x1,y1,nc_climo.L,nc_climo.L)
       covlocal_modelspace[nscale,n,:] = \
       np.clip(gaspcohn(dist/hcovlocal_scales[nscale]),a_min=mincovlocal,a_max=None)

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s_restart.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# assim interval = %s secs (%s time steps)' % (assim_interval,assim_timesteps))
print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obfits_b,osprd_b+R,obbias_b,inflation,tr(P^a)/tr(P^b)')

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[ntstart]
    models[nanal].timesteps = assim_timesteps

# initialize output file.
if savedata is not None:
   nc = Dataset('%s.nc' % exptname, mode='w', format='NETCDF4_CLASSIC')
   nc.r = models[0].r
   nc.f = models[0].f
   nc.U = models[0].U
   nc.L = models[0].L
   nc.H = models[0].H
   nc.nanals = nanals
   nc.hcovlocal_scales = hcovlocal_scales
   nc.oberrstdev = oberrstdev
   nc.g = nc_climo.g; nc.theta0 = nc_climo.theta0
   nc.nsq = models[0].nsq
   nc.tdiab = models[0].tdiab
   nc.dt = models[0].dt
   nc.diff_efold = models[0].diff_efold
   nc.diff_order = models[0].diff_order
   nc.filename_climo = filename_climo
   nc.filename_truth = filename_truth
   nc.symmetric = models[0].symmetric
   xdim = nc.createDimension('x',models[0].N)
   ydim = nc.createDimension('y',models[0].N)
   z = nc.createDimension('z',2)
   t = nc.createDimension('t',None)
   obs = nc.createDimension('obs',nobs)
   ens = nc.createDimension('ens',nanals)
   pv_t =\
   nc.createVariable('pv_t',np.float32,('t','z','y','x'),zlib=True)
   pv_b =\
   nc.createVariable('pv_b',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a =\
   nc.createVariable('pv_a',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a.units = 'K'
   pv_b.units = 'K'
   inf = nc.createVariable('inflation',np.float32,('t','z','y','x'),zlib=True)
   pv_obs = nc.createVariable('obs',np.float32,('t','obs'))
   x_obs = nc.createVariable('x_obs',np.float32,('t','obs'))
   y_obs = nc.createVariable('y_obs',np.float32,('t','obs'))
   # eady pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
   xvar = nc.createVariable('x',np.float32,('x',))
   xvar.units = 'meters'
   yvar = nc.createVariable('y',np.float32,('y',))
   yvar.units = 'meters'
   zvar = nc.createVariable('z',np.float32,('z',))
   zvar.units = 'meters'
   tvar = nc.createVariable('t',np.float32,('t',))
   tvar.units = 'seconds'
   ensvar = nc.createVariable('ens',np.int32,('ens',))
   ensvar.units = 'dimensionless'
   xvar[:] = np.arange(0,models[0].L,models[0].L/models[0].N)
   yvar[:] = np.arange(0,models[0].L,models[0].L/models[0].N)
   zvar[0] = 0; zvar[1] = models[0].H
   ensvar[:] = np.arange(1,nanals+1)

# initialize kinetic energy error/spread spectra
kespec_errmean = None; kespec_sprdmean = None

ncount = 0
nanalske = min(10,nanals) # ensemble members used for kespec spread
normfact = np.array(np.sqrt(nlscales*nanals-1),dtype=np.float32)
squeezewts = np.empty(nlscales,np.float32)

# loop over assimilation times.
for ntime in range(nassim):

    # check model clock
    if models[0].t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()

    # indxob defines observing network
    indxob = np.sort(rsobs.choice(2*nx*ny,nobs,replace=False))
    # pvob is sample nature run plus noise
    pvob = scalefact*pv_truth[ntime+ntstart,...].reshape(2*nx*ny)[indxob]
    pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x2[indxob]; yob = y2[indxob]
    # forward operatore to get ensemble in ob space
    hxens = np.empty((nanals,nobs),np.float32)
    for nanal in range(nanals):
        hxens[nanal] = scalefact*pvens[nanal,...].reshape(2*nx*ny)[indxob] # surface pv obs

    # first-guess spread (need later to compute inflation factor)
    fsprd = ((pvens - pvens.mean(axis=0))**2).sum(axis=0)/(nanals-1)

    hxensmean_b = hxens.mean(axis=0)
    hxprime_b = hxens-hxensmean_b
    obsprd = (hxprime_b**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvensmean_b = pvens.mean(axis=0).copy()
    pvpert = pvens-pvensmean_b
    pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
    pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)

    # filter backgrounds into different scale bands
    if nlscales == 1:
        pvens_filtered_lst=[pvpert]
    else:
        pvens_filtered_lst=[]
        pvfilt_save = np.zeros_like(pvpert)
        pvspec = rfft2(pvpert)
        for n,cutoff in enumerate(band_cutoffs):
            pvfiltspec = np.where(models[0].wavenums[np.newaxis,np.newaxis,...] < cutoff, pvspec, 0.+0.j)
            pvfilt = irfft2(pvfiltspec)
            pvens_filtered_lst.append(pvfilt-pvfilt_save)
            pvfilt_save=pvfilt
        pvsum = np.zeros_like(pvens)
        for n in range(nband_cutoffs):
            pvsum += pvens_filtered_lst[n]
        pvens_filtered_lst.append(pvpert-pvsum)
    # concatenate along ensemble dimension (nanals*nlscales)
    pvens_filtered = np.vstack(pvens_filtered_lst)
    pvens = pvensmean_b + pvens_filtered

    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_t[ntime] = pv_truth[ntime+ntstart]
            pv_b[ntime,:,:,:] = scalefact*pvens
            pv_obs[ntime] = pvob
            x_obs[ntime] = xob
            y_obs[ntime] = yob

    # EnKF update
    # create 1d state vector.
    xens = pvens.reshape(nlscales*nanals,2,nx*ny)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean

    # update state vector.
    # hxens,pvob are in PV units, xens is not

    # loop over model grid points, perform update in each local region.
    xprime_squeeze=np.empty((nanals*nlscales,2,nx*ny),xprime.dtype)
    hxprime_localsqueeze = np.empty(nanals*nlscales, xprime.dtype)
    for n in range(nx*ny):
        distob = cartdist(x1[n],y1[n],xob,yob,nc_climo.L,nc_climo.L)
        obindx = distob < np.abs(hcovlocal_scales[0])
        nobs_local = obindx.sum()
        squeezefact=[]
        for nlscale in range(nlscales):
            squeezefact.append( np.sqrt(covlocal_modelspace[nlscale,:,n]) )
        squeezefact = np.asarray(squeezefact)
        # squeeze' in ob space
        hxprime_local = hxprime_b[:,obindx]
        oberr_local = oberrvar[obindx]
        obs_local = pvob[obindx]
        hxmean_local = hxensmean_b[obindx]
        # loop over obs in local region
        for nob, ob, oberr in zip(np.arange(nobs_local), obs_local, oberr_local):
            # squeeze ob space
            squeezewts_norm = 0.
            for nlscale in range(nlscales):
                nanal1=nlscale*nanals; nanal2=(nlscale+1)*nanals
                obsqueezefact = squeezefact[nlscale,indxob[obindx]][nob]
                squeezewts[nlscale] = obsqueezefact
                squeezewts_norm += squeezewts[nlscale]**2
                hxprime_localsqueeze[nanal1:nanal2] = obsqueezefact*hxprime_local[nanal1:nanal2,nob]
            squeezewts = squeezewts/np.sqrt(squeezewts_norm)
            # step 1: update observed variable for ob being assimilated
            #varob = (hxprime_localsqueeze[:,nob]**2).sum(axis=0)/(nanals-1)
            varob = (hxprime_localsqueeze[:]**2).sum(axis=0)/(nanals-1)
            gainob = varob/(varob+oberr)
            hxmean_a = (1.-gainob)*hxmean_local[nob] + gainob*ob # linear interp
            hxprime_a = np.sqrt(1.-gainob)*hxprime_local[:,nob] # rescaling
            # step 2: update model priors in state and ob space 
            # (linear regression of model priors on observation priors)
            obinc_mean = hxmean_a - hxmean_local[nob]
            obinc_prime = hxprime_a - hxprime_local[:,nob]
            # state space
            xprime_weighted = (squeezewts[:,np.newaxis,np.newaxis]*(xprime[:,:,n].reshape(nlscales,nanals,2))).sum(axis=0)
            hxprime_weighted = (squeezewts[:,np.newaxis,np.newaxis]*(hxprime_local.reshape(nlscales,nanals,nobs_local))).sum(axis=0) 
            hpbht = (hxprime_weighted[:,nob]**2).sum(axis=0) / (nanals-1)
            for k in range(2):
                pbht = (xprime_weighted[:, k].T * hxprime_weighted[:,nob]).sum(axis=0) / (nanals-1)
                xmean[k, n] += (pbht/hpbht)*obinc_mean
                xprime[:, k, n] += (pbht/hpbht)*obinc_prime
            # ob space (only really need to update obs not yet assimilated)
            pbht = (hxprime_weighted[:,nob:].T * hxprime_weighted[:,nob]).sum(axis=1) / (nanals-1)
            hxmean_local[nob:] += (pbht/hpbht)*obinc_mean
            hxprime_local[:,nob:] += (pbht[np.newaxis,:]/hpbht)*obinc_prime[:,np.newaxis]
        xens[:,:,n] = xmean[:,n] + xprime[:,:,n]


    # back to 3d state vector
    pvens = xens.reshape((nlscales*nanals,2,ny,nx))
    pvensmean_a = pvens.mean(axis=0)
    pvens_filtered = pvens - pvensmean_a
    pvens_filtered = pvens_filtered.reshape(nlscales,nanals,2,ny,nx)
    pvens = pvens_filtered.sum(axis=0) + pvensmean_a
    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    # posterior multiplicative inflation.
    pvprime = pvens-pvensmean_a
    asprd = (pvprime**2).sum(axis=0)/(nanals-1)
    asprd_over_fsprd = asprd.mean()/fsprd.mean()
    # relaxation to prior stdev (Whitaker & Hamill 2012)
    asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
    inflation_factor = 1.+covinflate*(fsprd-asprd)/asprd
    pvprime = pvprime*inflation_factor
    pvens = pvprime + pvensmean_a

    # print out analysis error, spread and innov stats for background
    pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime+ntstart]))**2
    pvsprd_a = ((scalefact*(pvensmean_a-pvens))**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     np.sqrt(obfits_b),np.sqrt(obsprd_b+oberrstdev**2),obbias_b,
     inflation_factor.mean(),asprd_over_fsprd))

    # save data.
    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_a[ntime,:,:,:] = scalefact*pvens
            tvar[ntime] = obtimes[ntime+ntstart]
            inf[ntime] = inflation_factor
            nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(pvens[nanal])
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

    # compute spectra of error and spread
    if ntime >= nassim_spinup:
        pvfcstmean = pvens.mean(axis=0)
        pverrspec = scalefact*rfft2(pvfcstmean - pv_truth[ntime+ntstart+1])
        psispec = models[0].invert(pverrspec)
        psispec = psispec/(models[0].N*np.sqrt(2.))
        kespec = (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
        if kespec_errmean is None:
            kespec_errmean =\
            (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
        else:
            kespec_errmean = kespec_errmean + kespec
        for nanal in range(nanalske):
            pvsprdspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
            psispec = models[0].invert(pvsprdspec)
            psispec = psispec/(models[0].N*np.sqrt(2.))
            kespec = (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real
            if kespec_sprdmean is None:
                kespec_sprdmean =\
                (models[0].ksqlsq*(psispec*np.conjugate(psispec))).real/nanalske
            else:
                kespec_sprdmean = kespec_sprdmean+kespec/nanalske
        ncount += 1

if savedata: nc.close()

if ncount:
    kespec_sprdmean = kespec_sprdmean/ncount
    kespec_errmean = kespec_errmean/ncount
    N = models[0].N
    k = np.abs((N*np.fft.fftfreq(N))[0:(N//2)+1])
    l = N*np.fft.fftfreq(N)
    k,l = np.meshgrid(k,l)
    ktot = np.sqrt(k**2+l**2)
    ktotmax = (N//2)+1
    kespec_err = np.zeros(ktotmax,np.float32)
    kespec_sprd = np.zeros(ktotmax,np.float32)
    for i in range(kespec_errmean.shape[2]):
        for j in range(kespec_errmean.shape[1]):
            totwavenum = ktot[j,i]
            if int(totwavenum) < ktotmax:
                kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] +\
                kespec_errmean[:,j,i].mean(axis=0)
                kespec_sprd[int(totwavenum)] = kespec_sprd[int(totwavenum)] +\
                kespec_sprdmean[:,j,i].mean(axis=0)

    print('# mean error/spread',kespec_errmean.sum(), kespec_sprdmean.sum())
    plt.figure()
    wavenums = np.arange(ktotmax,dtype=np.float32)
    for n in range(1,ktotmax):
        print('# ',wavenums[n],kespec_err[n],kespec_sprd[n])
    plt.loglog(wavenums[1:-1],kespec_err[1:-1],color='r')
    plt.loglog(wavenums[1:-1],kespec_sprd[1:-1],color='b')
    plt.title('error (red) and spread (blue) spectra')
    plt.savefig('errorspread_spectra_%s.png' % exptname)
