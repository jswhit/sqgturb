import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist, lgetkf_ms_vloc, gaspcohn
from scipy.linalg import eigh

# LGETKF cycling for SQG turbulence model with boundary temp obs,
# multi-scale ob space horizontal localization, no vertical localization.
# cross-validation update (no inflation).
# Random observing network.

if len(sys.argv) == 1:
   msg="""
python sqg_lgetkf_cv.py hcovlocal_scale covinflate>
   hcovlocal_scales = horizontal localization scale(s) in km
   vcovlocal_facts = vertical localization factors
   band_cutoffs = filter waveband cutoffs 
   crossbandcov_facts = cross-band covariance factors
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scales = eval(sys.argv[1])
nlscales = len(hcovlocal_scales)
vcovlocal_facts = eval(sys.argv[2])
if len(hcovlocal_scales) != len(vcovlocal_facts):
    raise SystemExit('number of horiz and vertical localization factors must be the same')
band_cutoffs = eval(sys.argv[3])
nband_cutoffs = len(band_cutoffs)
if nband_cutoffs != nlscales-1:
    raise SystemExit('band_cutoffs should be one less than hcovlocal_scales')
crossbandcov_facts = eval(sys.argv[4])
if len(crossbandcov_facts) != nband_cutoffs:
    raise SystemExit('band_cutoffs and crossbandcov_facts should be same length')
crossband_covmat = np.ones((nlscales,nlscales),np.float32)
crossband_covmatr = np.ones((nlscales,nlscales),np.float32)
for i in range(nlscales):
    for j in range(nlscales):
        if j != i:
            crossband_covmat[j,i] = crossbandcov_facts[np.abs(i-j)-1] 
            crossband_covmatr[j,i] = -crossbandcov_facts[np.abs(i-j)-1] 

exptname = os.getenv('exptname','test')
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
nassim = 600 # assimilation times to run
nassim_spinup = 100

nanals = 16 # ensemble members
ngroups = 8  # number of groups for cross-validation (ngroups=nanals is "leave one out")

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqgu20_N96_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqgu20_N96_6hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqg_N256_N96_12hrly.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
#rsics = np.random.RandomState() # varying seed for initial conditions
rsics = np.random.RandomState(24) # fixed seed for initial conditions

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
    pvens[:] = ncinit.variables['pv_b'][-1,...]/scalefact
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
print("# hcovlocal=%s vcovlocal=%s diff_efold=%s nanals=%s ngroups=%s" %\
     (repr(hcovlocal_scales_km),repr(vcovlocal_facts),diff_efold,nanals,ngroups))
print('# band_cutoffs=%s crossbandcov_facts=%s' % (repr(band_cutoffs),repr(crossbandcov_facts)))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
#nobs = nx*ny//6 # number of obs to assimilate (randomly distributed)
nobs = 1024

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
print('# random network nobs = %s' % nobs)

oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)
covlocal = np.empty((ny,nx),np.float32)
covlocal_tmp = np.empty((nlscales,nobs,nx*ny),np.float32)

# square-root of vertical localization
vcovlocal_facts = np.asarray(vcovlocal_facts)
if np.all(vcovlocal_facts > 0.99): # no vertical localization
    vcovlocal_sqrt = np.ones((nlscales,2,1),np.float32)
else:
    # number of eigenvectores for each scale must be the same
    vcovlocal_sqrt = np.empty((nlscales,2,2),np.float32)
    for nl in range(nlscales):
        vloc = np.array([(1,vcovlocal_facts[nl]),(vcovlocal_facts[nl],1)],np.float32)
        evals, evecs = eigh(vloc)
        vcovlocal_sqrt[nl] = np.dot(evecs, np.diag(np.sqrt(evals)))

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# assim interval = %s secs (%s time steps)' % (assim_interval,assim_timesteps))
print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obfits_b,osprd_b+R,obbias_b,tr(P^a)/tr(P^b)')

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
   nc.band_cutoffs = band_cutoffs
   nc.crossbandcov_facts = crossband_covfacts
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
   pv_obs = nc.createVariable('obs',np.float32,('t','z','obs'))
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
pvspec_errmean = None; pvspec_sprdmean = None

ncount = 0
normfact = np.array(np.sqrt(nlscales*nanals-1),dtype=np.float32)

N = models[0].N
k = np.abs((N*np.fft.fftfreq(N))[0:(N//2)+1])
l = N*np.fft.fftfreq(N)
imax = len(k); jmax = len(l)
k,l = np.meshgrid(k,l)
ktotsq = (k**2+l**2).astype(np.int32)
jmax,imax = ktotsq.shape
ktot = np.sqrt(ktotsq)
ktotmax = (N//2)+1

for ntime in range(nassim):

    # check model clock
    if models[0].t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()
    indxob = np.sort(rsobs.choice(2*nx*ny,nobs,replace=False))
    pvob = scalefact*pv_truth[ntime+ntstart,...].reshape(2*nx*ny)[indxob]
    pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = np.concatenate((x.ravel(),x.ravel()))[indxob]
    yob = np.concatenate((y.ravel(),y.ravel()))[indxob]
    # compute covariance localization function for each ob
    for nl in range(nlscales):
        for nob in range(nobs):
            dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
            covlocal = gaspcohn(dist/hcovlocal_scales[nl])
            covlocal_tmp[nl,nob,...] = covlocal.ravel()

    # first-guess spread
    pvensmean_b = pvens.mean(axis=0)
    pvprime = pvens - pvensmean_b

    fsprd = (pvprime**2).sum(axis=0)/(nanals-1)

    # compute forward operator on modulated ensemble.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float32)

    for nanal in range(nanals):
        hxens[nanal] = scalefact*pvens[nanal,...].reshape(2*nx*ny)[indxob] # surface pv obs
    hxensmean_b = hxens.mean(axis=0)
    obsprd = ((hxens-hxensmean_b)**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvpert = pvens-pvensmean_b
    pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
    pvsprd_b = ((scalefact*pvpert)**2).sum(axis=0)/(nanals-1)

    # filter background perturbations into different scale bands
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
        pvsum = np.zeros_like(pvpert)
        for n in range(nband_cutoffs):
            pvsum += pvens_filtered_lst[n]
        pvens_filtered_lst.append(pvpert-pvsum)
    pvprime = np.asarray(pvens_filtered_lst)
    pvprime = np.dot(pvprime.T,crossband_covmat).T
    pvens = pvprime + pvensmean_b  # mean added back to all scales.

    # modulate ensemble
    def modens(pvprime, vcovlocal_sqrt):
        nlscales = vcovlocal_sqrt.shape[0]
        neig = vcovlocal_sqrt.shape[2]
        pvprime2 = np.empty((nlscales,neig*nanals,2,ny,nx),pvprime.dtype)
        nanal2 = 0
        nanal_index = np.empty((nlscales*nanals*neig),np.int32)
        for nl in range(nlscales):
            nanal1 = 0
            for j in range(neig):
                for nanal in range(nanals):
                    for k in range(2):
                        pvprime2[nl,nanal1,k,...] =\
                        pvprime[nl,nanal,k,...]*vcovlocal_sqrt[nl,k,neig-j-1]
                    nanal_index[nanal2]=nanal
                    nanal1 += 1
                    nanal2 += 1
        return nanal_index,pvprime2
    nanal_index,pvprime2 = modens(pvprime, vcovlocal_sqrt)
    nanals2 = pvprime2.shape[1]
    ## check modulation works
    ##crosscov1 = (pvprime[:,0,...]*pvprime[:,1,...]).sum(axis=0)/(nanals-1)
    ##crosscov2 = (pvprime2[:,0,...]*pvprime2[:,1,...]).sum(axis=0)/(nanals-1)
    ##print(nanals2,vcovlocal_fact,(crosscov2/crosscov1).mean()) # should be the same
    pvens2 = pvprime2 + pvensmean_b 

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
    # concatenate along ensemble dimension (nanals*nlscales)
    xens = pvens.reshape(nlscales*nanals,2,nx*ny)
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxprime = np.empty((nanals*nlscales,nobs),np.float32)
    for nanal in range(nanals*nlscales):
        hxprime[nanal] = (scalefact*xprime[nanal].reshape(2*nx*ny))[indxob] # surface pv obs
    xens2 = pvens2.reshape(nlscales*nanals2,2,nx*ny)
    xprime2 = xens2 - xmean
    hxprime2 = np.empty((nanals2*nlscales,nobs),np.float32)
    for nanal in range(nanals2*nlscales):
        hxprime2[nanal] = (scalefact*xprime2[nanal].reshape(2*nx*ny))[indxob] # surface pv obs

    # update state vector.

    # hxens,pvob are in PV units, xens is not
    xens = lgetkf_ms_vloc(nlscales,xens,xens2,hxprime,hxprime2,pvob-hxensmean_b,oberrvar,covlocal_tmp,nanal_index,ngroups=ngroups)

    # back to 3d state vector
    pvens = xens.reshape((nlscales*nanals,2,ny,nx))
    pvensmean_a = pvens.mean(axis=0) 
    pvens_filtered = pvens - pvensmean_a
    pvens_filtered = pvens_filtered.reshape(nlscales,nanals,2,ny,nx)
    pvprime = np.dot(pvens_filtered.T,crossband_covmatr).T
    pvens = pvprime.sum(axis=0) + pvensmean_a

    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    pvensmean_a = pvens.mean(axis=0)
    pvprime = pvens - pvensmean_a
    asprd = (pvprime**2).sum(axis=0)/(nanals-1)
    asprd_over_fsprd = asprd.mean()/fsprd.mean()

    # print out analysis error, spread and innov stats for background
    pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime+ntstart]))**2
    pvsprd_a = ((scalefact*(pvensmean_a-pvens))**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     np.sqrt(obfits_b),np.sqrt(obsprd_b+oberrstdev**2),obbias_b,
     asprd_over_fsprd))

    # save data.
    if savedata is not None:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_a[ntime,:,:,:] = scalefact*pvens
            tvar[ntime] = obtimes[ntime+ntstart]
            nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(pvens[nanal])
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)
    if not np.all(np.isfinite(pvens)):
        raise SystemExit('non-finite values detected after forecast, stopping...')

    # compute spectra of error and spread
    if ntime >= nassim_spinup:
        pvfcstmean = pvens.mean(axis=0)
        pverrspec = scalefact*rfft2(pvfcstmean - pv_truth[ntime+ntstart+1])
        pverrspec_mag = (pverrspec*np.conjugate(pverrspec)).real
        if pvspec_errmean is None:
            pvspec_errmean = pverrspec_mag
        else:
            pvspec_errmean = pvspec_errmean + pverrspec_mag
        for nanal in range(nanals):
            pvpertspec = scalefact*rfft2(pvens[nanal] - pvfcstmean)
            pvpertspec_mag = (pvpertspec*np.conjugate(pvpertspec)).real/(nanals-1)
            if pvspec_sprdmean is None:
                pvspec_sprdmean = pvpertspec_mag
            else:
                pvspec_sprdmean = pvspec_sprdmean+pvpertspec_mag
        ncount += 1

if savedata: nc.close()

if ncount:
    pvspec_sprdmean = pvspec_sprdmean/ncount
    pvspec_errmean = pvspec_errmean/ncount
    pvspec_err = np.zeros(ktotmax,np.float32)
    pvspec_sprd = np.zeros(ktotmax,np.float32)
    for i in range(pvspec_errmean.shape[2]):
        for j in range(pvspec_errmean.shape[1]):
            totwavenum = int(np.round(ktot[j,i]))
            if totwavenum < ktotmax:
                pvspec_err[totwavenum] = pvspec_err[totwavenum] +\
                pvspec_errmean[:,j,i].mean(axis=0) # average of upper/lower boundary
                pvspec_sprd[totwavenum] = pvspec_sprd[totwavenum] +\
                pvspec_sprdmean[:,j,i].mean(axis=0)

    print('# mean error/spread',pvspec_errmean.sum(), pvspec_sprdmean.sum())
    plt.figure()
    wavenums = np.arange(ktotmax,dtype=np.float32)
    for n in range(1,ktotmax):
        print('# ',wavenums[n],pvspec_err[n],pvspec_sprd[n])
    plt.loglog(wavenums[1:-1],pvspec_err[1:-1],color='r')
    plt.loglog(wavenums[1:-1],pvspec_sprd[1:-1],color='b')
    plt.title('error (red) and spread (blue)')
    plt.savefig('errorspread_spectra_cv_%s.png' % exptname)
