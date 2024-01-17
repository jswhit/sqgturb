from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist, gaspcohn
from scipy.linalg import lapack, inv

# LETKF cycling for SQG turbulence model with vertical mean temp obs,
# Multi-scale horizontal (Z localization)  but no vertical localization.
# Relaxation to prior spread inflation.
# Random or fixed observing network.

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

read_restart = True
# if savedata not None, netcdf filename will be defined by env var 'exptname'
# if savedata = 'restart', only last time is saved (so expt can be restarted)
#savedata = True
#savedata = 'restart'
savedata = None
#nassim = 101
#nassim_spinup = 1
nassim = 500 # assimilation times to run
nassim_spinup = 100

nanals = 20 # ensemble members

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqg_N64_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqg_N64_6hrly.nc' # file name for nature run to draw obs
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
print("# hcovlocal=%s diff_efold=%s covinflate=%s nanals=%s" %\
     (repr(hcovlocal_scales_km),diff_efold,covinflate,nanals))
print('# band_cutoffs=%s' % repr(band_cutoffs))

# if nobs > 0, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
# if nobs < 0, fixed network of every Nth grid point used (N = -nobs)
nobs = nx*ny//2 # number of obs to assimilate (randomly distributed)
#nobs = -1 # fixed network, every -nobs grid points. nobs=-1 obs at all pts.

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
if nobs < 0:
    nskip = -nobs
    if (nx*ny)%nobs != 0:
        raise ValueError('nx*ny must be divisible by nobs')
    nobs = (nx*ny)//nskip**2
    print('# fixed network nobs = %s' % nobs)
    fixed = True
else:
    fixed = False
    print('# random network nobs = %s' % nobs)
if nobs == nx*ny//2: fixed=True # used fixed network for obs every other grid point
oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)
xens = np.empty((nanals,2,nx*ny),np.float32)

# model-space localization matrix (only needed for Z localization)
n = 0
covlocal_modelspace = np.empty((nlscales,nx*ny,nx*ny),np.float32)
x1 = x.reshape(nx*ny); y1 = y.reshape(nx*ny)
mincovlocal = np.finfo(np.float32).eps
for nscale in range(nlscales):
   for n in range(nx*ny):
       dist = cartdist(x1[n],y1[n],x1,y1,nc_climo.L,nc_climo.L)
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
print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,meanpverr_b,meanpvsprd_b,obfits_b,osprd_b+R,obbias_b,inflation,tr(P^a)/tr(P^b)')

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

# forward operator.
# hxens is ensemble in observation space.
def hofx(x,indxob,model):
    # given temp at boundaries, return vertical mean
    # temp at locations specified by indxob.
    nanals = x.shape[0]
    scalefact = model.f*model.theta0/model.g
    if indxob.dtype == np.bool_:
        nobs = indxob.sum()
    else:
        nobs = len(indxob)
    if x.ndim < 4:
        nxny = x.shape[2]
        ny = int(np.sqrt(nxny)); nx = ny # grid is square
        pvens = x.reshape(nanals,2,ny,nx)
    else:
        ny = x.shape[2]; nx = x.shape[3]
        nxny = ny*nx
        pvens = x
    hxens = np.empty((nanals,nobs),np.float32)
    xens = np.zeros((nanals,nxny),np.float32)
    for nanal in range(nanals):
        pvspec = rfft2(pvens[nanal])
        xtmp = irfft2(model.meantemp(pvspec=pvspec))
        xtmp = xtmp.reshape(nxny)
        hxens[nanal,:] = scalefact*xtmp[indxob] # mean temp obs
        xens[nanal] = xtmp
    return xens.squeeze(), hxens.squeeze() # remove singleton dims

# initialize kinetic energy error/spread spectra
kespec_errmean = None; kespec_sprdmean = None

ncount = 0
nanalske = min(10,nanals) # ensemble members used for kespec spread
normfact = np.array(np.sqrt(nlscales*nanals-1),dtype=np.float32)

# loop over assimilation times.
for ntime in range(nassim):

    # check model clock
    if models[0].t != obtimes[ntime+ntstart]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime+ntstart]))

    t1 = time.time()
    if not fixed:
        # randomly choose points from model grid
        if nobs == nx*ny:
            indxob = np.arange(nx*ny)
        else:
            indxob = np.sort(rsobs.choice(nx*ny,nobs,replace=False))
    else:
        mask = np.zeros((ny,nx),np.bool_)
        # if every other grid point observed, shift every other time step
        # so every grid point is observed in 2 cycles.
        if nobs == nx*ny//2:
            if ntime%2:
                mask[0:ny,1:nx:2] = True
            else:
                mask[0:ny,0:nx:2] = True
        else:
            mask[0:ny:nskip,0:nx:nskip] = True
        indxob = np.flatnonzero(mask)

    # mean temp obs
    meanpv_truth,pvob = hofx(pv_truth[ntime+ntstart].reshape((1,2,ny,nx)),indxob,models[0])
    pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x1[indxob]; yob = y1[indxob]

    # first-guess spread (need later to compute inflation factor)
    fsprd = ((pvens - pvens.mean(axis=0))**2).sum(axis=0)/(nanals-1)

    meanpvens, hxens = hofx(pvens,indxob,models[0])
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
    meanpvensmean_b = meanpvens.mean(axis=0)
    pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
    meanpverr_b = (scalefact*(meanpvensmean_b-meanpv_truth))**2
    pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)
    meanpvsprd_b = ((scalefact*(meanpvensmean_b-meanpvens))**2).sum(axis=0)/(nanals-1)

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
    xmean_b = xmean.copy()
    xprime_b = xprime.copy()

    # update state vector.
    # hxens,pvob are in PV units, xens is not

    # loop over model grid points, perform update in each local region.
    xprime_squeeze=np.empty((nanals*nlscales,2,nx*ny),xprime.dtype)
    xprime_squeeze2=np.empty((nanals*nlscales,2,nx*ny),xprime.dtype)
    for n in range(nx*ny):
        distob = cartdist(x1[n],y1[n],xob,yob,nc_climo.L,nc_climo.L)
        obindx = distob < np.abs(hcovlocal_scales[0])
        ominusf = (pvob - hxensmean_b)[obindx]
        # Z localization
        # perform observation operator on 'squeezed' state vector
        # (same as R localization for identity H)
        for nlscale in range(nlscales):
            squeezefact = covlocal_modelspace[nlscale,:,n]
            nanal1=nlscale*nanals; nanal2=(nlscale+1)*nanals
            xprime_squeeze[nanal1:nanal2] = np.sqrt(squeezefact[np.newaxis,np.newaxis,:])*xprime[nanal1:nanal2]
            xprime_squeeze2[nanal1:nanal2] = squeezefact[np.newaxis,np.newaxis,:]*xprime[nanal1:nanal2]
        xtmp,hxprime_local = hofx(xprime_squeeze, indxob[obindx], models[0])
        xtmp,hxprime_local2 = hofx(xprime_squeeze2, indxob[obindx], models[0])
        Rinv = 1./oberrvar[obindx]
        YbRinv = hxprime_local2*Rinv/normfact
        YbsqrtRinv = hxprime_local*np.sqrt(Rinv)/normfact

        # LETKF update
        pa = np.eye(nlscales*nanals) + np.dot(YbsqrtRinv, YbsqrtRinv.T)

        # Using eigenanalysis
        evals, eigs, info = lapack.dsyevd(pa)
        evals = evals.clip(min=1.+np.finfo(evals.dtype).eps)
        pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)

        # Using cholesky decomp
        #pasqrt, info = lapack.dpotrf(pa,overwrite_a=0)
        #pasqrtinv = inv(np.triu(pasqrt))

        tmp = np.dot(np.dot(np.dot(pasqrtinv, pasqrtinv.T), YbRinv), ominusf)/normfact
        wts = pasqrtinv + tmp[:, np.newaxis]
        for k in range(2):
            xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime_b[:, k, n])

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
    print("%s %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     np.sqrt(meanpverr_b.mean()),np.sqrt(meanpvsprd_b.mean()),
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
