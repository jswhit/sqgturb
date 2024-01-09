from __future__ import print_function
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist,enkf_update,gaspcohn
from scipy.linalg import eigh, lapack

# EnKF cycling for SQG turbulence model with vertical mean temp obs,
# ob space horizontal and model space vertical localization.
# Relaxation to prior spread inflation.
# Random or fixed observing network.
# Options for LETKF or serial EnSRF.

if len(sys.argv) == 1:
   msg="""
python sqg_enkf.py hcovlocal_scale covinflate>
   hcovlocal_scale = horizontal localization scale in km
   covinflate: RTPS covinflate inflation parameter
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])
covinflate = float(sys.argv[2])
exptname = os.getenv('exptname','test')
threads = int(os.getenv('OMP_NUM_THREADS','1'))

diff_efold = None # use diffusion from climo file

profile = False # turn on profiling?

use_letkf = True  # use LGETKF, otherwise use serial EnSRF
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
filename_climo = 'sqg_N64_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqg_N64_6hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqg_N256_N96_12hrly.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState(24) # varying seed for initial conditions

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

print('# use_letkf=%s' % (use_letkf))
print("# hcovlocal=%g diff_efold=%s covinfate=%s nanals=%s" %\
     (hcovlocal_scale/1000.,diff_efold,covinflate,nanals))

# if nobs > 0, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
# if nobs < 0, fixed network of every Nth grid point used (N = -nobs)
nobs = nx*ny//4 # number of obs to assimilate (randomly distributed)
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
covlocal = np.empty((ny,nx),np.float32)
covlocal_tmp = np.empty((nobs,nx*ny),np.float32)
xens = np.empty((nanals,2,nx*ny),np.float32)
if not use_letkf:
    obcovlocal = np.empty((nobs,nobs),np.float32)
else:
    obcovlocal = None

# model-space horizontal localization matrix
n = 0
covlocal_modelspace = np.empty((nx*ny,nx*ny),np.float32)
x1 = x.reshape(nx*ny); y1 = y.reshape(nx*ny)
for n in range(nx*ny):
    dist = cartdist(x1[n],y1[n],x1,y1,nc_climo.L,nc_climo.L)
    covlocal_modelspace[n,:] = gaspcohn(dist/hcovlocal_scale)
percentvar_cutoff = 0.99

# square root of truncated localization matrix
evals, evecs = eigh(covlocal_modelspace,driver='evd')
neig = 1
for nn in range(1,nx*ny):
    percentvar = evals[-nn:].sum()/evals.sum()
    if percentvar > 0.95: # perc variance cutoff truncation
        neig = nn
        break
print('#neig = ',neig)
evecs_norm = (evecs*np.sqrt(evals/percentvar)).T
#sqrtcovlocal = evecs_norm[-neig:,:]
sqrtcovlocal = evecs_norm[-neig:,:].reshape((neig,ny,nx))

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
   nc.hcovlocal_scale = hcovlocal_scale
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
indx_ens=np.ones(nanals,np.bool_); indx_lev=np.ones(2,np.bool_)
# modulate ensemble
def modens(enspert,sqrtcovlocal):
    nanals = enspert.shape[0]
    neig = sqrtcovlocal.shape[0]
    return np.multiply(np.repeat(sqrtcovlocal[:,np.newaxis,:],nanals,axis=0),np.tile(enspert,(neig,1,1)))
def modens1(enspert,sqrtcovlocal):
    nanals = enspert.shape[0]
    neig = sqrtcovlocal.shape[0]
    return np.multiply(np.repeat(sqrtcovlocal,nanals,axis=0),np.tile(enspert,(neig,1)))

normfact = np.array(nanals-1,dtype=np.float32)

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
    mask = np.zeros(nx*ny,dtype=np.bool_); mask[indxob]=True
    # mean temp obs
    pvspec_truth = rfft2(pv_truth[ntime+ntstart])
    meanpv_truth = irfft2(models[0].meantemp(pvspec=pvspec_truth))
    pvob = scalefact*meanpv_truth.ravel()[indxob]
    pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x.ravel()[indxob]; yob = y.ravel()[indxob]

    # first-guess spread (need later to compute inflation factor)
    pvensmean = pvens.mean(axis=0)
    pvprime = pvens - pvensmean
    fsprd = (pvprime**2).sum(axis=0)/(nanals-1)

    # compute forward operator
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float32)
    meanpvens = np.zeros((nanals,ny,nx),np.float32)
    for nanal in range(nanals):
        pvspec = rfft2(pvens[nanal])
        meanpv = irfft2(models[0].meantemp(pvspec=pvspec))
        meanpvens[nanal] = meanpv
        hxens[nanal,...] = scalefact*meanpv.ravel()[indxob] # mean temp obs

    #for nanal in range(nanals2):
    #    pvspec = rfft2(pvens2[nanal])
    #    meanpv = irfft2(models[0].meantemp(pvspec=pvspec))
    #    hxens2[nanal,...] = scalefact*meanpv.ravel()[indxob] # mean temp obs

    hxensmean_b = hxens.mean(axis=0)
    hxprime = hxens-hxensmean_b
    obsprd = ((hxprime)**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvensmean_b = pvens.mean(axis=0).copy()
    meanpvensmean_b = meanpvens.mean(axis=0)
    meanpvprime = (meanpvens-meanpvensmean_b).reshape((nanals,nx*ny))
    pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
    meanpverr_b = (scalefact*(meanpvensmean_b-meanpv_truth))**2
    pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)
    meanpvsprd_b = ((scalefact*(meanpvensmean_b-meanpvens))**2).sum(axis=0)/(nanals-1)

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
    xens = pvens.reshape(nanals,2,nx*ny)
    xmean = pvensmean_b.reshape(2,nx*ny)
    xmean_b = xmean.copy()
    xprime = xens - xmean
    xprime_b = xprime.copy()

    # update state vector.
    # hxens,pvob are in PV units, xens is not

    for n in range(nx*ny):
        dist = cartdist(x1[n],y1[n],x1,y1,nc_climo.L,nc_climo.L)
        indx = dist < np.abs(hcovlocal_scale)
        nmindist = np.argmin(dist[indx])
        covlocal_local = covlocal_modelspace[np.ix_(indx,indx)]
        #evals, evecs = eigh(covlocal_local,driver='evd')
        evals, evecs, info = lapack.dsyevd(covlocal_local)
        neig = 1
        for nn in range(1,nx*ny):
            percentvar = evals[-nn:].sum()/evals.sum()
            if percentvar > percentvar_cutoff: # perc variance cutoff truncation
                neig = nn
                break
        evecs_norm = (evecs*np.sqrt(evals/percentvar)).T
        sqrtcovlocal_local = evecs_norm[-neig:,:]
        xprime_local = xprime_b[np.ix_(indx_ens,indx_lev,indx)]
        xprime2_local = modens(xprime_local,sqrtcovlocal_local)
        #print(neig, xprime_local.shape,xprime2_local.shape)
        distob = cartdist(x1[n],y1[n],xob,yob,nc_climo.L,nc_climo.L)
        obindx = distob < np.abs(hcovlocal_scale)
        #ndim_local = np.sum(indx); nobs_local = np.sum(obindx)
        #print(ndim_local,nobs_local)
        #print(np.nonzero(indx))
        #print(indxob[obindx])
        #print(np.nonzero(np.in1d(np.nonzero(indx)[0],indxob[obindx])))
        indxob_local = np.in1d(np.nonzero(indx)[0],indxob[obindx])
        hxensmean_local = hxensmean_b[obindx]
        obs_local = pvob[obindx]
        hxprime_local = hxprime[np.ix_(indx_ens,obindx)]
        #print(meanpvprime.shape)
        meanpvprime_local = meanpvprime[np.ix_(indx_ens,indx)]
        meanpvprime2_local = modens1(meanpvprime_local,sqrtcovlocal_local)
        #print(meanpvprime_local.shape, meanpvprime2_local.shape)
        indx_ens2=np.ones(nanals*neig,np.bool_)
        hxprime2_local = scalefact*meanpvprime2_local[np.ix_(indx_ens2,indxob_local)]
        #meanpvensmean_b = meanpvensmean_b.reshape(nx*ny)
        #meanpvensmean_b_local1 = meanpvensmean_b[indxob[obindx]]
        #print(meanpvensmean_b_local1.shape)
        #print(meanpvensmean_b_local1)
        #meanpvensmean_local = meanpvensmean_b[indx]
        #meanpvensmean_b_local2 = meanpvensmean_local[indxob_local]
        #print(meanpvensmean_b_local2.shape)
        #print(meanpvensmean_b_local2)
        #raise SystemExit

        ominusf = obs_local - hxensmean_local
        Rinv = 1./oberrvar[obindx]
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv = hxprime2_local*np.sqrt(Rinv)/normfact  
        YbRinv = hxprime2_local*Rinv/normfact               
        pa = np.dot(YbsqrtRinv,YbsqrtRinv.T)
        evals, evecs, info = lapack.dsyevd(pa)
        gamma_inv = np.zeros_like(evals)
        for n in range(evals.shape[0]):
            if evals[n] > np.finfo(evals.dtype).eps:
                gamma_inv[n] = 1./evals[n]
            else:
                evals[n] = 0.
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # create HZ^T R**-1/2
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        wts_ensmean = np.dot(pa, np.dot(YbRinv,ominusf))/normfact
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # For DEnKF factor is -0.5*C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 HXprime
        # = -0.5 Pa (HZ)^ T R**-1/2 HXprime (Pa already computed)
        # pa = C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T
        # gammapI = sqrt(1.0/gammapI)
        # ( pa=0.5*pa for denkf)
        pa=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # if denkf, wts_ensperts = -0.5 C (Gamma + I)**-1 C^T (HZ)^T R**-1/2 HXprime
        wts_ensperts = -np.dot(pa, np.dot(YbRinv,hxprime_local.T)).T/normfact # use orig ens here

        for k in range(2):
            xmean[k,n] += np.dot(wts_ensmean,xprime2_local[:,k,nmindist]) 
            # use orig ens on lhs, mod ens on rhs
            xprime[:,k,n] += np.dot(wts_ensperts,xprime2_local[:,k,nmindist]) 

    xens = xmean + xprime
    #xens =\
    #enkf_update(xens,xens2,hxens,hxens2,pvob,oberrvar,covlocal_tmp,obcovlocal=obcovlocal)

    # back to 3d state vector
    pvens = xens.reshape((nanals,2,ny,nx))
    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    # posterior multiplicative inflation.
    pvensmean_a = pvens.mean(axis=0)
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
