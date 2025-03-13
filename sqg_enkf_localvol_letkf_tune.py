import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist, gaspcohn
from scipy.linalg import lapack

# EnKF cycling for SQG turbulence model with boundary temp obs,
# ob space horizontal and model space vertical localization.
# Relaxation to prior spread inflation.
# Random or fixed observing network.
# Options for LETKF or serial EnSRF.

if len(sys.argv) == 1:
   msg="""
python sqg_enkf.py hcovlocal_scale covinflate corr_power corr_thresh vcovlocal_fact> 
   hcovlocal_scale = horizontal localization scale in km
   covinflate: RTPS covinflate inflation parameter
   corr_power: power to raise correlation in ob error inflation (set to zero for no extra factor in oberr inflation)
   corr_thresh: corr threshold for serial assimilation (set to zero for use all obs, or neg to turn off sorting by corr)
   vcovlocal_fact: optional - vertical localizatino factor (default 1)
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])
covinflate = float(sys.argv[2])
corr_power = float(sys.argv[3])
corr_thresh = float(sys.argv[4])
if len(sys.argv) > 5:
    vcovlocal_fact = float(sys.argv[5])
else:
    vcovlocal_fact = 1.
exptname = os.getenv('exptname','test')
threads = int(os.getenv('OMP_NUM_THREADS','1'))
verbose = bool(int(os.getenv('VERBOSE','0')))

diff_efold = None # use diffusion from climo file

read_restart = False
nassim = 300 # assimilation times to run
nassim_spinup = 100

nanals = 20 # ensemble members

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqgu20_N96_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqgu20_N96_6hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqgu20_N128N96_6hrly.nc' # file name for nature run to draw obs

if verbose:
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
xx = x.ravel(); yy = y.ravel()
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

if verbose:
     print('# corr_power=%s corr_thresh=%s vcovlocal_fact=%s' % (corr_power,corr_thresh,vcovlocal_fact))
     print("# hcovlocal=%g diff_efold=%s covinfate=%s nanals=%s" %\
          (hcovlocal_scale/1000.,diff_efold,covinflate,nanals))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
nobs = 2*nx*ny//6 # number of obs to assimilate (randomly distributed)

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
if verbose:
    print('# random network nobs = %s' % nobs)

oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)

xens = np.empty((nanals,2,nx*ny),np.float32)

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    if verbose:
        print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
if verbose:
    print('# assim interval = %s secs (%s time steps) corr_thresh = %s corr_power=%s' % (assim_interval,assim_timesteps,corr_thresh,corr_power))
    print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obfits_b,osprd_b+R,obbias_b,inflation,tr(P^a)/tr(P^b),nobs_local')

def run_exp(models, pv_truth, pvens, hcovlocal_scale, vcovlocal_fact, corr_power, covinflate):
    # initialize model clock
    for nanal in range(nanals):
        models[nanal].t = obtimes[ntstart]
        models[nanal].timesteps = assim_timesteps
    ncount = 0
    pverra_mean = 0.
    pvsprda_mean = 0.
    pverrb_mean = 0.
    pvsprdb_mean = 0.
    
    for ntime in range(nassim):
    
        # check model clock
        if models[0].t != obtimes[ntime+ntstart]:
            raise ValueError('model/ob time mismatch %s vs %s' %\
            (models[0].t, obtimes[ntime+ntstart]))
    
        indxob = np.sort(rsobs.choice(2*nx*ny,nobs,replace=False))
        kob = (indxob >= nx*ny).astype(np.int32) # is ob at top or bottom?
        pvob = scalefact*pv_truth[ntime+ntstart,...].reshape(2*nx*ny)[indxob]
        pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
        xob = np.concatenate((x.ravel(),x.ravel()))[indxob]
        yob = np.concatenate((y.ravel(),y.ravel()))[indxob]
    
        # first-guess spread (need later to compute inflation factor)
        pvensmean = pvens.mean(axis=0)
        pvprime = pvens - pvensmean
    
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
        pvensmean_b = pvens.mean(axis=0).copy()
        pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime+ntstart]))**2
        pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)
    
        # EnKF update
        # create 1d state vector.
        xens = pvens.reshape(nanals,2,nx*ny)
    
        # update state vector.
    
        xmean = xens.mean(axis=0)
        xprime = xens - xmean
    
        normfact = np.array(np.sqrt(nanals-1),dtype=np.float32)
        nobs_local = 0
        for n in range(nx*ny):
            distob = cartdist(xx[n],yy[n],xob,yob,nc_climo.L,nc_climo.L)
            mask = distob < np.abs(hcovlocal_scale)
    
            # select obs in local region
            oberr_local = oberrvar[mask]
            obs_local = pvob[mask]
            hxmean_local = hxensmean_b[mask]
            hxprime_local = hxens[:,mask] - hxmean_local
            distob_local = distob[mask]
            vcovl_local = kob[mask]
            ominusf = obs_local - hxmean_local
    
            for k in range(2):
    
                # compute correlation between ob and state (precomputed)
                if corr_power > 0:
                    varobs = (hxprime_local**2).sum(axis=0)/normfact
                    varstate =  (xprime[:, k, n]**2).sum(axis=0)/normfact
                    pbht = (xprime[:,k,n,np.newaxis]*hxprime_local).sum(axis=0) / normfact
                    corr = np.abs(pbht/np.sqrt(varobs[np.newaxis,:]*varstate)).squeeze()
    
                    if corr_thresh > 0:
                        mask2 = corr > corr_thresh
                        oberr_local = oberr_local[mask2]
                        distob_local = distob_local[mask2]
                        hxprime_local = hxprime_local[:,mask2]
                        vcovl_local = vcovl_local[mask2]
                        corr = corr[mask2]
                        ominusf = ominusf[mask2]
    
                    covlocal_ob = (corr**corr_power)*gaspcohn(distob_local/hcovlocal_scale)
                else:
                    covlocal_ob = gaspcohn(distob_local/hcovlocal_scale)
    
                covlocal_ob = covlocal_ob.clip(min=1.e-7)
                nobs_local =+ nobs_local + len(oberr_local)
    
                vcovlocal = np.where(vcovl_local == k, 1, vcovlocal_fact)
                Rinvsqrt = np.sqrt(vcovlocal*covlocal_ob/oberr_local)
                YbRinv = hxprime_local*Rinvsqrt**2/normfact
                YbsqrtRinv = hxprime_local*Rinvsqrt/normfact
    
                # LETKF update
                pa = np.eye(nanals) + np.dot(YbsqrtRinv, YbsqrtRinv.T)
    
                # Using eigenanalysis
                evals, eigs, info = lapack.dsyevd(pa)
                evals = evals.clip(min=1.+np.finfo(evals.dtype).eps)
                pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
    
                # Using cholesky decomp
                #pasqrt, info = lapack.dpotrf(pa,overwrite_a=0)
                #pasqrtinv = inv(np.triu(pasqrt))
    
                tmp = np.dot(np.dot(np.dot(pasqrtinv, pasqrtinv.T), YbRinv), ominusf)/normfact
                wts = pasqrtinv + tmp[:, np.newaxis]
                xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime[:, k, n])
    
    
        nobs_local = nobs_local // (2*nx*ny)
        # back to 3d state vector
        pvens = xens.reshape((nanals,2,ny,nx))
    
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
        if ntime+ntstart >= nassim_spinup:
            pverra_mean += np.sqrt(pverr_a.mean())
            pvsprda_mean += np.sqrt(pvsprd_a.mean())
            pverrb_mean += np.sqrt(pverr_b.mean())
            pvsprdb_mean += np.sqrt(pvsprd_b.mean())
            ncount += 1
        if verbose:
            print("%s %g %g %g %g %g %g %g %g %g %g" %\
            (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
             np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
             np.sqrt(obfits_b),np.sqrt(obsprd_b+oberrstdev**2),obbias_b,
             inflation_factor.mean(),asprd_over_fsprd,nobs_local))
    
        # run forecast ensemble to next analysis time
        for nanal in range(nanals):
            pvens[nanal] = models[nanal].advance(pvens[nanal])

    return ncount, pverra_mean/ncount, pvsprda_mean/ncount, pverrb_mean/ncount, pvsprdb_mean/ncount

#for hcovl_scale in np.arange(2000,3201,100):
#   hcovlocal_scale = hcovl_scale*1000.
if verbose:
    ncount,pverra,pvsprda,pverrb,pvsprdb = run_exp(models, pv_truth, pvens, hcovlocal_scale, vcovlocal_fact, corr_power, covinflate)
    print('%g %g %g %g %s %s %g %g %g %g' % (hcovlocal_scale/1000,vcovlocal_fact,covinflate,corr_power,nassim_spinup,ncount,pverra,pvsprda,pverrb,pvsprdb))
else:
    for corr_power in np.arange(0.5,1.51,0.1):
        ncount,pverra,pvsprda,pverrb,pvsprdb = run_exp(models, pv_truth, pvens, hcovlocal_scale, vcovlocal_fact, corr_power, covinflate)
        print('%g %g %g %g %s %s %g %g %g %g' % (hcovlocal_scale/1000,vcovlocal_fact,covinflate,corr_power,nassim_spinup,ncount,pverra,pvsprda,pverrb,pvsprdb))
        
        #from tinydb import TinyDB
        #db = TinyDB('perfect96_results/db.json')
        #db.insert({'hcovlocal_scale': int(hcovlocal_scale/1000.), 'vcovlocal_fact': int(vcovlocal_fact*100), 'covinflate': int(covinflate*100), 
        #           'corr_power': int(corr_power*100),
        #           'PVError_anal': pverra, 'PVError_bkg': pverrb, 'PVSpread_anal': pvsprda, 'PVSpread_bkg': pvsprdb})
        #db.close()
