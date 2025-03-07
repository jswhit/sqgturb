from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, rfft2, irfft2, cartdist, gaspcohn
from scipy.linalg import eigh

# EnKF cycling for SQG turbulence model with boundary temp obs,
# serial solvers (local volume and global) with assimilation thresholding
# Relaxation to prior spread inflation.
# Random or fixed observing network.

if len(sys.argv) == 1:
   msg="""
python sqg_enkf.py hcovlocal_scale covinflate corr_thresh corr_power vlocal>
   hcovlocal_scale = horizontal localization scale in km
   covinflate: RTPS covinflate inflation parameter
   corr_thresh: corr threshold for serial assimilation (set to zero for use all obs, or neg to turn off sorting by corr)
   corr_power: power to raise correlation in ob error inflation (set to zero for no extra factor in oberr inflation)
   vlocal: vertical localization (1, i.e. no vertical localization,  by default)
   """
   raise SystemExit(msg)

# horizontal covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])
covinflate = float(sys.argv[2])
corr_thresh = float(sys.argv[3])
corr_power = float(sys.argv[4])
if len(sys.argv) > 5:
    vcovlocal_fact = float(sys.argv[5])
else:
    vcovlocal_fact = 1.
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
nassim = 300 # assimilation times to run
nassim_spinup = 100

nanals = 20 # ensemble members

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqgu20_N64_6hrly.nc' # file name for forecast model climo
# perfect model
#filename_truth = 'sqgu20_N64_6hrly.nc' # file name for nature run to draw obs
filename_truth = 'sqgu20_N128N64_6hrly.nc' # file name for nature run to draw obs

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
x1 = x.ravel(); y1 = y.ravel()
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

print("# hcovlocal=%g vcovlocal=%g diff_efold=%s covinfate=%s nanals=%s" %\
     (hcovlocal_scale/1000.,vcovlocal_fact,diff_efold,covinflate,nanals))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
nobs = 2*nx*ny//4 # number of obs to assimilate (randomly distributed)

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
print('# random network nobs = %s' % nobs)

oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)
covlocal = np.empty((ny,nx),np.float32)
covlocal_tmp = np.empty((nobs,nx*ny),np.float32)

xens = np.empty((nanals,2,nx*ny),np.float32)

# square-root of vertical localization
if vcovlocal_fact > 0.99: # no vertical localization
    vcovlocal_sqrt = np.ones((2,1),np.float32)
else:
    vloc = np.array([(1,vcovlocal_fact),(vcovlocal_fact,1)],np.float32)
    evals, evecs = eigh(vloc)
    vcovlocal_sqrt = np.dot(evecs, np.diag(np.sqrt(evals)))

obtimes = nc_truth.variables['t'][:]
if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
print('# assim interval = %s secs (%s time steps) corr_thresh = %s corr_power=%s' % (assim_interval,assim_timesteps,corr_thresh,corr_power))
print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obfits_b,osprd_b+R,obbias_b,inflation,tr(P^a)/tr(P^b),nobs_assim')

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
kespec_errmean = None; kespec_sprdmean = None

ncount = 0
nanalske = min(10,nanals) # ensemble members used for kespec spread

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

    # first-guess spread (need later to compute inflation factor)
    pvensmean = pvens.mean(axis=0)
    pvprime = pvens - pvensmean

    # modulate ensemble
    neig = vcovlocal_sqrt.shape[1]; nanals2 = neig*nanals
    pvprime2 = np.empty((nanals2,2,ny,nx),pvprime.dtype)
    nanal2 = 0
    for j in range(neig):
        for nanal in range(nanals):
            for k in range(2):
                pvprime2[nanal2,k,...] =\
                pvprime[nanal,k,...]*vcovlocal_sqrt[k,neig-j-1]
            nanal2 += 1

    # check modulation works
    #crosscov1 = (pvprime[:,0,...]*pvprime[:,1,...]).sum(axis=0)/(nanals-1)
    #crosscov2 = (pvprime2[:,0,...]*pvprime2[:,1,...]).sum(axis=0)/(nanals-1)
    #print(vcovlocal_fact,(crosscov2/crosscov1).mean()) # should be the same
    #raise SystemExit

    fsprd = (pvprime**2).sum(axis=0)/(nanals-1)
    pvens2 = pvprime2 + pvensmean # modulated ensemble (size nanals2=nanals*neig)

    # compute forward operator on modulated ensemble.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float32)
    hxens2 = np.empty((nanals2,nobs),np.float32)

    for nanal in range(nanals):
        hxens[nanal] = scalefact*pvens[nanal,...].reshape(2*nx*ny)[indxob] # surface pv obs
    for nanal in range(nanals2):
        hxens2[nanal] = scalefact*pvens2[nanal,...].reshape(2*nx*ny)[indxob] # surface pv obs
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
    xens2 = pvens2.reshape(nanals2,2,nx*ny)

    # update state vector.

    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime2 = xens2 - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    hxprime2 = hxens2 - hxmean

    nobscount = 0

    distob = np.empty((nx*ny,nobs), np.float32)
    for n in range(nx*ny):
        distob[n,:] = cartdist(xx[n],yy[n],xob,yob,nc_climo.L,nc_climo.L)

    for k in range(2):
        for n in range(nx*ny):
            mask = distob[n] < np.abs(hcovlocal_scale)

            # select obs in local region
            nobs_local = mask.sum()
            oberr_local = oberrvar[mask]
            obs_local = pvob[mask]
            hxmean_local = hxmean[mask]
            hxprime_local = hxprime[:,mask]
            hxprime2_local = hxprime2[:,mask]
            distob_local = distob[n,mask]

            # assimilate obs in local region in order of decreasing correlation
            # with state variable being updated.
            iassim = np.zeros(nobs_local, np.int32)
            corrmax = 1.e30; ncountassim = 0
            while corrmax > corr_thresh and ncountassim < nobs_local:

                # compute correlation between ob and state
                varobs = (hxprime2_local**2).sum(axis=0)/(nanals-1)
                varstate =  (xprime2[:, k, n]**2).sum(axis=0)/(nanals-1)
                pbht = (xprime2[:,k,n,np.newaxis]*hxprime2_local).sum(axis=0) / (nanals-1)
                corr = np.abs(pbht/np.sqrt(varobs[np.newaxis,:]*varstate)).squeeze()
                corr[iassim==1]=0 # set corr to zero for already assimilated obs
                if corr_thresh < 0: 
                    nobx = ncountassim # don't sort by correlation if corr_thresh<0
                else:
                    nobx = np.argmax(corr)
                corrmax = corr[nobx]
                
                #if n==0: print(k,ncountassim, nobs_local, nobx,corrmax)
                iassim[nobx]=1
                # step 1: update observed variable for ob being assimilated
                #varob = (hxprime2_local[:,nobx]**2).sum(axis=0)/(nanals-1)
                varob = varobs[nobx]
                # ob error inflation based on gaspari-cohn taper multiplied by correlation raised to a power.
                covlocal_ob = (corrmax**corr_power)*gaspcohn(distob_local[nobx]/hcovlocal_scale)
                gainob = varob/(varob+(oberr_local[nobx]/covlocal_ob)) # only place where localization enters in
                hxmean_a = (1.-gainob)*hxmean_local[nobx] + gainob*obs_local[nobx] # linear interp
                hxprime_a = np.sqrt(1.-gainob)*hxprime_local[:,nobx] # rescaling
                hxprime2_a = np.sqrt(1.-gainob)*hxprime2_local[:,nobx] 
                # step 2: update model priors in state and ob space 
                # (linear regression of model priors on observation priors)
                obinc_mean = hxmean_a - hxmean_local[nobx]
                obinc_prime = hxprime_a - hxprime_local[:,nobx]
                obinc_prime2 = hxprime2_a - hxprime2_local[:,nobx]
                # state space
                #pbht = (xprime2[:, k, n] * hxprime2_local[:,nobx]).sum(axis=0) / (nanals-1)
                pbht = pbht[nobx] 
                xmean[k, n] +=  (pbht/varob)*obinc_mean
                xprime[:, k, n] += (pbht/varob)*obinc_prime
                xprime2[:, k, n] += (pbht/varob)*obinc_prime2
                # ob space (only need to update obs not yet assimilated)
                mask_assim = iassim == 0
                pbht = (hxprime2_local[:,mask_assim] * hxprime2_local[:,nobx][:,np.newaxis]).sum(axis=0) / (nanals-1)
                hxmean_local[mask_assim] += (pbht/varob)*obinc_mean
                hxprime_local[:,mask_assim] += (pbht[np.newaxis,:]/varob)*obinc_prime[:,np.newaxis]
                hxprime2_local[:,mask_assim] += (pbht[np.newaxis,:]/varob)*obinc_prime2[:,np.newaxis]
                ncountassim += 1

            xens[:,:,n] = xmean[:,n]+xprime[:,:,n]
            nobscount += ncountassim
            #print(k,nobscount,ncountassim)
    nobscount = nobscount//(2*nx*ny)

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
    print("%s %g %g %g %g %g %g %g %g %g %g" %\
    (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     np.sqrt(obfits_b),np.sqrt(obsprd_b+oberrstdev**2),obbias_b,
     inflation_factor.mean(),asprd_over_fsprd,nobscount))

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
