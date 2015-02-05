from __future__ import print_function
from sqg import SQG
import numpy as np
from netCDF4 import Dataset
import sys, time
from enkf_utils import  cartdist,enkf_update,gaspcohn

# EnKF cycling for SQG turbulence model model with boundary temp obs,
# horizontal and vertical localization.  Adaptive relaxation to prior spread
# inflation.  Random or fixed observing network (obs on either boundary or
# both).

if len(sys.argv) == 1:
   msg="""
python sqg_enkf.py hcovlocal_scale vcovlocal_fact tau_covrelax diff_efold
   """
   raise SystemExit(msg)

# covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])
# factor to apply to horizontal location at boundary
# opposite to obs.
vcovlocal_fact = float(sys.argv[2])
# time scale for adaptive computation of relaxation to prior spread inflation parameter.
tau_covrelax = float(sys.argv[3])
# efolding time scale on smallest resolve wave for hyperdiffusion
diff_efold = float(sys.argv[4])

savedata = None # if not None, netcdf filename to save data.

profile = False # turn on profiling?

use_letkf = False # use serial EnSRF

# if nobs > 0, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
# if nobs < 0, fixed network of every Nth grid point used (N = -nobs)
nobs = 512  # number of obs to assimilate (randomly distributed)
#nobs = -1 # fixed network, every nobs grid points. nobs=-1 obs at all pts.

# if levob=0, sfc temp obs used.  if 1, lid temp obs used. If [0,1] obs at both
# boundaries.
levob = [0,1]; levob = list(levob); levob.sort()

direct_insertion = False # only relevant for nobs=-1, levob=[0,1]
if direct_insertion: print('# direct insertion!')

nanals = 40 # ensemble members

oberrstdev = 1.0 # ob error standard deviation in K
oberrextra = 0.0 # representativity error

nassim = 1200 # assimilation times to run

filename_climo = 'data/sqg_N64.nc' # file name for forecast model climo
filename_truth = 'data/sqg_N256_N64.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)
print('# nobs=%s levob=%s oberrextra=%s' % (nobs,levob,oberrextra))

# fix random seed for reproducibility.
np.random.seed(42)

# get model info
nc_climo = Dataset(filename_climo)
# parameter used to scale PV to temperature units.
scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
# initialize qg model instances for each ensemble member.
models = []
x = nc_climo.variables['x'][:]
y = nc_climo.variables['y'][:]
pv_climo = nc_climo.variables['pv']
indxran = np.random.choice(pv_climo.shape[0],size=nanals,replace=False)
x, y = np.meshgrid(x, y)
nx = len(x); ny = len(y)
pvens = np.empty((nanals,2,ny,nx),np.float32)
dt = nc_climo.dt
for nanal in range(nanals):
    pvens[nanal] = pv_climo[indxran[nanal]]
    models.append(\
    SQG(pvens[nanal],\
    nsq=nc_climo.nsq,f=nc_climo.f,dt=dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,symmetric=nc_climo.symmetric,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold))

print("# hcovlocal=%g vcovlocal=%s diff_efold=%s levob=%s tau_covrelax=%g nanals=%s" %\
        (hcovlocal_scale/1000.,vcovlocal_fact,diff_efold,levob,tau_covrelax,nanals))

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
if nobs < 0:
    nskip = -nobs
    if nx%nobs != 0:
        raise ValueError('nx must be divisible by nobs')
    nobs = (nx/nobs)**2
    fixed = True
else:
    fixed = False
oberrvar = oberrstdev**2*np.ones(nobs,np.float) + oberrextra**2
pvob = np.empty((len(levob),nobs),np.float)
covlocal = np.empty((ny,nx),np.float)
covlocal_tmp = np.empty((nobs,nx*ny),np.float)
xens = np.empty((nanals,2,nx*ny),np.float)
if not use_letkf:
    obcovlocal = np.empty((nobs,nobs),np.float)
else:
    obcovlocal = None
obtimes = nc_truth.variables['t'][:]
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[0]
    models[nanal].timesteps = assim_timesteps

# initialize relaxation to prior spread inflation factor.
if direct_insertion:
    tau_covrelax = -1.e-10
if tau_covrelax < 0.:
    # if negative, use fixed relaxation factor.
    covrelax = -tau_covrelax
covrelax_prev = 0.5

if savedata is not None:
   nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
   nc.r = models[0].r
   nc.f = models[0].f
   nc.U = models[0].U
   nc.L = models[0].L
   nc.H = models[0].H
   nc.nanals = nanals
   nc.hcovlocal_scale = hcovlocal_scale
   nc.vcovlocal_fact = vcovlocal_fact
   nc.tau_covrelax = tau_covrelax
   nc.oberrstdev = oberrstdev
   nc.levob = levob
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
   pv_b =\
   nc.createVariable('pv_b',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a =\
   nc.createVariable('pv_a',np.float32,('t','ens','z','y','x'),zlib=True)
   pv_a.units = 'K'
   pv_b.units = 'K'
   inf = nc.createVariable('inflation',np.float32,('t','z','y','x'),zlib=True)
   covr = nc.createVariable('covrelax',np.float32,('t',))
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

for ntime in range(nassim):

    # check model clock
    if models[0].t != obtimes[ntime]:
        raise ValueError('model/ob time mismatch %s vs %s' %\
        (models[0].t, obtimes[ntime]))

    t1 = time.time()
    if not fixed:
        p = np.ones((ny,nx),np.float)/(nx*ny)
        #psave = p.copy()
        #p[ny/4:3*ny/4,nx/4:3*nx/4] = 4.*psave[ny/4:3*ny/4,nx/4:3*nx/4]
        #p = p - p.sum()/(nx*ny) + 1./(nx*ny)
        indxob = np.random.choice(nx*ny,nobs,replace=False,p=p.ravel())
    else:
        mask = np.zeros((ny,nx),np.bool)
        nskip = int(nx/np.sqrt(nobs))
        mask[0:ny:nskip,0:nx:nskip] = True
        tmp = np.arange(0,nx*ny).reshape(ny,nx)
        indxob = tmp[mask.nonzero()].ravel()
    for k in range(len(levob)):
        # surface temp obs
        pvob[k] = scalefact*pv_truth[ntime,k,:,:].ravel()[indxob]
        pvob[k] += np.random.normal(scale=oberrstdev,size=nobs) # add ob errors
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]
    # plot ob network
    #import matplotlib.pyplot as plt
    #plt.contourf(x,y,pv_truth[0,1,...],15)
    #plt.scatter(xob,yob,color='k')
    #plt.axis('off')
    #plt.show()
    #raise SystemExit
    # compute covariance localization function for each ob
    if not fixed or ntime == 0:
        for nob in range(nobs):
            dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
            covlocal = gaspcohn(dist/hcovlocal_scale)
            covlocal_tmp[nob] = covlocal.ravel()
            dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
            if not use_letkf: obcovlocal[nob] = gaspcohn(dist/hcovlocal_scale)
            # plot covariance localization
            #import matplotlib.pyplot as plt
            #plt.contourf(x,y,covlocal,15)
            #plt.show()
            #raise SystemExit

    # first-guess spread (need later to compute inflation factor)
    fsprd = np.sqrt(((pvens - pvens.mean(axis=0))**2).sum(axis=0)/(nanals-1))

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,len(levob),nobs),np.float)
    for nanal in range(nanals):
        for k in range(len(levob)):
            hxens[nanal,k,...] = scalefact*pvens[nanal,k,...].ravel()[indxob] # surface pv obs
    hxensmean_b = hxens.mean(axis=0)
    obsprd = ((hxens-hxensmean_b)**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvensmean = pvens.mean(axis=0)
    pverr_b = (scalefact*(pvensmean-pv_truth[ntime]))**2
    pvsprd_b = ((scalefact*(pvensmean-pvens))**2).sum(axis=0)/(nanals-1)
    pv0err_b = np.sqrt(pverr_b[0].mean())
    pv1err_b = np.sqrt(pverr_b[1].mean())
    pv0sprd_b = np.sqrt(pvsprd_b[0].mean())
    pv1sprd_b = np.sqrt(pvsprd_b[1].mean())

    if savedata is not None:
        pv_b[ntime,:,:,:] = scalefact*pvens
        pv_obs[ntime] = pvob
        x_obs[ntime] = xob
        y_obs[ntime] = yob

    # EnKF update
    # create 1d state vector.
    for nanal in range(nanals):
        xens[nanal] = pvens[nanal].reshape((2,nx*ny))
    # update state vector.
    if direct_insertion and nobs == nx*ny and levob == [0,1]:
        for nanal in range(nanals):
            xens[nanal] =\
            pv_truth[ntime].reshape(2,nx*ny) + \
            np.random.normal(scale=oberrstdev,size=(2,nx*ny))/scalefact
        xens = xens - xens.mean(axis=0) + \
        pv_truth[ntime].reshape(2,nx*ny) + \
        np.random.normal(scale=oberrstdev,size=(2,nx*ny))/scalefact
    else:
        xens =\
        enkf_update(xens,hxens,pvob,oberrvar,covlocal_tmp,levob,vcovlocal_fact,obcovlocal=obcovlocal)
    # back to 3d state vector
    for nanal in range(nanals):
        pvens[nanal] = xens[nanal].reshape((2,ny,nx))
    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    # forward operator on posterior ensemble.
    for nanal in range(nanals):
        for k in range(len(levob)):
            hxens[nanal,k,...] = scalefact*pvens[nanal,k,...].ravel()[indxob] # surface pv obs
    # compute update to covariance relaxation parameter
    hxensmean_a = hxens.mean(axis=0)
    obsprd_a = (((hxens-hxensmean_a)**2).sum(axis=0)/(nanals-1)).mean()
    obinc_a = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_a)).mean()
    obinc_b = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_b)).mean()
    omaomb = ((pvob-hxensmean_a)*(pvob-hxensmean_b)).mean()
    # estimate inflation factor
    if obinc_a > 0.:
        # estimate inflation based on posterior spread
        estinf_a = np.sqrt(obinc_a/obsprd_a)
    else:
        estinf_a = 1.
    if obinc_b > 0.:
        # estimate inflation based on prior spread
        estinf_b = np.sqrt(obinc_b/obsprd_b)
    else:
        estinf_b = 1.
    estinf = 0.5*(estinf_a+estinf_b) # average the two estimates
    if estinf < 1.0: estinf=1.0 # inflation not deflation
    # convert global inflation parameter to relaxation to prior spread
    # parameter.
    if tau_covrelax >= 0.: # if < 0, used fixed relaxation factor.
        sprdfact = (np.sqrt(obsprd_b)-np.sqrt(obsprd_a))/np.sqrt(obsprd_a)
        covrelax = np.clip((estinf - 1.0)/sprdfact,0.0,None)
        # exponentially weighted moving average estimate.
        covrelax = covrelax_prev + (covrelax-covrelax_prev)/tau_covrelax
        covrelax_prev = covrelax

    # relaxation to prior spread inflation.
    pvensmean = pvens.mean(axis=0)
    pvprime = pvens-pvensmean
    asprd = np.sqrt((pvprime**2).sum(axis=0)/(nanals-1))
    inflation_factor = 1.+covrelax*(fsprd-asprd)/asprd
    pvprime = pvprime*inflation_factor
    pvens = pvprime + pvensmean

    # print out analysis error, spread and innov stats for background
    pverr_a = (scalefact*(pvensmean-pv_truth[ntime]))**2
    pvsprd_a = ((scalefact*(pvensmean-pvens))**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     obinc_b,obsprd_b,obinc_a,obsprd_a,omaomb,obbias_b,covrelax))
    #pv0err_a = np.sqrt(pverr_a[0].mean())
    #pv1err_a = np.sqrt(pverr_a[1].mean())
    #pv0sprd_a = np.sqrt(pvsprd_a[0].mean())
    #pv1sprd_a = np.sqrt(pvsprd_a[1].mean())
    #print("%s %g %g %g %g %g %g %g %g %g %g %g %g %g %g" %\
    #(ntime,pv0err_a,pv0sprd_a,pv1err_a,pv1sprd_a,\
    # pv0err_b,pv0sprd_b,pv1err_b,pv1sprd_b,\
    # obinc_b,obsprd_b,obinc_a,obsprd_a,obbias_b,covrelax))

    # save data.
    if savedata is not None:
        pv_a[ntime,:,:,:] = scalefact*pvens
        tvar[ntime] = obtimes[ntime]
        covr[ntime] = covrelax
        inf[ntime] = inflation_factor
        nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(pvens[nanal])
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
