#from __future__ import print_function
from sqg import SQG, rfft2, irfft2
import numpy as np
from scipy import linalg
from netCDF4 import Dataset
import sys, time
from enkf_meantemp_utils import  cartdist,enkf_update,enkf_update_modens,gaspcohn
from scipy.ndimage.filters import uniform_filter, gaussian_filter

# EnKF cycling for SQG turbulence model model with vertically
# integrated temp obs (and horizontal filter in forward operator).
# Random or fixed observing network.

np.seterr(all='raise') # raise error when overflow occurs

if len(sys.argv) == 1:
   msg="""
python sqg_enkf_meantemp_ml.py hcovlocal_scale covinflate1 covinflate2
   """
   raise SystemExit(msg)

# covariance localization length scale in meters.
hcovlocal_scale = float(sys.argv[1])
modelspace_local = bool(int(sys.argv[2])) # model or ob space localization
use_letkf = bool(int(sys.argv[3])) # (local) ETKF or serial assimilation
covinflate1=1.; covinflate2=1.
if len(sys.argv) > 4:
    # inflation parameters for Hodyss and Campbell inflation
    covinflate1 = float(sys.argv[4])
    covinflate2 = float(sys.argv[5])

# representativity error
oberrextra = 0.0

diff_efold = None # use diffusion from climo file

savedata = None # if not None, netcdf filename to save data.
#savedata = 'sqg_enkf.nc'

profile = False # turn on profiling?

# if nobs > 0, each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
# if nobs < 0, fixed network of every Nth grid point used (N = -nobs)
nobs = 128 # number of obs to assimilate (randomly distributed)
#nobs = -4 # fixed network, every -nobs grid points. nobs=-1 obs at all pts.

nanals = 20 # ensemble members

oberrstdev = 0.1 # ob error standard deviation in K

nassim = 2200 # assimilation times to run

# smoothing parameters for forward operator.
use_gaussian_filter=True
if use_gaussian_filter:
    filter_width = 3
else:
    filter_width = 10

filename_climo = 'data/sqg_N64.nc' # file name for forecast model climo
# perfect model
filename_truth = 'data/sqg_N64.nc' # file name for nature run to draw obs
# model error
#filename_truth = 'data/sqg_N128_N64.nc' # file name for nature run to draw obs

print('# filename_modelclimo=%s' % filename_climo)
print('# filename_truth=%s' % filename_truth)
print('# oberr=%s oberrextra=%s' % (oberrstdev,oberrextra))

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState() # varying seed for initial conditions

# get model info
nc_climo = Dataset(filename_climo)
# parameter used to scale PV to temperature units.
scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
# initialize qg model instances for each ensemble member.
models = []
x1 = nc_climo.variables['x'][:]
y1 = nc_climo.variables['y'][:]
nx = len(x1); ny = len(y1)
pv_climo = nc_climo.variables['pv']
indxran = rsics.choice(pv_climo.shape[0],size=nanals,replace=False)
x, y = np.meshgrid(x1, y1)
pvens = np.empty((nanals,2,ny,nx),np.float32)
dt = nc_climo.dt
if diff_efold == None: diff_efold=nc_climo.diff_efold
for nanal in range(nanals):
    pvens[nanal] = pv_climo[indxran[nanal]]
    models.append(\
    SQG(pvens[nanal],\
    nsq=nc_climo.nsq,f=nc_climo.f,dt=dt,U=nc_climo.U,H=nc_climo.H,\
    r=nc_climo.r,tdiab=nc_climo.tdiab,symmetric=nc_climo.symmetric,\
    diff_order=nc_climo.diff_order,diff_efold=diff_efold))
    #import matplotlib.pyplot as plt
    #pvspec = rfft2(pvens[nanal])
    #psispec = models[nanal].invert(pvspec=pvspec)
    #pvavspec = (psispec[1]-psispec[0])/models[nanal].H
    #pvav = scalefact*irfft2(pvavspec)
    #filter_widthg = 3
    #pvav2 = gaussian_filter(pvav, filter_widthg, mode='wrap')
    #filter_widthu = 10
    #pvav3 = uniform_filter(pvav, size=filter_widthu, mode='wrap')
    #print pvav.min(), pvav.max()
    #vmin = -30; vmax = 30
    #im = plt.imshow(pvav,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    #plt.title('unfiltered')
    #plt.figure()
    #im = plt.imshow(pvav2,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    #plt.title('gaussian filtered sigma=%s' % (filter_widthg))
    #plt.figure()
    #im = plt.imshow(pvav3,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
    #plt.title('uniform filtered width=%s' % (filter_widthu))
    #plt.show()
    #raise SystemExit

print("# hcovlocal=%g diff_efold=%s covinf1=%s covinf2=%s nanals=%s use_letkf=%s" %\
     (hcovlocal_scale/1000.,diff_efold,covinflate1,covinflate2,nanals,use_letkf))

# nature run
nc_truth = Dataset(filename_truth)
pv_truth = nc_truth.variables['pv']
# set up arrays for obs and localization function
if nobs < 0:
    nskip = -nobs
    if nx%nobs != 0:
        raise ValueError('nx must be divisible by nobs')
    nobs = (nx/nobs)**2
    print('# nobs=%s (fixed ob network), modelspace_local=%s' % (nobs,modelspace_local))
    fixed = True
else:
    print('# nobs=%s (random obs network), modelspace_local=%s' % (nobs,modelspace_local))
    fixed = False
if use_gaussian_filter:
    print('# forward operator gaussian filter with stdev %s' % filter_width)
else:
    print('# forward operator %s x %s block mean' %(filter_width,filter_width))
pvob = np.empty(nobs,np.float)
covlocal = np.empty((nx*ny,nx*ny),np.float)
xens = np.empty((nanals,2*nx*ny),np.float)
obtimes = nc_truth.variables['t'][:]
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))

if modelspace_local:
    thresh = 0.95
    nn = 0
    for i in range(nx):
        for j in range(ny):
            dist = cartdist(x1[i],y1[j],x,y,nc_climo.L,nc_climo.L)
            covloc2d = gaspcohn(dist/hcovlocal_scale)
            covlocal[nn] = covloc2d.ravel()
            #if nn == nx*ny/2+nx/2:
            #    import matplotlib.pyplot as plt
            #    plt.contourf(np.arange(nx),np.arange(ny),covloc2d,15)
            #    plt.colorbar()
            #    plt.show()
            #    raise SystemExit
            nn += 1
    evals, eigs = linalg.eigh(covlocal)
    evals = np.where(evals > 1.e-10, evals, 1.e-10)
    evalsum = evals.sum(); neig = 0; frac = 0.0
    while frac < thresh:
        frac = evals[nx*ny-neig-1:nx*ny].sum()/evalsum
        #print(neig,frac,evals[nx*ny-neig-1])
        neig += 1
    zz = (eigs*np.sqrt(evals/frac)).T
    zz = np.tile(zz,(1,2))
    z = zz[nx*ny-neig:nx*ny,:]
    print('# model space localization: neig = %s, variance expl = %5.2f%%' %
            (neig,100*frac))

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[0]
    models[nanal].timesteps = assim_timesteps

# for fixed ob network, initialize indxob,xob,yob
if fixed:
    mask = np.zeros((ny,nx),np.bool)
    nskip = int(nx/np.sqrt(nobs))
    mask[0:ny:nskip,0:nx:nskip] = True
    tmp = np.arange(0,nx*ny).reshape(ny,nx)
    indxob = tmp[mask.nonzero()].ravel()
    xob = x.ravel()[indxob]
    yob = y.ravel()[indxob]

# forward operator object
# (vertically integrated theta obs).
class Hop(object):
    def __init__(self,**kwargs):
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
    def calc(self,pv,indxob):
        pvspec = rfft2(pv)
        psispec = self.model.invert(pvspec=pvspec)
        pvavspec = (psispec[1]-psispec[0])/self.model.H
        pvav = irfft2(pvavspec)
        if self.filter_width > 0:
            if self.use_gaussian_filter:
                pvav = gaussian_filter(pvav, self.filter_width, mode='wrap')
            else:
                pvav = uniform_filter(pvav, size=self.filter_width, mode='wrap')
        return self.scalefact*pvav.ravel()[indxob]
fwdop = Hop(model=models[0],scalefact=scalefact,filter_width=filter_width,\
            use_gaussian_filter=use_gaussian_filter)
        
# initialize netcdf output file
if savedata is not None:
   nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
   nc.r = models[0].r
   nc.f = models[0].f
   nc.U = models[0].U
   nc.L = models[0].L
   nc.H = models[0].H
   nc.nanals = nanals
   nc.hcovlocal_scale = hcovlocal_scale
   nc.oberrstdev = oberrstdev_final
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

for ntime in range(nassim):

    oberrvar = oberrstdev**2*np.ones(nobs,np.float) + oberrextra**2

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
        indxob = rsobs.choice(nx*ny,nobs,replace=False,p=p.ravel())
        xob = x.ravel()[indxob]; yob = y.ravel()[indxob]
    # vertically integrated theta obs.
    pvob = fwdop.calc(pv_truth[ntime],indxob)
    pvob += rsobs.normal(scale=oberrstdev,size=nobs) # add ob errors
    # plot ob network
    #import matplotlib.pyplot as plt
    #plt.contourf(x,y,pv_truth[0,1,...],15)
    #plt.scatter(xob,yob,color='k')
    #plt.axis('off')
    #plt.show()
    #raise SystemExit

    # first-guess spread (need later to compute inflation factor)
    fsprd = ((pvens - pvens.mean(axis=0))**2).sum(axis=0)/(nanals-1)

    # compute forward operator.
    # hxens is ensemble in observation space.
    hxens = np.empty((nanals,nobs),np.float)
    for nanal in range(nanals):
        hxens[nanal] = fwdop.calc(pvens[nanal],indxob)
    hxensmean_b = hxens.mean(axis=0)
    obsprd = ((hxens-hxensmean_b)**2).sum(axis=0)/(nanals-1)
    # innov stats for background
    obfits = pvob - hxensmean_b
    obfits_b = (obfits**2).mean()
    obbias_b = obfits.mean()
    obsprd_b = obsprd.mean()
    pvensmean_b = pvens.mean(axis=0).copy()
    pverr_b = (scalefact*(pvensmean_b-pv_truth[ntime]))**2
    pvsprd_b = ((scalefact*(pvensmean_b-pvens))**2).sum(axis=0)/(nanals-1)

    if savedata is not None:
        pv_t[ntime] = pv_truth[ntime]
        pv_b[ntime,:,:,:] = scalefact*pvens
        pv_obs[ntime] = pvob
        x_obs[ntime] = xob
        y_obs[ntime] = yob

    # EnKF update
    # create 1d state vector.
    for nanal in range(nanals):
        xens[nanal] = np.ascontiguousarray(pvens[nanal].reshape((2*nx*ny)))
    # update state vector.
    if modelspace_local:
        xens =\
        enkf_update_modens(xens,hxens,fwdop,indxob,pvob,oberrvar,z,letkf=use_letkf)
    else:
        if not fixed or ntime == 0:
            covlocal = np.empty((nobs,nx*ny),np.float)
            obcovlocal = None
            if not use_letkf: obcovlocal = np.empty((nobs,nobs),np.float)
            for nob in range(nobs):
                dist = cartdist(xob[nob],yob[nob],x,y,nc_climo.L,nc_climo.L)
                covlocal[nob] = gaspcohn(dist/hcovlocal_scale).ravel()
                dist = cartdist(xob[nob],yob[nob],xob,yob,nc_climo.L,nc_climo.L)
                if not use_letkf: obcovlocal[nob] = gaspcohn(dist/hcovlocal_scale)
            covlocal = np.tile(covlocal,(1,2))
        xens =\
        enkf_update(xens,hxens,pvob,oberrvar,covlocal,obcovlocal=obcovlocal)
    # back to 3d state vector
    for nanal in range(nanals):
        pvens[nanal] = xens[nanal].reshape((2,ny,nx))
    t2 = time.time()
    if profile: print('cpu time for EnKF update',t2-t1)

    # forward operator on posterior ensemble.
    for nanal in range(nanals):
        hxens[nanal] = fwdop.calc(pvens[nanal],indxob)

    # ob space diagnostics
    hxensmean_a = hxens.mean(axis=0)
    obsprd_a = (((hxens-hxensmean_a)**2).sum(axis=0)/(nanals-1)).mean()
    # expected value is HPaHT (obsprd_a).
    obinc_a = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_a)).mean()
    # expected value is HPbHT (obsprd_b).
    obinc_b = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_b)).mean()
    # expected value R (oberrvar).
    omaomb = ((pvob-hxensmean_a)*(pvob-hxensmean_b)).mean()

    # posterior multiplicative inflation.
    pvensmean_a = pvens.mean(axis=0)
    pvprime = pvens-pvensmean_a
    asprd = (pvprime**2).sum(axis=0)/(nanals-1)
    if covinflate2 > 0:
        # Hodyss and Campbell (covinflate1=covinflate2=1 works best in perfect
        # model scenario)
        inc = pvensmean_a - pvensmean_b
        inflation_factor = covinflate1*asprd + \
        (asprd/fsprd)**2*((fsprd/nanals) + covinflate2*(2.*inc**2/(nanals-1)))
        inflation_factor = np.sqrt(inflation_factor/asprd)
    else: # RTPS inflation
        asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
        inflation_factor = 1.+covinflate1*(fsprd-asprd)/asprd
    pvprime = pvprime*inflation_factor
    pvens = pvprime + pvensmean_a

    # print out analysis error, spread and innov stats for background
    pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime]))**2
    pvsprd_a = ((scalefact*(pvensmean_a-pvens))**2).sum(axis=0)/(nanals-1)
    print("%s %g %g %g %g %g %g %g %g %g %g" %\
    (ntime,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
     np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
     obinc_b,obsprd_b,obinc_a,obsprd_a,omaomb/oberrvar.mean(),obbias_b))

    # save data.
    if savedata is not None:
        pv_a[ntime,:,:,:] = scalefact*pvens
        tvar[ntime] = obtimes[ntime]
        inf[ntime] = inflation_factor
        nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(pvens[nanal])
    t2 = time.time()
    if profile: print('cpu time for ens forecast',t2-t1)

if savedata: nc.close()
