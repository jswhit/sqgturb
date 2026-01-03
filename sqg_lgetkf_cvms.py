import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from sqgturb import SQG, fft_forward, fft_backward, cartdist, lgetkf_ms, gaspcohn, newDistArrayGrid, newDistArraySpec, MPI
from pyfftw.interfaces import numpy_fft
rfft2 = numpy_fft.rfft2
irfft2 = numpy_fft.irfft2

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()

# LGETKF cycling for SQG turbulence model with boundary temp obs,
# multi-scale ob space horizontal localization, no vertical localization.
# cross-validation update (no inflation).
# Random observing network.

if rank == 0:
    if len(sys.argv) == 1:
       msg="""
    python sqg_lgetkf_cv.py hcovlocal_scale covinflate>
       hcovlocal_scales = horizontal localization scale(s) in km
       band_cutoffs = filter waveband cutoffs 
       crossbandcov_facts = cross-band covariance factors
       """
       raise SystemExit(msg)
    
    # horizontal covariance localization length scale in meters.
    hcovlocal_scales = eval(sys.argv[1])
    nlscales = len(hcovlocal_scales)
    band_cutoffs = eval(sys.argv[2])
    nband_cutoffs = len(band_cutoffs)
    if nband_cutoffs != nlscales-1:
        raise SystemExit('band_cutoffs should be one less than hcovlocal_scales')
    crossbandcov_facts = eval(sys.argv[3])
    if len(crossbandcov_facts) != nband_cutoffs:
        raise SystemExit('band_cutoffs and crossbandcov_facts should be same length')
else:
    hcovlocal_scales=None; band_cutoffs=None; nband_cutoffs=None; nlscales=None; crossbandcov_facts=None

hcovlocal_scales = comm.bcast(hcovlocal_scales, root=0)
band_cutoffs = comm.bcast(band_cutoffs, root=0)
nband_cutoffs = comm.bcast(nband_cutoffs, root=0)
nlscales = comm.bcast(nlscales, root=0)
crossbandcov_facts = comm.bcast(crossbandcov_facts, root=0)
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
ngroups = nanals//2  # number of groups for cross-validation (ngroups=nanals//N is "leave N out")

oberrstdev = 1. # ob error standard deviation in K

# nature run created using sqg_run.py.
filename_climo = 'sqgu20_dek0_N96_6hrly.nc' # file name for forecast model climo
# perfect model
filename_truth = 'sqgu20_dek0_N96_6hrly.nc' # file name for nature run to draw obs
#filename_truth = 'sqg_N256_N96_12hrly.nc' # file name for nature run to draw obs

if rank==0:
    print('# filename_modelclimo=%s' % filename_climo)
    print('# filename_truth=%s' % filename_truth)

# fix random seed for reproducibility.
rsobs = np.random.RandomState(42) # fixed seed for observations
rsics = np.random.RandomState(24) # fixed seed for initial conditions

# get model info
if rank==0:
    nc_climo = Dataset(filename_climo)
    x = nc_climo.variables['x'][:]
    y = nc_climo.variables['y'][:]
    nx = len(x); ny = len(y)
    x, y = np.meshgrid(x, y)
else: 
    nx = None; ny = None
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
if rank != 0:
    x = np.empty((ny,nx), np.float32)
    y = np.empty((ny,nx), np.float32)
comm.Bcast(x, root=0)
comm.Bcast(y, root=0)

pvens = np.empty((nanals,2,ny,nx),np.float32)
if rank == 0:
    # parameter used to scale PV to temperature units.
    scalefact = nc_climo.f*nc_climo.theta0/nc_climo.g
    dt = nc_climo.dt
    nsq = nc_climo.nsq
    f = nc_climo.f
    r = nc_climo.r
    U = nc_climo.U
    H = nc_climo.H
    tdiab = nc_climo.tdiab
    diff_order = nc_climo.diff_order
    if diff_efold == None: diff_efold=nc_climo.diff_efold
    tstart = 0
    if not read_restart:
        pv_climo = nc_climo.variables['pv']
        indxran = rsics.choice(pv_climo.shape[0],size=nanals,replace=False)
        for nanal in range(nanals):
            pvens[nanal] = pv_climo[indxran[nanal]]
    else:
        ncinit = Dataset('%s_restart.nc' % exptname, mode='r', format='NETCDF4_CLASSIC')
        ncinit.set_auto_mask(False)
        pvens[:] = ncinit.variables['pv_b'][-1,...]/scalefact
        tstart = ncinit.variables['t'][-1]
        #for nanal in range(nanals):
        #    print(nanal, pvens[nanal].min(), pvens[nanal].max())
else:
    tdiab=None; dt=None; f=None; U=None; H=None; nsq=None; scalefact=None; diff_order=None; diff_efold=None; r=None; tstart=None
comm.Bcast(pvens, root=0)
dt = comm.bcast(dt, root=0)
diff_efold = comm.bcast(diff_efold, root=0)
scalefact = comm.bcast(scalefact, root=0)
nsq = comm.bcast(nsq, root=0)
tstart = comm.bcast(tstart, root=0)
f = comm.bcast(f, root=0)
U = comm.bcast(U, root=0)
H = comm.bcast(H, root=0)
tdiab = comm.bcast(tdiab, root=0)
diff_order = comm.bcast(diff_order, root=0)
diff_efold = comm.bcast(diff_efold, root=0)
dt = comm.bcast(dt, root=0)
r = comm.bcast(r, root=0)

models = []
for nanal in range(nanals):
    models.append(\
    SQG(pvens[nanal],dt=dt,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,diff_order=diff_order,diff_efold=diff_efold,threads=threads))
pvspecens = np.empty((nanals,2,)+(models[0].pvspec).global_shape,models[0].pvspec.dtype)
for nanal in range(nanals):
    pvspecens[nanal] = models[nanal].pv_spec()
if rank==0 and read_restart: ncinit.close()

hcovlocal_scales_km = [lscale/1000. for lscale in hcovlocal_scales]
if rank==0:
    print("# hcovlocal=%s diff_efold=%s nanals=%s ngroups=%s" %\
         (repr(hcovlocal_scales_km),diff_efold,nanals,ngroups))
    print('# band_cutoffs=%s crossbandcov_facts=%s' % (repr(band_cutoffs),repr(crossbandcov_facts)))

# each ob time nobs ob locations are randomly sampled (without
# replacement) from the model grid
#nobs = nx*ny//6 # number of obs to assimilate (randomly distributed)
#nobs = 2*nx*ny//24 # 768
nobs = 2*nx*ny//18 # 1024
#nobs = 2*nx*ny//12 # 1536
#nobs = 2*nx*ny//9 # 2048

# nature run
if rank == 0:
    nc_truth = Dataset(filename_truth)
    pv_truth = nc_truth.variables['pv'][:]
    obtimes = nc_truth.variables['t'][:]
    nc_truth.close()
    ntimes = pv_truth.shape[0]
    # set up arrays for obs and localization function
    print('# random network nobs = %s' % nobs)
else:
    ntimes = None
ntimes = comm.bcast(ntimes, root=0)
if rank != 0:
    pv_truth = np.empty((ntimes, 2, ny ,nx), np.float32)
    obtimes = np.empty(ntimes, np.float32)
comm.Bcast(pv_truth, root=0)
comm.Bcast(obtimes, root=0)

oberrvar = oberrstdev**2*np.ones(nobs,np.float32)
pvob = np.empty(nobs,np.float32)
covlocal = np.empty((ny,nx),np.float32)
covlocal_tmp = np.empty((nlscales,nobs,nx*ny),np.float32)

if read_restart:
    timeslist = obtimes.tolist()
    ntstart = timeslist.index(tstart)
    print('# restarting from %s.nc ntstart = %s' % (exptname,ntstart))
else:
    ntstart = 0
assim_interval = obtimes[1]-obtimes[0]
assim_timesteps = int(np.round(assim_interval/models[0].dt))
if rank == 0:
    print('# assim interval = %s secs (%s time steps)' % (assim_interval,assim_timesteps))
    print('# ntime,pverr_a,pvsprd_a,pverr_b,pvsprd_b,obfits_b,osprd_b+R,obbias_b,tr(P^a)/tr(P^b)')

# initialize model clock
for nanal in range(nanals):
    models[nanal].t = obtimes[ntstart]

# initialize output file.
if savedata is not None and rank == 0:
   nc = Dataset('%s.nc' % exptname, mode='w', format='NETCDF4_CLASSIC')
   nc.r = models[0].r[0]
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
k = N*np.fft.rfftfreq(N)
l = N*np.fft.fftfreq(N)
imax = len(k); jmax = len(l)
k,l = np.meshgrid(k,l)
ktotsq = (k**2+l**2).astype(np.int32)
jmax,imax = ktotsq.shape
ktot = np.sqrt(ktotsq)
ktotmax = (N//2)+1
# grid points updated on this task
ix = np.arange(models[0].N**2).reshape(N,N)
npts_dist = ix[models[0].local_slice_grid].ravel()

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
            dist = cartdist(xob[nob],yob[nob],x,y,models[0].L,models[0].L)
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

        #pv_dist = newDistArrayGrid(models[0].FFT) 
        #pvspec = np.zeros((nanals,2,)+models[0].pvspec.global_shape, models[0].pvspec.dtype)
        #for nanal in range(nanals):
        #    for k in range(2):
        #        pv_dist[k,...] = pvpert[nanal,k,...][pv_dist.local_slice()]
        #    pvspec_dist = fft_forward(models[0].FFT, pv_dist)
        #    for k in range(2):
        #        pvspec[nanal,k,...][pvspec_dist.local_slice()] = pvspec_dist[k]
        #MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,np.ascontiguousarray(pvspec),op=MPI.SUM)
        #pvspec = rfft2(pvpert)
        #if rank==0:
        #    pvspec = rfft2(pvpert)
        #else:
        #    pvspec = np.empty((nanals,2,)+models[0].pvspec.global_shape, models[0].pvspec.dtype)
        #comm.Bcast(np.ascontiguousarray(pvspec), root=0)

        pvspec_dist = newDistArraySpec(models[0].FFT) 
        for n,cutoff in enumerate(band_cutoffs):
            #filtfact = np.exp(-(ktot/cutoff)**8)
            #pvfiltspec = filtfact*pvspecens
            pvfiltspec = np.where(ktot < cutoff, pvspecens, 0.+0.j)

            pvfilt = np.zeros_like(pvpert)
            for nanal in range(nanals):
                for k in range(2):
                    pvspec_dist[k,...] = pvfiltspec[nanal,k,...][pvspec_dist.local_slice()]
                pv_dist = fft_backward(models[0].FFT, pvspec_dist)
                for k in range(2):
                    pvfilt[nanal,k,...][pv_dist.local_slice()] = pv_dist[k]
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,np.ascontiguousarray(pvfilt),op=MPI.SUM)
            #if rank==0:
            #    pvfilt = irfft2(pvfiltspec)
            #else:
            #    pvfilt = np.empty_like(pvens)
            #comm.Bcast(np.ascontiguousarray(pvfilt), root=0)

            pvens_filtered_lst.append(pvfilt-pvfilt_save)
            #plt.figure()
            #plt.imshow((pvfilt-pvfilt_save)[0,0,...],cmap=plt.cm.bwr)
            #plt.title('scale = %s' % n)
            pvfilt_save=pvfilt
        pvsum = np.zeros_like(pvpert)
        for n in range(nband_cutoffs):
            pvsum += pvens_filtered_lst[n]
        #plt.figure()
        #plt.imshow((pvpert-pvsum)[0,0,...],cmap=plt.cm.bwr)
        #plt.title('scale = %s' % nlscales)
        pvens_filtered_lst.append(pvpert-pvsum)
        #plt.show()
        #raise SystemExit
    pvens_filtered = np.asarray(pvens_filtered_lst)
    pvens = np.dot(pvens_filtered.T,crossband_covmat).T
    pvens += pvensmean_b  # mean added back to all scales.

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

    # update state vector.

    # hxens,pvob are in PV units, xens is not
    xens_updated = np.zeros_like(xens) 
    xens = lgetkf_ms(nlscales,xens,hxprime,pvob-hxensmean_b,oberrvar,covlocal_tmp,ngroups=ngroups,npts_dist=npts_dist)
    xens_updated[:,:,npts_dist] = xens[:,:,npts_dist]
    comm.Allreduce(MPI.IN_PLACE, xens_updated, op=MPI.SUM)
    xens = xens_updated

    # back to 3d state vector
    pvens = xens.reshape((nlscales*nanals,2,ny,nx))
    pvensmean_a = pvens.mean(axis=0) 
    pvens_filtered = pvens - pvensmean_a
    pvens_filtered = pvens_filtered.reshape(nlscales,nanals,2,ny,nx)
    pvprime = np.dot(pvens_filtered.T,crossband_covmatr).T
    pvens = pvprime.sum(axis=0) + pvensmean_a
    t2 = time.time()
    if profile and rank == 0: print('cpu time for EnKF update',t2-t1)

    pvensmean_a = pvens.mean(axis=0)
    pvprime = pvens - pvensmean_a
    asprd = (pvprime**2).sum(axis=0)/(nanals-1)
    asprd_over_fsprd = asprd.mean()/fsprd.mean()

    # print out analysis error, spread and innov stats for background
    pverr_a = (scalefact*(pvensmean_a-pv_truth[ntime+ntstart]))**2
    pvsprd_a = ((scalefact*(pvensmean_a-pvens))**2).sum(axis=0)/(nanals-1)
    if rank == 0:
        print("%s %g %g %g %g %g %g %g %g" %\
        (ntime+ntstart,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
         np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
         np.sqrt(obfits_b),np.sqrt(obsprd_b+oberrstdev**2),obbias_b,
         asprd_over_fsprd))

    # save data.
    if savedata is not None and rank == 0:
        if savedata == 'restart' and ntime != nassim-1:
            pass
        else:
            pv_a[ntime,:,:,:] = scalefact*pvens
            tvar[ntime] = obtimes[ntime+ntstart]
            nc.sync()

    # run forecast ensemble to next analysis time
    t1 = time.time()
    for nanal in range(nanals):
        pvens[nanal] = models[nanal].advance(timesteps=assim_timesteps,pv=pvens[nanal])
        pvspecens[nanal] = models[nanal].pv_spec()
    t2 = time.time()
    if profile and rank == 0: print('cpu time for ens forecast',t2-t1)
    if not np.all(np.isfinite(pvens)):
        raise SystemExit('non-finite values detected after forecast, stopping...')

    # compute spectra of error and spread
    if ntime >= nassim_spinup and rank == 0:
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

if savedata and rank == 0: nc.close()

if ncount and rank == 0:
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
    plt.loglog(wavenums[1:-1],pvspec_err[1:-1],color='r',label='error')
    plt.loglog(wavenums[1:-1],pvspec_sprd[1:-1],color='b',label='spread')
    plt.title('expt=%s' % exptname)
    plt.legend()
    plt.savefig('errorspread_spectra_cv_%s.png' % exptname)
