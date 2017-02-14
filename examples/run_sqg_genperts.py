from sqgturb import SQG, rfft2, irfft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# run SQG turbulence simulation, plotting results to screen and/or saving to
# netcdf file.

# model parameters.
N = 128 # number of grid points in each direction (waves=N/2)
dt = 600 # time step in seconds
# efolding time scale (seconds) for smallest wave (N/2) in del**norder hyperdiffusion
norder = 8; diff_efold = 5400
dealias = True # dealiased with 2/3 rule?
nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
H = 10.e3 # lid height
U = 30.; L = 20.e6; r = 0.
# thermal relaxation time scale
tdiab = 10.*86400 # in seconds
symmetric = True # (asymmetric equilibrium jet with zero wind at sfc)
# parameter used to scale PV to temperature units.
scalefact = f*theta0/g

# create random noise
pv = np.random.normal(0,100.,size=(2,N,N)).astype(np.float32)
# add isolated blob on lid
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
x,y = np.meshgrid(x,y)
x = x.astype(np.float32); y = y.astype(np.float32)
pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
# remove area mean from each level.
for k in range(2):
    pv[k] = pv[k] - pv[k].mean()

# get OMP_NUM_THREADS (threads to use) from environment.
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# single or double precision
precision='single' # pyfftw FFTs twice as fast as double

# initialize qg model instance
model = SQG(pv,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=dealias,symmetric=symmetric,threads=threads,
            precision=precision)

diff_order_pert = 2
diff_efold_pert = model.dt
ktot = np.sqrt(model.ksqlsq)
ktotcutoff = np.pi*N/model.L
hyperdiff_pert =\
  np.exp((-model.dt/diff_efold_pert)*(ktot/ktotcutoff)**diff_order_pert)

#  initialize figure.
outputinterval = 21600. # interval between frames in seconds
tmin = 100.*86400. # time to start saving data (in days)
tmax = 200.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
model.timesteps = int(outputinterval/model.dt)
savedata = 'sqg_N%s_perts2.nc' % N # save data plotted in a netcdf file.
#savedata = None # don't save data
plot = True # animate data as model is running?


if savedata is not None:
    from netCDF4 import Dataset
    nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
    nc.r = model.r
    nc.f = model.f
    nc.U = model.U
    nc.L = model.L
    nc.H = model.H
    nc.g = g; nc.theta0 = theta0
    nc.nsq = model.nsq
    nc.tdiab = model.tdiab
    nc.dt = model.dt
    nc.diff_efold = diff_efold_pert
    nc.diff_order = diff_order_pert
    nc.symmetric = int(model.symmetric)
    nc.dealias = int(model.dealias)
    x = nc.createDimension('x',N)
    y = nc.createDimension('y',N)
    z = nc.createDimension('z',2)
    t = nc.createDimension('t',None)
    psivar =\
    nc.createVariable('psi',np.float32,('t','z','y','x'),zlib=True)
    psivar.units = 'm**2/s'
    xvar = nc.createVariable('x',np.float32,('x',))
    xvar.units = 'meters'
    yvar = nc.createVariable('y',np.float32,('y',))
    yvar.units = 'meters'
    zvar = nc.createVariable('z',np.float32,('z',))
    zvar.units = 'meters'
    tvar = nc.createVariable('t',np.float32,('t',))
    tvar.units = 'seconds'
    xvar[:] = np.arange(0,model.L,model.L/N)
    yvar[:] = np.arange(0,model.L,model.L/N)
    zvar[0] = 0; zvar[1] = model.H

levplot = 1; nout = 0
if plot:
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0, bottom=0.0, top=1., right=1.)
    vmin = -3.e5; vmax = 3.e5
    def initfig():
        global im
        ax = fig.add_subplot(111)
        ax.axis('off')
        pvspec = model.pvspec - hyperdiff_pert*model.pvspec
        psispec = model.invert(pvspec)
        psi = irfft2(psispec)
        im = ax.imshow(psi[levplot],cmap=plt.cm.bwr,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        return im,
    def updatefig(*args):
        global nout
        model.advance()
        t = model.t
        pvspec = model.pvspec - hyperdiff_pert*model.pvspec
        psispec = model.invert(pvspec)
        psi = irfft2(psispec)
        hr = t/3600.
        print hr, psi.min(), psi.max()
        im.set_data(psi[levplot])
        if savedata is not None and t >= tmin:
            print 'saving data at t = t = %g hours' % hr
            psivar[nout,:,:,:] = psi
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1
        return im,

    # interval=0 means draw as fast as possible
    ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
          init_func=initfig,interval=0,blit=True)
    plt.show()
else:
    t = 0.0
    while t < tmax:
        model.advance()
        t = model.t
        pv = irfft2(model.pvspec)
        hr = t/3600.
        spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
        print hr,spd.max(),scalefact*pv.min(),scalefact*pv.max()
        if savedata is not None and t >= tmin:
            print 'saving data at t = t = %g hours' % hr
            pvvar[nout,:,:,:] = pv
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1
