import matplotlib
matplotlib.use('qt4agg')
from sqgturb import SQG, rfft2, irfft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# run SQG turbulence simulation, optionally plotting results to screen and/or saving to
# netcdf file.

# model parameters.

#N = 512 # number of grid points in each direction (waves=N/2)
#dt = 90 # time step in seconds
#diff_efold = 1800. # time scale for hyperdiffusion at smallest resolved scale

N = 192
dt = 300
diff_efold = 86400./8.

N = 128
dt = 600
diff_efold = 86400./3.

N = 96 
dt = 900
diff_efold = 86400./3.

#N = 64
#dt = 1200
#diff_efold = 86400.

norder = 8 # order of hyperdiffusion
dealias = True # dealiased with 2/3 rule?

# Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
# Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
# for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
# corresponding to ekman depth of 141-316 m over ocean.
# spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
# H=10km, f=0.0001, dek=100m.
dek = 0 # applied only at surface if symmetric=False
nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
H = 10.e3 # lid height
r = dek*nsq/f
U = 30 # jet speed
Lr = np.sqrt(nsq)*H/f # Rossby radius
L = 20.*Lr
# thermal relaxation time scale
tdiab = 10.*86400 # in seconds
symmetric = True # (if False, asymmetric equilibrium jet with zero wind at sfc)
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

#  initialize figure.
outputinterval = 2.*10800. # interval between frames in seconds
tmin = 100.*86400. # time to start saving data (in days)
tmax = 300.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
model.timesteps = int(outputinterval/model.dt)
savedata = 'sqg_N%s_6hrly.nc' % N # save data plotted in a netcdf file.
#savedata = None # don't save data
plot = False # animate data as model is running?

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
    nc.diff_efold = model.diff_efold
    nc.diff_order = model.diff_order
    nc.symmetric = int(model.symmetric)
    nc.dealias = int(model.dealias)
    x = nc.createDimension('x',N)
    y = nc.createDimension('y',N)
    z = nc.createDimension('z',2)
    t = nc.createDimension('t',None)
    pvvar =\
    nc.createVariable('pv',np.float32,('t','z','y','x'),zlib=True)
    pvvar.units = 'K'
    # pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
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
    vmin = scalefact*model.pvbar[levplot].min()
    vmax = scalefact*model.pvbar[levplot].max()
    def initfig():
        global im
        ax = fig.add_subplot(111)
        ax.axis('off')
        pv = irfft2(model.pvspec[levplot])  # spectral to grid
        im = ax.imshow(scalefact*pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        return im,
    def updatefig(*args):
        global nout
        model.advance()
        t = model.t
        pv = irfft2(model.pvspec)
        hr = t/3600.
        spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
        print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
        im.set_data(scalefact*pv[levplot])
        if savedata is not None and t >= tmin:
            print('saving data at t = t = %g hours' % hr)
            pvvar[nout,:,:,:] = pv
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
        print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
        if savedata is not None and t >= tmin:
            print('saving data at t = t = %g hours' % hr)
            pvvar[nout,:,:,:] = pv
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1
