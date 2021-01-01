import matplotlib
matplotlib.use('qt4agg')
from sqgturb import SQG, rfft2, irfft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from netCDF4 import Dataset

# run SQG turbulence simulation, optionally plotting results to screen and/or saving to
# netcdf file.

# model parameters (start from end of existing run).

filenamein = 'sqg_N512_6hrly.nc'
nc = Dataset(filenamein,'a')

N = len(nc.dimensions['x']) # size of domain
dt = nc.dt # time step
norder = nc.diff_order # order of hyperdiffusion
diff_efold = nc.diff_efold # efolding time on shortest wave
dealias = True # dealiased with 2/3 rule?

# Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
# Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
# for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
# corresponding to ekman depth of 141-316 m over ocean.
# spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
# H=10km, f=0.0001, dek=100m.
r = nc.r # applied only at surface if symmetric=False
nsq = nc.nsq; f=nc.f; g = nc.g; theta0 = nc.theta0
H = nc.H # lid height
U = nc.U # jet speed
Lr = np.sqrt(nsq)*H/f # Rossby radius
L = nc.L
# thermal relaxation time scale
tdiab = nc.tdiab # in seconds
symmetric = nc.symmetric # (if False, asymmetric equilibrium jet with zero wind at sfc)
# parameter used to scale PV to temperature units.
scalefact = f*theta0/g

# create random noise
pv = nc['pv'][-1].squeeze()
t = nc['t'][-1]
threads = int(os.getenv('OMP_NUM_THREADS','1'))

# single or double precision
precision='single' # pyfftw FFTs twice as fast as double

# initialize qg model instance
model = SQG(pv,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            dealias=dealias,symmetric=symmetric,threads=threads,
            precision=precision,tstart=t)

#  initialize figure.
outputinterval = 86400./4. # interval between frames in seconds
tmax = t + 1.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
model.timesteps = int(outputinterval/model.dt)
savedata = filenamein
#savedata = None # don't save data
plot = False # animate data as model is running?

if savedata is not None:
    pvvar = nc['pv']
    tvar = nc['t']

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
        if savedata is not None:
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
    while t < tmax:
        model.advance()
        t = model.t
        pv = irfft2(model.pvspec)
        hr = t/3600.
        spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
        print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
        if savedata is not None:
            print('saving data at t = t = %g hours' % hr)
            pvvar[nout,:,:,:] = pv
            tvar[nout] = t
            nc.sync()
            if t >= tmax: nc.close()
            nout = nout + 1
