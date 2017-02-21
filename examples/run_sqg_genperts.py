from sqgturb import SQG, rfft2, irfft2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
from netCDF4 import Dataset

# run SQG turbulence simulation, plotting results to screen and/or saving to
# netcdf file.

filename = sys.argv[1]
ncin = Dataset(filename)
savedata = sys.argv[2]

model = SQG(ncin['pv'][0],\
        nsq=ncin.nsq,f=ncin.f,dt=ncin.dt,U=ncin.U,H=ncin.H,\
        r=ncin.r,tdiab=ncin.tdiab,symmetric=ncin.symmetric,\
        diff_order=ncin.diff_order,diff_efold=ncin.diff_efold,threads=1)

N = ncin['pv'].shape[-1]
nsteps = ncin['pv'].shape[0]
diff_order_pert = 2
diff_efold_pert = 3600.
ktot = np.sqrt(model.ksqlsq)
ktotcutoff = np.pi*N/model.L
hyperdiff_pert =\
  np.exp((-model.dt/diff_efold_pert)*(ktot/ktotcutoff)**diff_order_pert)

#  initialize figure.
outputinterval = 10800. # interval between frames in seconds

nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
nc.r = model.r
nc.f = model.f
nc.U = model.U
nc.L = model.L
nc.H = model.H
nc.g = ncin.g; nc.theta0 = ncin.theta0
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
    pvspec = rfft2(ncin['pv'][nout])
    pvspec = pvspec - hyperdiff_pert*pvspec
    psispec = model.invert(pvspec)
    psi = irfft2(psispec)
    print nout, psi.min(), psi.max()
    im.set_data(psi[levplot])
    hr = ncin['t'][nout]/3600.
    print 'saving data at t = %g hours' % hr
    psivar[nout,:,:,:] = psi
    tvar[nout] = ncin['t'][nout]
    nc.sync()
    if nout >= nsteps: nc.close()
    nout = nout + 1
    return im,

# interval=0 means draw as fast as possible
ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
      init_func=initfig,interval=0,blit=True)
plt.show()
