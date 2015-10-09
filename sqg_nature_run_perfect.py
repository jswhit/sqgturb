import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sqg import SQG, rfft2, irfft2
from netCDF4 import Dataset

# run eady turbulence simulation, plotting results to screen and/or saving to
# netcdf file.

# model parameters.
N = 64   # number of points in each direction.
# Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
# Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
# for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
# corresponding to ekman depth of 141-316 m over ocean.
# spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
# H=10km, f=0.0001, dek=100m.
dek = 0.
nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
r = dek*nsq/f
# model time step.
#dt = 128.*1200./N
dt = 3600.
U = 30 # jet speed
L = 20.e6 # domain size (square)
H = 10.e3 # lid height
Lr = np.sqrt(nsq)*H/f # Rossby radius
tdiab =  10.*Lr/U # thermal relaxtion time scale in advective time scale units.
# efolding time scale (seconds) for smallest wave (N/2) in del**norder hyperdiffusion
norder = 8
efold = 9000.
# parameter used to scale PV to temperature units.
scalefact = f*theta0/g
symmetric = True

# fix random seed for reproducibility.
np.random.seed(4)

# create random noise
pv = np.random.normal(0,500.,size=(2,N,N)).astype(np.float32)
# add isolated blob on lid
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
x,y = np.meshgrid(x,y)
x = x.astype(np.float32); y = y.astype(np.float32)
pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
# remove area mean from each level.
for k in range(2):
    pv[k] = pv[k] - pv[k].mean()

# initialize qg model instance
model =\
SQG(pv,nsq=nsq,f=f,dt=dt,U=U,H=H,r=r,tdiab=tdiab,diff_order=norder,diff_efold=efold,symmetric=symmetric)

#  initialize figure.
outputinterval = 21600. # interval between frames in seconds
tmin = 100.*86400. # time to start saving data (in days)
tmax = 700.*86400. # time to stop (in days)
nsteps = int(tmax/dt) # number of time steps to animate
savedata = True # save data plotted in a netcdf file.

fig = plt.figure(figsize=(8,8))
def initfig():
    global im,txt,txt2
    ax = fig.add_subplot(111)
    ax.axis('off')
    pv = irfft2(model.pvspec)  # spectral to grid
    im =\
    ax.imshow(scalefact*pv[1],interpolation='nearest',origin='lower',vmin=-10000*scalefact,vmax=10000*scalefact)
    txt = ax.text(0.05,0.95,'PV(Z=H) (%s x %s) time t = %g hrs' %\
          (N,N,float(model.t/3600.)),color='k',fontweight='bold',fontsize=16,transform=ax.transAxes)
    return im,txt

if savedata:
   filename  = 'data/sqg_N%s.nc' % N # name of netcdf file.
   nc = Dataset(filename, mode='w', format='NETCDF4_CLASSIC')
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
   x = nc.createDimension('x',N)
   y = nc.createDimension('y',N)
   z = nc.createDimension('z',2)
   t = nc.createDimension('t',None)
   pvvar =\
   nc.createVariable('pv',np.float32,('t','z','y','x'),zlib=True)
   pvvar.units = 'K'
   # eady pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
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

nout = 0
def updatefig(*args):
    global nout
    model.timestep()
    t = model.t
    if t % outputinterval == 0.0:
        pv = irfft2(model.pvspec)
        hr = t/3600.
        spd = np.sqrt(model.u**2+model.v**2)
        umean = model.u.mean(axis=-1)
        print hr,scalefact*pv.min(),scalefact*pv.max(),\
        spd[0].max(),spd[1].max(),umean[0].min(),umean[0].max(),\
        umean[1].min(),umean[1].max()
        im.set_data(scalefact*pv[1])
        txt.set_text('PV(Z=H) (%s x %s) time t = %g hrs' % (N,N,hr))
        if savedata and t >= tmin:
            print 'saving data at t = t = %g hours' % hr
            pvvar[nout,:,:,:] = pv
            tvar[nout] = t
            nc.sync()
            if t >= tmax:
                nc.close()
            nout = nout + 1
    return im,txt

#interval=0 means draw as fast as possible
ani = animation.FuncAnimation(fig, updatefig, nsteps, repeat=False,\
      init_func=initfig,interval=0,blit=False)
plt.show()
