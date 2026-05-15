from sqgturb import SQG, SQGState
import jax.numpy as jnp
import numpy as np

# run SQG turbulence simulation, save data to netcdf file

# model parameters.

#N = 512 # number of grid points in each direction (waves=N/2)
#dt = 90 # time step in seconds
#diff_efold = 1800. # time scale for hyperdiffusion at smallest resolved scale

#N = 192
#dt = 300
#diff_efold = 86400./8.
 
#N = 128
#dt = 600
#diff_efold = 86400./3.

#N = 96
#dt = 900
#diff_efold = 86400./2.

N = 64
dt = 1200    
diff_efold = 86400. 

norder = 8 # order of hyperdiffusion

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
U = 20 # jet speed
Lr = np.sqrt(nsq)*H/f # Rossby radius
L = 20.*Lr
# thermal relaxation time scale
tdiab = 10.*86400 # in seconds
# parameter used to scale PV to temperature units.
scalefact = f*theta0/g

# create random noise
rs = np.random.RandomState(42) # fixed seed
pv = rs.normal(0,100.,size=(2,N,N)).astype(np.float32)
# add isolated blob on lid
nexp = 20
x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
x,y = np.meshgrid(x,y)
x = x.astype(np.float32); y = y.astype(np.float32)
pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
# remove area mean from each level.
for k in range(2):
    pv[k] = pv[k] - pv[k].mean()

# initialize sqg model instance
model = SQG(N,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
            diff_order=norder,diff_efold=diff_efold,
            precision='single')

#  initialize figure.
outputinterval = 6.*3600. # interval between frames in seconds
tmin = 100.*86400. # time to start saving data (in days)
tmax = 430.*86400. # time to stop (in days)
nsteps = int(tmax/outputinterval) # number of time steps to animate
# set number of timesteps to integrate for each call to model.advance
ntimesteps = int(outputinterval/model.dt)
savedata = 'sqgu%s_N%s_6hrly_24hdiff_jax.nc' % (U,N) # save data plotted in a netcdf file.

if savedata is not None:
    from netCDF4 import Dataset
    nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
    nc.r = model.r[0]
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

t = 0.0
state = SQGState(
        pvspec=jnp.fft.rfft2(jnp.array(pv)),
        t=t)
nout = 0
while t < tmax:
    state, pv = model.advance(state,timesteps=ntimesteps)
    t = state.t
    hr = t/3600.
    print(hr,scalefact*pv.min(),scalefact*pv.max())
    if savedata is not None and t >= tmin:
        print('saving data at t = %g hours' % hr)
        pvvar[nout,:,:,:] = np.asarray(pv)
        tvar[nout] = t
        nc.sync()
        if t >= tmax: nc.close()
        nout = nout + 1
