from netCDF4 import Dataset
import numpy as np
import sys
from sqg import rfft2, irfft2
from scipy import ndimage

def block_mean(ar, fact):
    # downsample 2d array by averaging fact x fact blocks
    # requires scipy.ndimage
    #assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions,
                       index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res

def spectrunc(specarr,N):
    fact = float(N)/float(specarr.shape[1])
    specarr_trunc = np.zeros((2, N, N/2+1), specarr.dtype)
    specarr_trunc[:,0:N/2,0:N/2] = fact**2*specarr[:,0:N/2,0:N/2]
    specarr_trunc[:,-N/2:,0:N/2] = fact**2*specarr[:,-N/2:,0:N/2]
    return specarr_trunc

# spectrally truncate data in filenamein, write to filenameout on Nout x Nout
# grid.
filenamein = sys.argv[1]
filenameout = sys.argv[2]
Nout = int(sys.argv[3])

ncin = Dataset(filenamein)
nc = Dataset(filenameout, mode='w', format='NETCDF4_CLASSIC')
nc.r = ncin.r
nc.f = ncin.f
nc.U = ncin.U
nc.L = ncin.L
nc.H = ncin.H
nc.g = ncin.g; nc.theta0 = ncin.theta0
nc.nsq = ncin.nsq
nc.tdiab = ncin.tdiab
nc.dt = ncin.dt
nc.diff_efold = ncin.diff_efold
nc.diff_order = ncin.diff_order
nc.symmetric = ncin.symmetric
nc.dealias = ncin.dealias
x = nc.createDimension('x',Nout)
y = nc.createDimension('y',Nout)
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
xvar[:] = np.arange(0,ncin.L,ncin.L/Nout)
yvar[:] = np.arange(0,ncin.L,ncin.L/Nout)
zvar[0] = 0; zvar[1] = ncin.H
N = ncin['pv'].shape[-1]
nskip = N/Nout
for n in range(len(ncin.dimensions['t'])):
    tvar[n] = ncin['t'][n]
    pvin = ncin['pv'][n]
    # downsample by averaging nskip x nskip blocks of pixels
    pvout = np.empty((2,Nout,Nout), dtype=pvin.dtype)
    for k in range(2):
        pvout[k,:,:] = block_mean(pvin[k],nskip)
    # spectrally truncate.
    #pvout = irfft2(spectrunc(rfft2(pvin),Nout))
    pvvar[n] = pvout
    print n,tvar[n]/86400.,pvin.shape,pvout.shape,pvin.min(),pvin.max(),pvout.min(),pvout.max()
