import numpy as np
from scipy.ndimage import gaussian_filter

class RandomPatternSample:
    def __init__(self, ncvar, temporal_corr_efold=0, dt=600, scale = 1.0):
        self.dt = dt
        self.ncvar = ncvar
        self.ntimes = ncvar.shape[0]
        self.tcorr = temporal_corr_efold
        self.scale = scale
        self.N = ncvar.shape[-1]
        if self.tcorr == 0:
            self.lag1corr = 0.
        else:
            self.lag1corr = np.exp(-1)**(self.dt/self.tcorr)
        nt = np.random.randint(0,self.ntimes)
        self.pattern = self.scale*self.ncvar[nt]

    def evolve(self):
        nt = np.random.randint(0,self.ntimes)
        newpattern = self.scale*self.ncvar[nt]
        # blend new pattern with old pattern.
        self.pattern = \
        np.sqrt(1.-self.lag1corr**2)*newpattern + \
        self.lag1corr*self.pattern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset
    import sys
    filename = sys.argv[1]
    scale = 5.
    nc = Dataset(filename)
    psi = nc['psi']
    rp=RandomPattern(psi,temporal_corr_efold=1800.,dt=600.,scale=scale)
    # plot random samples.
    x = rp.pattern
    minmax = max(np.abs(x[1].min()), np.abs(x[1].max()))
    ntimes = 10
    for n in range(ntimes):
        rp.evolve()
        plt.figure()
        plt.imshow(rp.pattern[1],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    ntimes = 1000
    x = rp.pattern[1]
    lag1cov = np.zeros(x.shape, x.dtype)
    lag1var = np.zeros(x.shape, x.dtype)
    spatial_cov = np.zeros(x.shape, x.dtype)
    for nt in range(ntimes):
        xold = x.copy()
        rp.evolve()
        x = rp.pattern[1]
        lag1cov = lag1cov + x*xold/(ntimes-1)
        lag1var = lag1var + x*x/(ntimes-1)
        spatial_cov = spatial_cov + x[rp.N/2,rp.N/2]*x/(ntimes-1)
    plt.figure()
    x = np.arange(rp.N)-rp.N/2
    plt.plot(x,0.5*(spatial_cov[:,rp.N/2]+spatial_cov[rp.N/2,:]),'r')
    plt.axhline(0); plt.axvline(0)
    plt.show()
    lag1corr = lag1cov/lag1var
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',rp.lag1corr
