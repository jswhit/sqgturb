import numpy as np
from scipy.ndimage import gaussian_filter

class RandomPattern:
    def __init__(self, spatial_corr_efold, temporal_corr_efold, L, N, dt, \
            nsamples=1, stdev=1.0):
        """
        define an ensemble of random patterns with specified temporal
        and spatial covariance structure by applying Gaussian blur to
        white noise.
        spatial_corr_efold:  horizontal efolding scale for 
        isotropic spatial correlation structure.
        temporal_corr_efold:  efolding time scale for temporal
        correlation.
        L: size of square domain (m)
        N: number of grid points in each periodic direction
        dt: time step to evolve paptern.
        nsamples:  number of ensemble members.
        stdev:  spatial standard deviation (amplitude).
        """
        self.hcorr = spatial_corr_efold
        self.tcorr = temporal_corr_efold
        self.dt = dt
        self.lag1corr = np.exp(-1)**(self.dt/self.tcorr)
        self.L = L
        self.stdev = stdev
        self.nsamples = nsamples
        self.N = N
        # initialize patterns.
        # generate white noise.
        self.pattern = self.stdev*np.random.normal(\
                       size=(self.nsamples,self.N,self.N))
        # apply gaussian filter
        self.filter_stdev = self.hcorr*self.N/(self.L*np.sqrt(4.))
        for n in range(nsamples):
            self.pattern[n] = gaussian_filter(self.pattern[n], 
            self.filter_stdev, order=0, output=None,
            mode='wrap', cval=0.0, truncate=4.0)
        # restore variance removed by gaussian blur.
        self.pattern =\
        self.pattern*(self.filter_stdev*2.*np.sqrt(np.pi))

    def evolve(self):
        """
        evolve random patterns one time step
        """
        # generate white noise.
        newpattern = self.stdev*np.random.normal(\
                     size=(self.nsamples,self.N,self.N))
        # apply gaussian filter
        for n in range(self.nsamples):
            newpattern[n] = gaussian_filter(newpattern[n], 
            self.filter_stdev, order=0, output=None,
            mode='wrap', cval=0.0, truncate=4.0)
        # restore variance removed by gaussian blur.
        newpattern =\
        newpattern*(self.filter_stdev*2.*np.sqrt(np.pi))
        # blend new pattern with old pattern.
        self.pattern = \
        np.sqrt(1.-self.lag1corr**2)*newpattern + \
        self.lag1corr*self.pattern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nsamples = 10; stdev = 2
    rp=RandomPattern(500.e3,3600.,20.e6,128,1800,nsamples=nsamples,stdev=stdev)
    # plot random sample.
    xens = rp.pattern
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(nsamples):
        plt.figure()
        plt.imshow(xens[n],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    print 'variance =', ((xens**2).sum(axis=0)/(nsamples-1)).mean()
    print '(expected ',stdev**2,')'
    nsamples = 1; stdev = 1
    rp = RandomPattern(1000.e3,3600.,20.e6,128,1800,nsamples=nsamples,stdev=stdev)
    ntimes = 1000
    x = rp.pattern[0]
    lag1cov = np.zeros(x.shape, x.dtype)
    lag1var = np.zeros(x.shape, x.dtype)
    spatial_cov = np.zeros(x.shape, x.dtype)
    for nt in range(ntimes):
        xold = x.copy()
        rp.evolve()
        x = rp.pattern[0]
        lag1cov = lag1cov + x*xold/(ntimes-1)
        lag1var = lag1var + x*x/(ntimes-1)
        spatial_cov = spatial_cov + x[rp.N/2,rp.N/2]*x/(ntimes-1)
    plt.figure()
    x = (rp.L/rp.N)*np.arange(rp.N)-rp.L/2
    plt.plot(x,0.5*(spatial_cov[:,rp.N/2]+spatial_cov[rp.N/2,:]),'r')
    plt.plot(x,np.exp(-(x/rp.hcorr)**2),'k')
    plt.axhline(0); plt.axvline(0)
    plt.show()
    lag1corr = lag1cov/lag1var
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',rp.lag1corr
    print 'variance = ',lag1var.mean(),' (expected ',stdev**2,')'
