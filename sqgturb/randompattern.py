import numpy as np
from scipy.ndimage import gaussian_filter

class RandomPattern:
    def __init__(self, spatial_corr_efold, temporal_corr_efold, L, N, dt, \
            nsamples=1, stdev=1.0):
        """
        define an ensemble of random patterns with specified temporal
        and spatial covariance structure by applying Gaussian blur to
        white noise.
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
        self.filter_stdev = self.hcorr*self.N/self.L
        for n in range(nsamples):
            self.pattern[n] = gaussian_filter(self.pattern[n], 
            self.filter_stdev, order=0, output=None,
            mode='wrap', cval=0.0, truncate=4.0)
        # restore variance removed by gaussian blur.
        self.pattern =\
        self.pattern*(self.filter_stdev*2.*np.sqrt(np.pi))

    def evolve(self):
        """
        evolve sample one time step
        """
        self.pattern = \
        np.sqrt(1.-self.lag1corr**2)* \
        self.stdev*np.random.normal(size=(self.nsamples,self.N,self.N)) + \
        self.lag1corr*self.pattern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nsamples = 100; stdev = 2
    rp=RandomPattern(200.e3,3600.,20.e6,128,1800,nsamples=nsamples,stdev=stdev)
    # plot random sample.
    xens = rp.pattern
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(5):
        plt.figure()
        plt.imshow(xens[n],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    print 'variance =', ((xens**2).sum(axis=0)/(nsamples-1)).mean()
    print '(expected ',stdev**2,')'
    plt.show()
    nsamples = 1; stdev = 2
    rp = RandomPattern(200.e3,3600.,20.e6,128,1800,nsamples=nsamples,stdev=stdev)
    ntimes = 100
    x = rp.pattern
    lag1cov = np.zeros(x.shape, x.dtype)
    lag1var = np.zeros(x.shape, x.dtype)
    for nt in range(ntimes):
        xold = x.copy()
        rp.evolve()
        x = rp.pattern
        lag1cov = lag1cov + x*xold/(ntimes-1)
        lag1var = lag1var + x*x/(ntimes-1)
    lag1corr = lag1cov/lag1var
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',rp.lag1corr
    print 'variance = ',lag1var.mean()
