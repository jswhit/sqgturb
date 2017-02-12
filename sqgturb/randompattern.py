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
        evolve sample one time step
        """
        newpattern = self.stdev*np.random.normal(\
                     size=(self.nsamples,self.N,self.N))
        for n in range(nsamples):
            newpattern[n] = gaussian_filter(newpattern[n], 
            self.filter_stdev, order=0, output=None,
            mode='wrap', cval=0.0, truncate=4.0)
        newpattern =\
        newpattern*(self.filter_stdev*2.*np.sqrt(np.pi))
        self.pattern = \
        np.sqrt(1.-self.lag1corr**2)*newpattern + \
        self.lag1corr*self.pattern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nsamples = 100; stdev = 2
    rp=RandomPattern(1000.e3,3600.,20.e6,128,1800,nsamples=nsamples,stdev=stdev)
    # plot random sample.
    xens = rp.pattern
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(1):
        plt.figure()
        plt.imshow(xens[n],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    #plt.show()
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
        #plt.figure()
        #plt.imshow(x,plt.cm.bwr,interpolation='nearest',origin='lower')
        #plt.show()
        #raise SystemExit
        lag1cov = lag1cov + x*xold/(ntimes-1)
        lag1var = lag1var + x*x/(ntimes-1)
        spatial_cov = spatial_cov + x[rp.N/2,rp.N/2]*x/(ntimes-1)
    plt.figure()
    x = (rp.L/rp.N)*np.arange(rp.N)-rp.L/2
    print x.shape, spatial_cov.shape
    plt.plot(x,spatial_cov[:,rp.N/2],'r')
    plt.plot(x,np.exp(-(x/rp.hcorr)**2),'k')
    plt.axhline(0); plt.axvline(0)
    #plt.imshow(spatial_cov,plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-1.,vmax=1.)
    plt.show()
    lag1corr = lag1cov/lag1var
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',rp.lag1corr
    print 'variance = ',lag1var.mean()
