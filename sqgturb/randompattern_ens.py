import numpy as np
from scipy.ndimage import gaussian_filter

class RandomPatternEns:
    def __init__(self, spatial_corr_efold, temporal_corr_efold, L, N, dt, nens, \
                 stdev=1.0, order=0):
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
        nens: number of ensemble members
        then pattern is duplicated..  If set to 2, independent
        patterns are generated for each boundary.
        stdev:  spatial standard deviation (amplitude).
        """
        self.hcorr = spatial_corr_efold
        self.tcorr = temporal_corr_efold
        self.dt = dt
        self.L = L
        self.stdev = stdev
        self.N = N
        self.order = order
        self.nens = nens
        # initialize patterns.
        # generate white noise.
        self.pattern = self.stdev*np.random.normal(\
                       size=(nens,2,self.N,self.N))
        if self.hcorr > 0:
            # apply gaussian filter
            self.filter_stdev = self.hcorr*self.N/(self.L*np.sqrt(4.))
            for ne in range(nens):
                for n in range(2):
                    self.pattern[ne,n,:,:] = gaussian_filter(self.pattern[ne,n,:,:],
                    self.filter_stdev, order=order, output=None,
                    mode='wrap', cval=0.0, truncate=6.0)
                # restore variance removed by gaussian blur.
                if order == 0:
                    self.pattern[ne,n,:,:] =\
                    self.pattern[ne,n,:,:]*(self.filter_stdev*2.*np.sqrt(np.pi))
                else:
                    for n in range(2):
                        stdev_computed = np.sqrt((self.pattern[ne,n,:,:]**2).mean())
                        self.pattern[ne,n,:,:] = self.pattern[ne,n,:,:]*stdev/stdev_computed
        self.pattern = self.pattern - self.pattern.mean(axis=0) # ensure zero mean          

    def evolve(self,dt=None):
        """
        evolve random patterns one time step
        """
        if dt is None: dt = self.dt
        lag1corr = np.exp(-1)**(dt/self.tcorr)
        # generate white noise.
        newpattern = self.stdev*np.random.normal(\
                     size=(self.nens,2,self.N,self.N))
        if self.hcorr > 0:
            for ne in range(self.nens):
                # apply gaussian filter
                for n in range(2):
                    newpattern[ne,n,:,:] = gaussian_filter(newpattern[ne,n,:,:],
                    self.filter_stdev, order=self.order, output=None,
                    mode='wrap', cval=0.0, truncate=6.0)
                # restore variance removed by gaussian blur.
                if self.order == 0:
                    newpattern[ne,n,:,:] =\
                    newpattern[ne,n,:,:]*(self.filter_stdev*2.*np.sqrt(np.pi))
                else:
                    for n in range(2):
                        stdev_computed = np.sqrt((self.pattern[ne,n,:,:]**2).mean())
                        self.pattern[ne,n,:,:]  = self.pattern[ne,n,:,:]*self.stdev/stdev_computed
                # blend new pattern with old pattern.
                self.pattern[ne] = np.sqrt(1.-lag1corr**2)*newpattern[ne] + lag1corr*self.pattern[ne]
        self.pattern = self.pattern - self.pattern.mean(axis=0) # ensure zero mean          

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nens = 10; stdev = 2
    rp=RandomPatternEns(500.e3,3600.,20.e6,128,1800,nens,stdev=stdev)
    for nt in range(100):
        rp.evolve()
    # plot random sample.
    xens = rp.pattern
    xensmean = xens.mean(axis=0)
    print xensmean.min(), xensmean.max(), xensmean.mean()
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(nens):
        plt.figure()
        plt.imshow(xens[n,1,...],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    nens = 10; stdev = 1
    rp = RandomPatternEns(1000.e3,3600.,20.e6,128,1800,nens,stdev=stdev)
    ntimes = 1000
    x = rp.pattern[0,1]
    lag1cov = np.zeros(x.shape, x.dtype)
    lag1var = np.zeros(x.shape, x.dtype)
    spatial_cov = np.zeros(x.shape, x.dtype)
    for nt in range(ntimes):
        xold = x.copy()
        rp.evolve()
        x = rp.pattern[0,1]
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
    lag1corr_exp = np.exp(-1)**(rp.dt/rp.tcorr)
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',lag1corr_exp
    print 'variance = ',lag1var.mean(),' (expected ',stdev**2,')'
