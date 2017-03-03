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
        nsamples:  number of ensemble members. Can be 1 or 2.  If set to 1,
        then pattern is duplicated..  If set to 2, independent
        patterns are generated for each boundary.
        stdev:  spatial standard deviation (amplitude).
        """
        self.hcorr = np.array(spatial_corr_efold,np.float)
        if self.hcorr.shape == ():
            self.hcorr.shape = (1,)
        self.tcorr = np.array(temporal_corr_efold,np.float)
        if self.tcorr.shape == ():
            self.tcorr.shape = (1,)
        self.stdev = np.array(stdev,np.float)
        if self.stdev.shape == ():
            self.stdev.shape = (1,)
        self.npatterns = len(self.stdev)
        self.filter_stdev = np.zeros(self.npatterns, np.float)
        self.dt = float(dt)
        self.L = float(L)
        self.nsamples = nsamples
        self.N = N
        self.filter_stdev = self.hcorr*self.N/(self.L*np.sqrt(4.))
        self.pattern = self.genpattern()

    def genpattern(self):
        # initialize patterns.
        # generate white noise.
        pattern = np.zeros((2,self.N,self.N),np.float)
        for npattern in range(self.npatterns):
            newpattern = self.stdev[npattern]*np.random.normal(\
                         size=(2,self.N,self.N))
            if self.nsamples == 2:
                pass
            elif self.nsamples == 1:
                newpattern[1] = newpattern[0]
            else:
                raise ValueError('nsamples must be 1 or 2')
            if self.hcorr[npattern] > 0:
                # apply gaussian filter
                if self.nsamples == 2:
                    for n in range(self.nsamples):
                        newpattern[n] = gaussian_filter(newpattern[n],
                        self.filter_stdev[npattern],output=None,
                        order=0,mode='wrap', cval=0.0, truncate=6.0)
                else:
                    newpattern[0] = gaussian_filter(newpattern[0],
                    self.filter_stdev[npattern],output=None,
                    order=0,mode='wrap', cval=0.0, truncate=6.0)
                    newpattern[1]=newpattern[0]
                # restore variance removed by gaussian blur.
                newpattern = newpattern*(self.filter_stdev[npattern]*2.*np.sqrt(np.pi))
            pattern += newpattern
        return pattern

    def evolve(self,dt=None):
        """
        evolve random patterns one time step
        """
        if dt is None: dt = self.dt
        newpattern = self.genpattern()
        # blend new pattern with old pattern.
        for npattern in range(self.npatterns):
            lag1corr = np.exp(-1.0)**(dt/self.tcorr[npattern])
            self.pattern[npattern] = np.sqrt(1.-lag1corr**2)*newpattern[npattern] + lag1corr*self.pattern[npattern]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nsamples = 2; stdev = [1.,2]
    rp=RandomPattern([100.e3,500.e3],[600.,3600.],20.e6,128,1200,nsamples=nsamples,stdev=stdev)
    rp.evolve()
    # plot random sample.
    xens = rp.pattern
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(2):
        plt.figure()
        plt.imshow(xens[n],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    nsamples = 1; stdev = 1
    rp = RandomPattern(1000.e3,1200.,20.e6,128,1200,nsamples=nsamples,stdev=stdev)
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
    lag1corr_exp = np.exp(-1)**(rp.dt/rp.tcorr)
    print 'lag 1 autocorr = ',lag1corr.mean(), ', expected ',lag1corr_exp
    print 'variance = ',lag1var.mean(),' (expected ',stdev**2,')'
