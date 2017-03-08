import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma,kv

def _gaussian(rr,corrl):
    # gaussian covariance model.
    r = rr/corrl
    return np.exp(-r**2)

def _matern(rr,corrl,kappa=5./2.):
    # matern covariance model
    r = rr/(corrl/np.sqrt(2.))
    r = np.where(r < 1.e-10, 1.e-10, r)
    r1 = 2 ** (kappa-1.0) * gamma(kappa)
    bes = kv(kappa,r)
    return (1.0/r1) * r ** kappa * bes

def _cartdist(x1,y1,x2,y2,xmax,ymax):
    # cartesian distance on doubly periodic plane
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5*xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5*ymax, ymax - dy, dy)
    return np.sqrt(dx**2 + dy**2)

class RandomPattern:
    def __init__(self, spatial_corr_efold, temporal_corr_efold, L, N, dt, \
            nsamples=1, stdev=1.0, seed=None, truncate=4,
            calcweights=_gaussian):
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
        self.hcorr = spatial_corr_efold
        self.tcorr = temporal_corr_efold
        self.stdev = stdev
        self.dt = float(dt)
        self.L = float(L)
        self.nsamples = nsamples
        self.N = N
        # initialize random coefficients.
        if seed is None:
            self.rs = np.random.RandomState()
        else:
            self.rs = np.random.RandomState(seed)
        # generate weights.
        dx = L/N
        nwindow = int(truncate*self.hcorr/dx)
        npts = 2*nwindow+1
        self.weights = np.zeros((npts,npts),np.float)
        for j in range(-nwindow,nwindow+1):
            for i in range(-nwindow,nwindow+1):
                self.weights[i+nwindow,j+nwindow] =\
                calcweights(np.sqrt(i**2+j**2),self.hcorr/dx)
        self.weights = self.weights/self.weights.sum()
        # initialize random pattern.
        self.pattern = self.genpattern()

    def genpattern(self,seed=None):
        # initialize patterns.
        # generate white noise.
        pattern = self.stdev*self.rs.normal(\
                     size=(2,self.N,self.N))
        if self.nsamples == 2:
            pass
        elif self.nsamples == 1:
            pattern[1] = pattern[0]
        else:
            raise ValueError('nsamples must be 1 or 2')
        # apply filter
        if self.nsamples == 2:
            for n in range(self.nsamples):
                pattern[n] = convolve(pattern[n],
                self.weights,output=None,
                mode='wrap', cval=0.0)
        else:
            pattern[0] = convolve(pattern[0],
            self.weights,output=None,
            mode='wrap', cval=0.0)
            pattern[1]=pattern[0]
        # restore variance removed by filter
        for n in range(2):
            stdev_computed = np.sqrt((pattern[n,:,:]**2).mean())
            pattern[n,:,:] = pattern[n,:,:]*self.stdev/stdev_computed
        return pattern

    def copy(self,seed):
        import copy
        newself = copy.copy(self)
        newself.rs = np.random.RandomState(seed)
        newself.pattern = self.genpattern()
        return newself

    def evolve(self,dt=None):
        """
        evolve random patterns one time step
        """
        if dt is None: dt = self.dt
        pattern = self.genpattern()
        # blend new pattern with old pattern.
        lag1corr = np.exp(-1.0)**(dt/self.tcorr)
        self.pattern = np.sqrt(1.-lag1corr**2)*pattern + lag1corr*self.pattern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nsamples = 2
    N = 128; L = 20.e6; stdev = 2.
    rp0=RandomPattern(0.5*L/N,3600.,L,N,1200,nsamples=nsamples,stdev=stdev,calcweights=_matern)
    rp = rp0.copy(seed=42)
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
    rp = RandomPattern(1000.e3,1200.,20.e6,128,1200,nsamples=nsamples,stdev=stdev,calcweights=_gaussian)
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
