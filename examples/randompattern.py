import numpy as np
from scipy.linalg import eigh

def _gaussian(rr,corrl):
    # gaussian covariance model.
    r = rr/corrl
    return np.exp(-r**2)

def _cartdist(x1,y1,x2,y2,xmax,ymax):
    # cartesian distance on doubly periodic plane
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5*xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5*ymax, ymax - dy, dy)
    return np.sqrt(dx**2 + dy**2)

class RandomPattern:
    def __init__(self, spatial_corr_efold, temporal_corr_efold, L, N,\
                 dt, nsamples=1, stdev=1.0, thresh = 0.99):
        self.hcorr = spatial_corr_efold
        self.tcorr = temporal_corr_efold
        self.dt = dt
        self.lag1corr = np.exp(-1)**(self.dt/self.tcorr)
        self.L = L
        self.stdev = stdev
        self.nsamples = nsamples
        self.N = N
        self.thresh = thresh
        # construct covariance matrix.
        x1 = np.arange(0,self.L,self.L/self.N)
        y1 = np.arange(0,self.L,self.L/self.N)
        x, y = np.meshgrid(x1, y1)
        x2 = x.flatten(); y2 = y.flatten()
        self.cov = np.zeros((N**2,N**2),np.float64)
        n = 0
        for x0,y0 in zip(x2,y2):
            r = _cartdist(x0,y0,x2,y2,self.L,self.L)
            self.cov[n,:] = _gaussian(r,self.hcorr)
            n = n + 1
        # eigenanalysis
        evals, evecs = eigh(self.cov)
        evals = np.where(evals > 1.e-10, evals, 1.e-10)
        if self.thresh == 1.0:
            self.scaledevecs = evecs*np.sqrt(evals)
            self.nevecs = self.N**2
        else:
            evalsum = evals.sum(); neig = 0; frac = 0.
            while frac < self.thresh:
                frac = evals[self.N**2-neig-1:self.N**2].sum()/evalsum
                neig += 1
            self.scaledevecs = (evecs*np.sqrt(evals/frac))[:,self.N**2-neig:self.N**2]
            print '%s eigenvectors explain %s percent of variance' %\
            (neig,100*self.thresh)
            self.nevecs = neig
        # initialize random coefficients.
        self.coeffs = np.random.normal(size=(self.nsamples,self.nevecs))

    def random_sample(self):
        xens = np.zeros((nsamples,self.N*self.N),np.float32)
        for n in range(nsamples):
            for j in range(self.nevecs):
                xens[n] = xens[n]+self.stdev*self.coeffs[n,j]*self.scaledevecs[:,j]
        return np.squeeze(xens.reshape((nsamples, self.N, self.N)))

    def evolve(self):
        self.coeffs = \
        np.sqrt(1.-self.lag1corr**2)* \
        np.random.normal(size=(self.nsamples,self.nevecs)) + \
        self.lag1corr*coeffs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cPickle
    nsamples = 10; stdev = 2
    rp1 = RandomPattern(1000.e3,3600.,20.e6,64,1800,nsamples=nsamples,stdev=stdev)
    f = open('saved_rp.pickle','wb')
    cPickle.dump(rp1, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    f = open('saved_rp.pickle','rb')
    rp = cPickle.load(f)
    f.close()
    # plot random sample.
    xens = rp.random_sample()
    minmax = max(np.abs(xens.min()), np.abs(xens.max()))
    for n in range(nsamples):
        plt.figure()
        plt.imshow(xens[n],plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
        plt.title('pattern %s' % n)
        plt.colorbar()
    print 'variance =', ((xens**2).sum(axis=0)/(nsamples-1)).mean()
    print '(expected ',stdev**2,')'
    plt.show()
