import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def gaussian(rr,corrl):
    # gaussian covariance model.
    r = rr/corrl
    return np.exp(-r**2)

def cartdist(x1,y1,x2,y2,xmax,ymax):
    # cartesian distance on doubly periodic plane
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5*xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5*ymax, ymax - dy, dy)
    return np.sqrt(dx**2 + dy**2)

# construct covariance matrix.
corrl = 0.05
xmax = 1.; ymax = xmax
nmax = 64
x1 = np.arange(0,xmax,xmax/nmax)
y1 = np.arange(0,ymax,ymax/nmax)
x, y = np.meshgrid(x1, y1)
x2 = x.flatten(); y2 = y.flatten()
cov = np.zeros((nmax**2,nmax**2),np.float64)
n = 0
for x0,y0 in zip(x2,y2):
    r = cartdist(x0,y0,x2,y2,xmax,ymax)
    cov[n,:] = gaussian(r,corrl)
    n = n + 1

# plot covariance matrix.
plt.figure()
plt.imshow(cov,plt.cm.hot_r,interpolation='nearest',origin='lower')
plt.title('covariance matrix')

# eigenanalysis
evals, evecs = eigh(cov)
evals = np.where(evals > 1.e-10, evals, 1.e-10)
scaledevecs = evecs*np.sqrt(evals)

# construct random sample.
nsamples = 10
xens = np.zeros((nsamples,nmax*nmax),np.float64)
for n in range(nsamples):
    coeffs = np.random.normal(size=len(evals))
    for j in range(len(coeffs)):
        xens[n] = xens[n]+coeffs[j]*scaledevecs[:,j]

# plot random sample.
minmax = max(np.abs(xens.min()), np.abs(xens.max()))
for n in range(nsamples):
    x = xens[n].reshape((nmax,nmax))
    plt.figure()
    plt.imshow(x,plt.cm.bwr,interpolation='nearest',origin='lower',vmin=-minmax,vmax=minmax)
    plt.title('pattern %s' % n)
    plt.colorbar()

plt.show()
