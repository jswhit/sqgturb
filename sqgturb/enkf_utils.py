import numpy as np
from scipy.linalg import lapack, inv

# function definitions.


def cartdist(x1, y1, x2, y2, xmax, ymax):
    """cartesian distance on doubly periodic plane"""
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5 * xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5 * ymax, ymax - dy, dy)
    return np.sqrt(dx ** 2 + dy ** 2)


def gaspcohn(r):
    """
    Gaspari-Cohn taper function.
    very close to exp(-(r/c)**2), where c = sqrt(0.15)
    r should be >0 and normalized so taper = 0 at r = 1
    """
    rr = 2.0 * r
    rr += 1.0e-13  # avoid divide by zero warnings from numpy
    taper = np.where(
        r <= 0.5,
        (((-0.25 * rr + 0.5) * rr + 0.625) * rr - 5.0 / 3.0) * rr ** 2 + 1.0,
        np.zeros(r.shape, r.dtype),
    )
    taper = np.where(
        np.logical_and(r > 0.5, r < 1.0),
        ((((rr / 12.0 - 0.5) * rr + 0.625) * rr + 5.0 / 3.0) * rr - 5.0) * rr
        + 4.0
        - 2.0 / (3.0 * rr),
        taper,
    )
    return taper

def lgetkf(xens, hxens, obs, oberrs, covlocal):

    """returns ensemble updated by LGETKF with cross-validation"""

    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    nanals = hxens.shape[0]
    ndim = covlocal.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime_b = xprime.copy()

    def calcwts_mean(nens, hx, Rinv, ominusf):
        normfact = np.array(np.sqrt(nens-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv = hx*np.sqrt(Rinv)/normfact
        YbRinv = hx*Rinv/normfact
        a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
        evals, evecs, info = lapack.dsyevd(a)
        evals = evals.clip(min=np.finfo(evals.dtype).eps)
        gamma_inv = 1./evals
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(hx_orig, hx, Rinv):
        normfact = np.array(np.sqrt(hx_orig.shape[0]-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv = hx*np.sqrt(Rinv)/normfact
        YbRinv = hx*Rinv/normfact
        a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
        evals, evecs, info = lapack.dsyevd(a)
        evals = evals.clip(min=np.finfo(evals.dtype).eps)
        gamma_inv = 1./evals
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        pa=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        return -np.dot(pa, np.dot(YbRinv,hx_orig.T)).T/normfact # use orig ens here

    for n in range(ndim):
        mask = covlocal[:,n] > 1.0e-10
        Rinv = covlocal[mask, n] / oberrs[mask]
        ominusf = (obs-hxmean)[mask]
        wts_ensmean = calcwts_mean(nanals, hxprime[:, mask], Rinv, ominusf)
        for k in range(2):
            xmean[k,n] += np.dot(wts_ensmean,xprime_b[:,k,n])
        # update one member at a time, using cross validation.
        for nanal_cv in range(nanals):
            hxprime_cv = np.delete(hxprime,nanal_cv,axis=0); xprime_cv = np.delete(xprime_b,nanal_cv,axis=0)
            wts_ensperts_cv = calcwts_perts(hxprime[nanal_cv,mask], hxprime_cv[:,mask], Rinv)
            for k in range(2):
                xprime[nanal_cv,k,n] += np.dot(wts_ensperts_cv,xprime_cv[:,k,n])
        xens[:,:,n] = xmean[:,n]+xprime[:,:,n]

    return xens
