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

def lgetkf(xens, hxens, obs, oberrs, covlocal, nerger=True, ngroups=None):

    """returns ensemble updated by LGETKF with cross-validation"""

    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    nanals = hxens.shape[0]
    ndim = covlocal.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime_b = xprime.copy()
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(hx, Rlocal, oberrvar, nerger=True):
        normfact = np.array(np.sqrt(hx.shape[0]-1),dtype=np.float32)
        if nerger:
            # Nerger regularization
            hpbht = (hx**2).sum(axis=0)/normfact**2
            hpbhtplusR = hpbht+oberrvar
            Rlocalfact = (Rlocal*oberrvar/hpbhtplusR)/(1.-Rlocal*hpbht/hpbhtplusR)
            Rinvsqrt = np.sqrt(Rlocalfact/oberrvar)
            YbRinv = hx*Rinvsqrt**2/normfact
            YbsqrtRinv = hx*Rinvsqrt/normfact
        else:
            YbsqrtRinv = hx*np.sqrt(Rlocal/oberrvar)/normfact
            YbRinv = hx*(Rlocal/oberrvar)/normfact
        return YbsqrtRinv, YbRinv

    def calcwts_mean(nens, hx, Rlocal, oberrvar, ominusf, nerger=True):
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(nens-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= nens:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(nens, hx_orig, hx, Rlocal, oberrvar,nerger=True):
        # hx_orig contains the ensemble for the witheld member
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(nens-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= nens:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    for n in range(ndim):
        mask = covlocal[:,n] > 1.0e-10
        nobs_local = mask.sum()
        if nobs_local > 0:
            Rlocal = covlocal[mask, n]
            oberrvar_local = oberrs[mask]
            ominusf_local = (obs-hxmean)[mask]
            hxprime_local = hxprime[:,mask]
            wts_ensmean = calcwts_mean(nanals, hxprime_local, Rlocal, oberrvar_local, ominusf_local, nerger=nerger)
            for k in range(2):
                xmean[k,n] += np.dot(wts_ensmean,xprime_b[:,k,n])
            # update sub-ensemble groups, using cross validation.
            for ngrp in range(ngroups):
                nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
                hxprime_cv = np.delete(hxprime_local,nanal_cv,axis=0); xprime_cv = np.delete(xprime_b[:,:,n],nanal_cv,axis=0)
                wts_ensperts_cv = calcwts_perts(nanals-nanals//ngroups, hxprime_local[nanal_cv], hxprime_cv, Rlocal, oberrvar_local, nerger=nerger)
                for k in range(2):
                    xprime[nanal_cv,k,n] += np.dot(wts_ensperts_cv,xprime_cv[:,k])
            xprime_mean = xprime[:,:,n].mean(axis=0) 
            xprime[:,:,n] -= xprime_mean # ensure zero mean
            xens[:,:,n] = xmean[:,n]+xprime[:,:,n]

    return xens

def lgetkf_vloc(xens, xens2, hxens, hxens2, obs, oberrs, covlocal, nanal_index, nerger=True, ngroups=None):

    """returns ensemble updated by LGETKF with cross-validation"""

    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    hxprime2 = hxens2 - hxmean # modulated ens
    nanals = hxens.shape[0]
    nanals2 = hxens2.shape[0] # modulated ens size
    ndim = covlocal.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    xprime2 = xens2 - xmean # modulated ens
    if ngroups is None: # default is "leave one out" (nanals must be multiple of ngroups)
        ngroups = nanals
    if nanals % ngroups:
        raise ValueError('nanals must be a multiple of ngroups')
    else:
        nanals_per_group = nanals//ngroups

    def getYbvecs(hx, Rlocal, oberrvar, nerger=True):
        normfact = np.array(np.sqrt(hx.shape[0]-1),dtype=np.float32)
        if nerger:
            # Nerger regularization
            hpbht = (hx**2).sum(axis=0)/normfact**2
            hpbhtplusR = hpbht+oberrvar
            Rlocalfact = (Rlocal*oberrvar/hpbhtplusR)/(1.-Rlocal*hpbht/hpbhtplusR)
            Rinvsqrt = np.sqrt(Rlocalfact/oberrvar)
            YbRinv = hx*Rinvsqrt**2/normfact
            YbsqrtRinv = hx*Rinvsqrt/normfact
        else:
            YbsqrtRinv = hx*np.sqrt(Rlocal/oberrvar)/normfact
            YbRinv = hx*(Rlocal/oberrvar)/normfact
        return YbsqrtRinv, YbRinv

    def calcwts_mean(nens, hx, Rlocal, oberrvar, ominusf,nerger=True):
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(nens-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= nens:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for mean update).
        # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        # in Bishop paper (eqs 10-12).
        # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
        # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
        pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
        return np.dot(pa, np.dot(YbRinv,ominusf))/normfact

    def calcwts_perts(nens, hx_orig, hx, Rlocal, oberrvar,nerger=True):
        # hx_orig contains the ensemble for the witheld member
        # nens is the original (unmodulated) ens size
        nobs = hx.shape[1]
        normfact = np.array(np.sqrt(nens-1),dtype=np.float32)
        # gain-form etkf solution
        # HZ^T = hxens * R**-1/2
        # compute eigenvectors/eigenvalues of A = HZ^T HZ (C=left SV)
        # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
        # normalize so dot product is covariance
        YbsqrtRinv, YbRinv = getYbvecs(hx,Rlocal,oberrvar,nerger=nerger)
        if nobs >= nens:
            a = np.dot(YbsqrtRinv,YbsqrtRinv.T)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
        else:
            a = np.dot(YbsqrtRinv.T,YbsqrtRinv)
            evals, evecs, info = lapack.dsyevd(a)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            evecs = np.dot(YbsqrtRinv,evecs/np.sqrt(evals))
        # gammapI used in calculation of posterior cov in ensemble space
        gamma_inv = 1./evals; gammapI = evals+1.
        # compute factor to multiply with model space ensemble perturbations
        # to compute analysis increment (for perturbation update), save in single precision.
        # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        # in Bishop paper (eqn 29).
        # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
        pasqrt=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
        return -np.dot(pasqrt, np.dot(YbRinv,hx_orig.T)).T/normfact # use witheld ens member here

    for n in range(ndim):
        mask = covlocal[:,n] > 1.0e-10
        nobs_local = mask.sum()
        if nobs_local > 0:
            Rlocal = covlocal[mask, n]
            oberrvar_local = oberrs[mask]
            ominusf_local = (obs-hxmean)[mask]
            hxprime2_local = hxprime2[:,mask]
            hxprime_local = hxprime[:,mask]
            wts_ensmean = calcwts_mean(nanals, hxprime2_local, Rlocal, oberrvar_local, ominusf_local, nerger=nerger)
            for k in range(2):
                xmean[k,n] += np.dot(wts_ensmean,xprime2[:,k,n])
            # update sub-ensemble groups, using cross validation.
            for ngrp in range(ngroups):
                # nanal_index has original ens index for modulated member
                nanal_cv = [na + ngrp*nanals_per_group for na in range(nanals_per_group)]
                nanals_sub = np.nonzero(np.isin(nanal_index,nanal_cv))
                hxprime_cv = np.delete(hxprime2_local,nanals_sub,axis=0); xprime_cv = np.delete(xprime2[:,:,n],nanals_sub,axis=0)
                wts_ensperts_cv = calcwts_perts(nanals-nanals//ngroups, hxprime_local[nanal_cv], hxprime_cv, Rlocal, oberrvar_local, nerger=nerger)
                for k in range(2):
                    xprime[nanal_cv,k,n] += np.dot(wts_ensperts_cv,xprime_cv[:,k])
            xprime_mean = xprime[:,:,n].mean(axis=0) 
            xprime[:,:,n] -= xprime_mean # ensure zero mean
            xens[:,:,n] = xmean[:,n]+xprime[:,:,n]

    return xens
