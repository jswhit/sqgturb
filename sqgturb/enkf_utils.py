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


def enkf_update(
    xens, hxens, obs, oberrs, covlocal, obcovlocal=None, gainform=False
):
    """serial potter method or LETKF (if obcovlocal is None)"""

    nanals, nlevs, ndim = xens.shape
    nobs = obs.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    fact = np.array([1.0, 1.0], np.float32)

    if obcovlocal is not None:  # serial EnSRF update

        for nob, ob, oberr in zip(np.arange(nobs), obs, oberrs):
            ominusf = ob - hxmean[nob].copy()
            hxens = hxprime[:, nob].copy()
            hpbht = (hxens ** 2).sum() / (nanals - 1)
            gainfact = (
                (hpbht + oberr)
                / hpbht
                * (1.0 - np.sqrt(oberr / (hpbht + oberr)))
            )
            # state space update
            # only update points closer than localization radius to ob
            mask = covlocal[nob, :] > 1.0e-10
            for k in range(2):
                pbht = (xprime[:, k, mask].T * hxens).sum(axis=1) / float(
                    nanals - 1
                )
                kfgain = covlocal[nob, mask] * pbht / (hpbht + oberr)
                xmean[k, mask] += kfgain * ominusf
                xprime[:, k, mask] -= gainfact * kfgain * hxens[:,np.newaxis]
            # observation space update
            # only update obs within localization radius
            mask = obcovlocal[nob, :] > 1.0e-10
            pbht = (hxprime[:, mask].T * hxens).sum(axis=1) / float(
                nanals - 1
            )
            kfgain = obcovlocal[nob, mask] * pbht / (hpbht + oberr)
            hxmean[mask] += kfgain * ominusf
            hxprime[:, mask] -= gainfact * kfgain * hxens[:,np.newaxis]

        return xmean + xprime

    else:  # LGETKF update

        def calcwts(hx, Rinv, ominusf, gainform=gainform):

            nanals = hx.shape[0]
            if not gainform:
                # 'regular' etkf solution
                YbRinv = hx*Rinv
                pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
                evals, eigs, info = lapack.dsyevd(pa)
                evals = evals.clip(min=np.finfo(evals.dtype).eps)
                painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
                tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
                return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis], None
            else:
                # gain-form etkf solution
                # HZ^T = hxens * R**-1/2
                # compute eigenvectors/eigenvalues of HZ^T HZ (C=left SV)
                # (in Bishop paper HZ is nobs, nanals, here is it nanals, nobs)
                # normalize so dot product is covariance
                normfact = np.array(np.sqrt(nanals-1),dtype=np.float32)
                YbsqrtRinv = hx*np.sqrt(Rinv)/normfact  # use modulated ens here
                YbRinv = hx*Rinv/normfact               # use modulated ens here
                pa = np.dot(YbsqrtRinv,YbsqrtRinv.T)
                evals, evecs, info = lapack.dsyevd(pa)
                gamma_inv = np.zeros_like(evals)
                for n in range(evals.shape[0]):
                    if evals[n] > np.finfo(evals.dtype).eps:
                        gamma_inv[n] = 1./evals[n]
                    else:
                        evals[n] = 0.
                # gammapI used in calculation of posterior cov in ensemble space
                gammapI = evals+1.
                # create HZ^T R**-1/2
                # compute factor to multiply with model space ensemble perturbations
                # to compute analysis increment (for mean update).
                # This is the factor C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
                # in Bishop paper (eqs 10-12).
                # pa = C (Gamma + I)**-1 C^T (analysis error cov in ensemble space)
                pa = np.dot(evecs/gammapI[np.newaxis,:],evecs.T)
                # wts_ensmean = C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 (y - HXmean)
                wts_ensmean = np.dot(pa, np.dot(YbRinv,ominusf))/normfact
                # compute factor to multiply with model space ensemble perturbations
                # to compute analysis increment (for perturbation update), save in single precision.
                # This is -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
                # in Bishop paper (eqn 29).
                # For DEnKF factor is -0.5*C (Gamma + I)**-1 C^T (HZ)^ T R**-1/2 HXprime
                # = -0.5 Pa (HZ)^ T R**-1/2 HXprime (Pa already computed)
                # pa = C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T
                # gammapI = sqrt(1.0/gammapI)
                # ( pa=0.5*pa for denkf)
                pa=np.dot(evecs*(1.-np.sqrt(1./gammapI[np.newaxis,:]))*gamma_inv[np.newaxis,:],evecs.T)
                # wts_ensperts = -C [ (I - (Gamma+I)**-1/2)*Gamma**-1 ] C^T (HZ)^T R**-1/2 HXprime
                # if denkf, wts_ensperts = -0.5 C (Gamma + I)**-1 C^T (HZ)^T R**-1/2 HXprime
                wts_ensperts = -np.dot(pa, np.dot(YbRinv,hx.T)).T/normfact # use orig ens here
                return wts_ensmean, wts_ensperts

        for n in range(covlocal.shape[-1]):
            mask = covlocal[:,n] > 1.0e-10
            Rinv = covlocal[mask, n] / oberrs[mask]
            ominusf = (obs-hxmean)[mask]
            if not gainform:
                wts, junk = calcwts(hxprime[:, mask], Rinv, ominusf, gainform=gainform)
                for k in range(2):
                    xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime[:, k, n])
            else:
                wts_ensmean,wts_ensperts = calcwts(hxprime[:, mask], Rinv, ominusf, gainform=gainform)
                # increments constructed from weighted modulated ensemble member prior perts.
                for k in range(2):
                    xmean[k,n] += np.dot(wts_ensmean,xprime[:,k,n]) # use mod ens for xprime
                    # use orig ens on lhs, mod ens on rhs
                    xprime[:,k,n] += np.dot(wts_ensperts,xprime[:,k,n]) 
                xens[:,:,n] = xmean[:,n]+xprime[:,:,n]
 
        return xens
