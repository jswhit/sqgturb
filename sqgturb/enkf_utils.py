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
    xens, hxens, obs, oberrs, covlocal, obcovlocal=None
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

    else:  # LETKF update

        def calcwts(hx, Rinv, ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
            evals, eigs, info = lapack.dsyevd(pa)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
            tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
            return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]

        for n in range(covlocal.shape[-1]):
            mask = covlocal[:,n] > 1.0e-10
            Rinv = np.diag(covlocal[mask, n] / oberrs[mask])
            ominusf = (obs-hxmean)[mask]
            wts = calcwts(hxprime[:, mask], Rinv, ominusf)
            for k in range(2):
                xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime[:, k, n])

        return xens
