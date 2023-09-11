import numpy as np
from scipy.linalg import inv, lapack

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

def letkf_multiscale_update(xprime,xmean,hxprime,hxmean,obs,oberrs,covlocal,vcovlocal_facts):

    nlscales, nanals, nlevs, ndim = xprime.shape
    nobs = obs.shape[-1]
    fact = np.array([1.0, 1.0], np.float64)

    ndim1 = covlocal.shape[-1]
    hx = np.empty((nlscales, nanals, 2 * nobs), np.float64)
    omf = np.empty(2 * nobs, np.float64)
    oberrvar = np.empty(2 * nobs, np.float64)
    covlocal_tmp = np.empty((nlscales, 2 * nobs, 2, ndim1), np.float64)
    for kob in range(2):
        oberrvar[kob * nobs : (kob + 1) * nobs] = oberrs[:]
        omf[kob * nobs : (kob + 1) * nobs] = obs[kob, :] - hxmean[kob, :]
        for nlscale in range(nlscales):
            fact[:] = 1.0
            fact[1 - kob] = vcovlocal_facts[nlscale]
            hx[nlscale, :, kob * nobs : (kob + 1) * nobs] = hxprime[nlscale, :, kob, :]
            for k in range(2):
                covlocal_tmp[nlscale, kob * nobs : (kob + 1) * nobs, k, :] = (
                    fact[k] * covlocal[nlscale, :, :]
            )

    def letkf_update(hx, Rinv, x, xm, ominusf):

        for n in range(nlscales):
            Yb_Rinv = np.dot(hx[n], Rinv[n])
            Yb_sqrtRinv = np.dot(hx[n], np.sqrt(Rinv[n]))
            pa = (nanals-1)*np.eye(nanals) +\
                  np.dot(Yb_sqrtRinv, Yb_sqrtRinv.T)
            # Eigenanalysis
            #evals, eigs, info = lapack.dsyevd(pa)
            #evals = evals.clip(min=np.finfo(evals.dtype).eps)
            #painv = np.dot(np.dot(eigs, np.diag(1.0 / evals)), eigs.T)
            #pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
            # Cholesky decomp
            pasqrt, info = lapack.dpotrf(pa,overwrite_a=0)
            painv, info = lapack.dpotri(pasqrt)
            pasqrt = np.triu(pasqrt)
            painv += np.triu(painv, k=1).T
            pasqrtinv = inv(pasqrt)
            kfgain = np.dot(x[n], np.dot(painv, Yb_Rinv))
            x[n] = np.sqrt(nanals-1)*np.dot(pasqrtinv.T, x[n])
            xm += np.dot(kfgain, ominusf)

        #Yb_Rinv_lst=[]
        #Yb_sqrtRinv_lst=[]
        #for n in range(nlscales):
        #    Yb_Rinv_lst.append(np.dot(hx[n], Rinv[n]))
        #    Yb_sqrtRinv_lst.append(np.dot(hx[n], np.sqrt(Rinv[n])))
        #Yb_sqrtRinv = np.vstack(Yb_sqrtRinv_lst)
        #Yb_Rinv = np.vstack(Yb_Rinv_lst)
        #pa = (nanals-1)*np.eye(nanals*nlscales) +\
        #     np.dot(Yb_sqrtRinv, Yb_sqrtRinv.T)

        ## Using eigenanalysis
        ##evals, eigs, info = lapack.dsyevd(pa)
        ##evals = evals.clip(min=np.finfo(evals.dtype).eps)
        ##painv = np.dot(np.dot(eigs, np.diag(1.0 / evals)), eigs.T)
        ##pasqrtinv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)

        ## Using cholesky decomp
        #pasqrt, info = lapack.dpotrf(pa,overwrite_a=0)
        #painv, info = lapack.dpotri(pasqrt)
        #pasqrt = np.triu(pasqrt)
        #painv += np.triu(painv, k=1).T
        #pasqrtinv = inv(pasqrt)

        #xtmp = np.concatenate(x)
        #kfgain = np.dot(xtmp, np.dot(painv, Yb_Rinv))
        #x = np.sqrt(nanals-1)*np.dot(pasqrtinv.T, xtmp)
        #xm += np.dot(kfgain, ominusf)

        return x.reshape(nlscales,nanals), xm

    covlocal_tmp = covlocal_tmp.clip(min=np.finfo(covlocal_tmp.dtype).eps)
    for n in range(ndim1):
        for k in range(2):
            Rinv_lst = []
            # use largest localization scale to define local volume
            mask = covlocal_tmp[0, :, k, n] > 1.0e-10
            for nscale in range(nlscales):
                Rinv_lst.append(np.diag(covlocal_tmp[nscale, mask, k, n] /
                                    oberrvar[mask]))
            xprime[:,:,k,n],xmean[k,n] = \
            letkf_update(hx[:, :, mask], Rinv_lst, xprime[:,:,k,n], xmean[k,n], omf[mask])

    return xprime, xmean

def bulk_ensrf_multiscale(
    xens, xensmean, indxobi, obs, oberrs, covlocal1, vcovlocal_facts, pv_scalefact
):
    """bulk potter method (global matrix solution)"""

    nlscales, nanals, nlevs, ndim1 = xens.shape
    nobs1 = obs.shape[-1]
    nobs = 2 * nobs1
    ndim = 2 * ndim1

    # create H operator
    iob = np.zeros(ndim1, np.bool)
    iob[indxobi] = True
    indxob = np.concatenate((iob, iob))

    xmean = xensmean.reshape(ndim)
    obs = obs.reshape(nobs)
    oberrstd = np.sqrt(np.concatenate((oberrs, oberrs)))
    # normalize obs by ob error stdev
    obs = obs / oberrstd
    # forward operator
    hxmean = pv_scalefact * xmean[indxob]

    Pb = np.zeros((ndim,ndim),np.float64)
    for n in range(nlscales):
        # create cov localization matrix
        covlocal = np.block(
            [
                [covlocal1[n], vcovlocal_facts[n] * covlocal1[n]],
                [vcovlocal_facts[n] * covlocal1[n], covlocal1[n]],
            ]
        )

        # create 2d state vector array
        xprime = xens[n].reshape((nanals, ndim))

        Pb += covlocal * np.dot(xprime.T, xprime) / (nanals - 1)


    # see https://doi.org/10.1175/JTECH-D-16-0140.1 eqn 5

    D = pv_scalefact ** 2 * Pb[np.ix_(indxob, indxob)] + np.eye(nobs)
    PbHT = pv_scalefact * Pb[:, indxob]
    # using Cholesky and LU decomp
    Dsqrt, info = lapack.dpotrf(D,overwrite_a=0)
    Dinv, info = lapack.dpotri(Dsqrt)
    # lapack only returns the upper triangular part
    Dinv += np.triu(Dinv, k=1).T
    kfgain = np.dot(PbHT, Dinv)
    Dsqrt = np.triu(Dsqrt)
    DplusDsqrtinv = inv(D+Dsqrt) # uses lapack dgetrf,dgetri
    reducedgain = np.dot(PbHT, DplusDsqrtinv)

    # Using eigenanalysis
    #evals, eigs, info = lapack.dsyevd(D)
    ##evals, eigs, info, isuppz, info = lapack.dsyevr(D)
    #evals = evals.clip(min=np.finfo(evals.dtype).eps)
    #Dinv = (eigs * (1.0 / evals)).dot(eigs.T)
    #kfgain = np.dot(PbHT, Dinv)
    #DplusDsqrtinv = (eigs * (1.0 / (evals + np.sqrt(evals)))).dot(eigs.T)
    #reducedgain = np.dot(PbHT, DplusDsqrtinv)

    # mean and perturbation update
    xmean += np.dot(kfgain, obs - hxmean)
    xprime_full = (xens.sum(axis=0)).reshape((nanals,ndim))
    hxprime_full = pv_scalefact * xprime_full[:, indxob] / oberrstd
    xprime_full -= np.dot(reducedgain, hxprime_full.T).T

    # back to 2d state vectors
    xens = xmean.reshape((2,ndim1)) + xprime_full.reshape((nanals, 2, ndim1))
    return xens
