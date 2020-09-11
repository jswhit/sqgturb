import numpy as np
from scipy.linalg import eigh, cho_solve, cho_factor, svd, inv, pinvh

# function definitions.

def symsqrt_psd(a, inv=False):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = eigh(a)
    symsqrt =  (eigs * np.sqrt(np.maximum(evals,0))).dot(eigs.T)
    if inv:
        inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
        return symsqrt, inv
    else:
        return symsqrt

def cartdist(x1,y1,x2,y2,xmax,ymax):
    """cartesian distance on doubly periodic plane"""
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5*xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5*ymax, ymax - dy, dy)
    return np.sqrt(dx**2 + dy**2)

def gaspcohn(r):
    """
    Gaspari-Cohn taper function.
    very close to exp(-(r/c)**2), where c = sqrt(0.15)
    r should be >0 and normalized so taper = 0 at r = 1
    """
    rr = 2.*r
    rr += 1.e-13 # avoid divide by zero warnings from numpy
    taper = np.where(r<=0.5, \
            ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
            np.zeros(r.shape,r.dtype))
    taper = np.where(np.logical_and(r>0.5,r<1.), \
            ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
               + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper

def enkf_update(xens,hxens,obs,oberrs,covlocal,vcovlocal_fact,obcovlocal=None):
    """serial potter method or LETKF (if obcovlocal is None)"""

    nanals, nlevs, ndim = xens.shape; nobs = obs.shape[-1]
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
    fact = np.array([1.,1.],np.float)

    if obcovlocal is not None:  # serial EnSRF update

        for kob in range(2):
            fact[:] = 1.; fact[1-kob] = vcovlocal_fact
            for nob,ob,oberr in zip(np.arange(nobs),obs[kob],oberrs):
                ominusf = ob-hxmean[kob,nob].copy()
                hxens = hxprime[:,kob,nob].copy().reshape((nanals, 1))
                hpbht = (hxens**2).sum()/(nanals-1)
                gainfact = ((hpbht+oberr)/hpbht*\
                           (1.-np.sqrt(oberr/(hpbht+oberr))))
                # state space update
                # only update points closer than localization radius to ob
                mask = covlocal[nob,:] > 1.e-10
                for k in range(2):
                    pbht = (xprime[:,k,mask].T*hxens[:,0]).sum(axis=1)/float(nanals-1)
                    kfgain = fact[k]*covlocal[nob,mask]*pbht/(hpbht+oberr)
                    xmean[k,mask] = xmean[k,mask] + kfgain*ominusf
                    xprime[:,k,mask] = xprime[:,k,mask] - gainfact*kfgain*hxens
                # observation space update
                # only update obs within localization radius
                mask = obcovlocal[nob,:] > 1.e-10
                for k in range(2):
                    pbht = (hxprime[:,k,mask].T*hxens[:,0]).sum(axis=1)/float(nanals-1)
                    kfgain = fact[k]*obcovlocal[nob,mask]*pbht/(hpbht+oberr)
                    hxmean[k,mask] = hxmean[k,mask] + kfgain*ominusf
                    hxprime[:,k,mask] = hxprime[:,k,mask] - gainfact*kfgain*hxens
        return xmean + xprime

    else:  # LETKF update

        ndim1 = covlocal.shape[-1]
        hx = np.empty((nanals,2*nobs),np.float)
        omf = np.empty(2*nobs,np.float)
        oberrvar = np.empty(2*nobs, np.float)
        covlocal_tmp = np.empty((2*nobs,2,ndim1),np.float)
        for kob in range(2):
            fact[:] = 1.; fact[1-kob] = vcovlocal_fact
            oberrvar[kob*nobs:(kob+1)*nobs] = oberrs[:]
            omf[kob*nobs:(kob+1)*nobs] = obs[kob,:]-hxmean[kob,:]
            hx[:,kob*nobs:(kob+1)*nobs] = hxprime[:,kob,:]
            for k in range(2):
                covlocal_tmp[kob*nobs:(kob+1)*nobs,k,:] = fact[k]*covlocal[:,:]
        def calcwts(hx,Rinv,ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals-1)*np.eye(nanals) + np.dot(YbRinv, hx.T)
            evals, eigs = np.linalg.eigh(pa)
            painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1./evals))), eigs.T)
            tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
            return np.sqrt(nanals-1)*painv + tmp[:,np.newaxis]
        for n in range(ndim1):
            for k in range(2):
                mask = covlocal_tmp[:,k,n] > 1.e-10
                Rinv = np.diag(covlocal_tmp[mask,k,n]/oberrvar[mask])
                wts = calcwts(hx[:,mask],Rinv,omf[mask])
                xens[:,k,n] = xmean[k,n] + np.dot(wts.T, xprime[:,k,n])
        return xens

def bulk_ensrf(xens,indxobi,obs,oberrs,covlocal1,vcovlocal_fact,pv_scalefact,denkf=False):
    """bulk potter method (global matrix solution)"""

    nanals, nlevs, ndim1 = xens.shape; nobs1 = obs.shape[-1]
    nobs = 2*nobs1; ndim = 2*ndim1
    xmean2 = xens.mean(axis=0); xprime2 = xens-xmean2

    # create H operator
    iob = np.zeros(ndim1,np.bool)
    iob[indxobi] = True
    indxob = np.concatenate((iob,iob))

    # create cov localization matrix
    covlocal = np.block([[covlocal1,vcovlocal_fact*covlocal1],[vcovlocal_fact*covlocal1,covlocal1]])

    # create 1d state vector arrays
    xmean = xmean2.reshape(ndim)
    xprime = xprime2.reshape((nanals,ndim))
    obs = obs.reshape(nobs)
    oberrvar = np.concatenate((oberrs,oberrs))

    # forward operator
    hxmean = pv_scalefact*xmean[indxob]
    hxprime = pv_scalefact*xprime[:,indxob]

    R = np.diag(oberrvar)
    Pb = np.dot(xprime.T,xprime)/(nanals-1)
    Pb = covlocal*Pb
    D = pv_scalefact**2*Pb[np.ix_(indxob,indxob)] + R
    eye = np.eye(nobs)
    if denkf:
        Dinv = cho_solve(cho_factor(D), eye)
    else:
        Dsqrt,Dinv = symsqrt_psd(D,inv=True)
        # check square root
        #Dtmp = np.dot(Dsqrt.T, Dsqrt)
        #print(np.allclose(Dtmp-D, np.zeros((nobs, nobs))))
        #raise SystemExit
    kfgain = np.dot(pv_scalefact*Pb[:,indxob],Dinv)
    if denkf: # approximate reduced gain
        reducedgain = 0.5*kfgain
    else:
        tmp = Dsqrt + np.sqrt(R)
        tmpinv = cho_solve(cho_factor(tmp),eye)
        reducedgain = np.dot(kfgain, np.dot(Dsqrt,tmpinv))

    # mean and perturbation update
    xmean += np.dot(kfgain, obs-hxmean)
    xprime -= np.dot(reducedgain,hxprime.T).T

    # back to 2d state vectors
    xmean2 = xmean.reshape((2,ndim1))
    xprime2 = xprime.reshape((nanals,2,ndim1))
    xens = xmean2 + xprime2

    return xens
