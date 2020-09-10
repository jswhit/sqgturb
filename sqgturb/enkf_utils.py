import numpy as np
from numpy.linalg import eigh # scipy.linalg.eigh broken on my mac
from scipy.linalg import cho_solve, cho_factor, svd, inv

def symsqrt_psd(a, inv=False):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = eigh(a)
    symsqrt =  (eigs * np.sqrt(np.maximum(evals,0))).dot(eigs.T)
    if inv:
        inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
        return symsqrt, inv
    else:
        return symsqrt

def symsqrtinv_psd(a):
    """inverse and inverse symmetric square-root of a symmetric positive
    definite matrix"""
    evals, eigs = eigh(a)
    symsqrtinv =  (eigs * (1./np.sqrt(np.maximum(evals,0)))).dot(eigs.T)
    inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
    return symsqrtinv, inv

# function definitions.

def cartdist(x1,y1,x2,y2,xmax,ymax):
    # cartesian distance on doubly periodic plane
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dx = np.where(dx > 0.5*xmax, xmax - dx, dx)
    dy = np.where(dy > 0.5*ymax, ymax - dy, dy)
    return np.sqrt(dx**2 + dy**2)

def gaspcohn(r):
    # Gaspari-Cohn taper function.
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be >0 and normalized so taper = 0 at r = 1
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
    """bulk potter method"""

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
    if denkf:
        Dinv = cho_solve(cho_factor(D),np.eye(nobs))
    else:
        Dsqrt,Dinv = symsqrt_psd(D,inv=True)
    kfgain = np.dot(pv_scalefact*Pb[:,indxob],Dinv)
    if denkf: # approximate reduced gain
        reducedgain = 0.5*kfgain
    else:
        tmp = Dsqrt + np.sqrt(R)
        tmpinv = cho_solve(cho_factor(tmp),np.eye(nobs))
        gainfact = np.dot(Dsqrt,tmpinv)
        reducedgain = np.dot(kfgain, gainfact)

    # mean and perturbation update
    xmean += np.dot(kfgain, obs-hxmean)
    xprime -= np.dot(reducedgain,hxprime.T).T

    # back to 2d state vectors
    xmean2 = xmean.reshape((2,ndim1))
    xprime2 = xprime.reshape((nanals,2,ndim1))
    xens = xmean2 + xprime2

    return xens

#def etkf_update_modens(xens,hxens,obs,oberrs,indxob,covlocalsqrt,pv_scalefact):
#    """ETKF with modulated ensemble model space localization"""
#
#    nanals, ndim = xens.shape; nobs = obs.shape[-1]
#    xmean = xens.mean(axis=0); xprime = xens-xmean
#    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
#
#    # modulation ensemble
#    neig = covlocalsqrt.shape[0]; nanals2 = neig*nanals
#    xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
#    hxprime2 = np.zeros((nanals2,nobs),xprime.dtype)
#    nanal2 = 0
#    for j in range(neig):
#        for nanal in range(nanals):
#            xprime2[nanal2,:] = xprime[nanal,:]*covlocalsqrt[neig-j-1,:]
#            nanal2 += 1
#    # normalize modulated ensemble so total variance unchanged.
#    var = ((xprime**2).sum(axis=0)/(nanals-1)).mean()
#    var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
#    #print np.sqrt(var/var2), np.sqrt(float(nanals2-1)/float(nanals-1))
#    xprime2 = np.sqrt(var/var2)*xprime2
#    # note: this assumes that 1st eigenvector is constant, or at least positive definite.
#    # scalefact is rescaled first eigenvector of covlocal.
#    scalefact = np.sqrt(var/var2)*covlocalsqrt[-1].max()
#    #xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
#    #var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
#    #print(var,var2)
#
#    # compute forward operator on modulated ensemble.
#    for nanal in range(nanals2):
#        hxprime2[nanal] = (pv_scalefact*xprime2[nanal]).ravel()[indoxb]
#
#    YbRinv = np.dot(hxprime2,(1./oberrs)*np.eye(nobs))
#    pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,hxprime2.T)
#    pasqrt_inv, painv = symsqrtinv_psd(pa)
#    kfgain = np.dot(xprime2.T,np.dot(painv,YbRinv))
#    xmean = xmean + np.dot(kfgain, obs-hxmean).T
#    #if denkf:
#    #    xprime = xprime - np.dot(0.5*kfgain, hxprime).T
#    #else: # demodulated ETKF (requires positive definite scalefact)
#    enswts = np.sqrt(nanals2-1)*pasqrt_inv
#    xprime = np.dot(enswts[:,0:nanals].T,xprime2)/scalefact
#
#    return xmean + xprime
