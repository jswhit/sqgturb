import numpy as np

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

def enkf_update(xens,hxens,obs,oberrs,covlocal,levob,vcovlocal_fact,obcovlocal=None):
    """serial potter method or LETKF (if obcovlocal is None)"""

    nanals, nlevs, ndim = xens.shape; nobs = obs.shape[-1]
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean
    nlevob = len(levob); fact = np.array([1.,1.],np.float)

    if obcovlocal is not None:  # serial EnSRF update

        for kob in range(nlevob):
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
        hx = np.empty((nanals,nlevob*nobs),np.float)
        omf = np.empty(nlevob*nobs,np.float)
        oberrvar = np.empty(nlevob*nobs, np.float)
        covlocal_tmp = np.empty((nlevob*nobs,2,ndim1),np.float)
        for kob in range(nlevob):
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
