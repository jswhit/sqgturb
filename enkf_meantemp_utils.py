import numpy as np
from numpy.linalg import eigh
from scipy.linalg import cho_solve, cho_factor
# function definitions.

def symsqrtinv_psd(a):
    """inverse and inverse symmetric square-root of a symmetric positive
    definite matrix"""
    evals, eigs = eigh(a)
    symsqrtinv =  (eigs * (1./np.sqrt(np.maximum(evals,0)))).dot(eigs.T)
    inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
    return symsqrtinv, inv

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

def enkf_update(xens,hxens,obs,oberrs,covlocal,obcovlocal=None):
    """serial potter method or LETKF (if obcovlocal is None)"""

    nanals, ndim = xens.shape; nobs = obs.shape[-1]
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean

    if obcovlocal is not None:  # serial EnSRF update

        for nob,ob,oberr in zip(np.arange(nobs),obs,oberrs):
            omf = ob-hxmean[nob]; hx = hxprime[:,nob]
            hpbht = np.dot(hx,hx)/(nanals-1)
            gainfact = ((hpbht+oberr)/hpbht*\
                       (1.-np.sqrt(oberr/(hpbht+oberr))))
            # state space update
            pbht = np.dot(xprime.T,hx)/(nanals-1)
            kfgain = covlocal[nob]*pbht/(hpbht+oberr)
            xmean = xmean + kfgain*omf
            xprime = xprime - gainfact*kfgain*hx[:,np.newaxis]
            # observation space update
            # only update obs within localization radius
            pbht = np.dot(hxprime.T,hx)/(nanals-1)
            kfgain = obcovlocal[nob]*pbht/(hpbht+oberr)
            hxmean = hxmean + kfgain*omf
            hxprime = hxprime - gainfact*kfgain*hx[:,np.newaxis]
        return xmean + xprime

    else:  # LETKF update

        ndim1 = covlocal.shape[-1]
        def calcwts(hx,Rinv,ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals-1)*np.eye(nanals) + np.dot(YbRinv, hx.T)
            pasqrt_inv, painv = symsqrtinv_psd(pa)
            tmp = np.dot(np.dot(painv, YbRinv), ominusf)
            return np.sqrt(nanals-1)*pasqrt_inv + tmp[:,np.newaxis]
        for n in range(ndim1):
            mask = covlocal[:,n] > 1.e-10
            Rinv = np.diag(covlocal[mask,n]/oberrs[mask])
            wts = calcwts(hxprime[:,mask],Rinv,(obs-hxmean)[mask])
            xens[:,n] = xmean[n] + np.dot(wts.T, xprime[:,n])
        return xens

def enkf_update_modens(xens,hxens,fwdop,indxob,obs,oberrs,z,rs,letkf=False):
    """serial potter method or ETKF with modulated ensemble, no localization"""

    nanals, ndim = xens.shape; nobs = obs.shape[-1]
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean

    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals
    xprime2 = np.zeros((nanals2,ndim),xprime.dtype)
    hxprime2 = np.zeros((nanals2,nobs),xprime.dtype)
    nanal2 = 0
    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2,:] = xprime[nanal,:]*z[neig-j-1,:]
            nanal2 += 1
    # normalize modulated ensemble so total variance unchanged.
    var = ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #print np.sqrt(var/var2), np.sqrt(float(nanals2-1)/float(nanals-1))
    xprime2 = np.sqrt(var/var2)*xprime2
    scalefact = np.sqrt(var/var2)*z[-1].max()
    #xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    #var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #print(var,var2)

    # compute forward operator on modulated ensemble.
    for nanal in range(nanals2):
        hxprime2[nanal] = fwdop.calc(xprime2[nanal].reshape(2,fwdop.model.N,fwdop.model.N),indxob)

    if not letkf:  # serial EnSRF update

        for nob,ob,oberr in zip(np.arange(nobs),obs,oberrs):
            hx2 = hxprime2[:,nob]; hx = hxprime[:,nob]
            omf = ob-hxmean[nob]
            hpbht = np.dot(hx2,hx2)/(nanals2-1)
            gainfact = ((hpbht+oberr)/hpbht*\
                       (1.-np.sqrt(oberr/(hpbht+oberr))))
            # state space update
            pbht = np.dot(xprime2.T,hx2)/(nanals2-1)
            kfgain = pbht/(hpbht+oberr)
            xmean = xmean + kfgain*omf
            xprime2 = xprime2 - gainfact*kfgain*hx2[:,np.newaxis]
            xprime = xprime - gainfact*kfgain*hx[:,np.newaxis]
            # observation space update
            pbht = np.dot(hxprime2.T,hx2)/(nanals2-1)
            kfgain = pbht/(hpbht+oberr)
            hxmean = hxmean + kfgain*omf
            hxprime2 = hxprime2 - gainfact*kfgain*hx2[:,np.newaxis]
            hxprime = hxprime - gainfact*kfgain*hx[:,np.newaxis]
        return xmean + xprime

    else:  # ETKF update
        YbRinv = np.dot(hxprime2,(1./oberrs)*np.eye(nobs))
        pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,hxprime2.T)
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        kfgain = np.dot(xprime2.T,np.dot(painv,YbRinv))
        xmean = xmean + np.dot(kfgain, obs-hxmean).T
        enswts = np.sqrt(nanals2-1)*pasqrt_inv
        xprime = np.dot(enswts[:,0:nanals].T,xprime2)/scalefact
        return xmean + xprime
