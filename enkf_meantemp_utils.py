import numpy as np
from scipy import linalg
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

def enkf_update(xens,hxens,obs,oberrs,covlocal,obcovlocal=None):
    """serial potter method or LETKF (if obcovlocal is None)"""

    nanals, nlevs, ndim = xens.shape; nobs = obs.shape[-1]
    N = int(np.sqrt(ndim))
    xmean = xens.mean(axis=0); xprime = xens-xmean
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean

    if obcovlocal is not None:  # serial EnSRF update

        for nob,ob,oberr in zip(np.arange(nobs),obs,oberrs):
            ominusf = ob-hxmean[nob].copy()
            hxens = hxprime[:,nob].copy().reshape((nanals, 1))
            hpbht = (hxens**2).sum()/(nanals-1)
            gainfact = ((hpbht+oberr)/hpbht*\
                       (1.-np.sqrt(oberr/(hpbht+oberr))))
            # state space update
            # only update points closer than localization radius to ob
            mask = covlocal[nob,:] > -1.e-10
            for k in range(2):
                pbht = (xprime[:,k,mask].T*hxens[:,0]).sum(axis=1)/float(nanals-1)
                kfgain = covlocal[nob,mask]*pbht/(hpbht+oberr)
                #import matplotlib.pyplot as plt
                #plt.contourf(np.arange(N),np.arange(N),(kfgain*ominusf).reshape(N,N),np.linspace(-2000,2000,21),extend='both')
                #plt.colorbar()
                #plt.show()
                #raise SystemExit
                xmean[k,mask] = xmean[k,mask] + kfgain*ominusf
                xprime[:,k,mask] = xprime[:,k,mask] - gainfact*kfgain*hxens
            # observation space update
            # only update obs within localization radius
            mask = obcovlocal[nob,:] > 1.e-10
            pbht = (hxprime[:,mask].T*hxens[:,0]).sum(axis=1)/float(nanals-1)
            kfgain = obcovlocal[nob,mask]*pbht/(hpbht+oberr)
            hxmean[mask] = hxmean[mask] + kfgain*ominusf
            hxprime[:,mask] = hxprime[:,mask] - gainfact*kfgain*hxens
        return xmean + xprime

    else:  # LETKF update

        ndim1 = covlocal.shape[-1]
        def calcwts(hx,Rinv,ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals-1)*np.eye(nanals) + np.dot(YbRinv, hx.T)
            evals, eigs = linalg.eigh(pa)
            painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1./evals))), eigs.T)
            tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
            return np.sqrt(nanals-1)*painv + tmp[:,np.newaxis]
        for n in range(ndim1):
            mask = covlocal[:,n] > 1.e-10
            Rinv = np.diag(covlocal[mask,n]/oberrs[mask])
            for k in range(2):
                wts = calcwts(hxprime[:,mask],Rinv,(obs[mask]-hxmean[mask]))
                xens[:,k,n] = xmean[k,n] + np.dot(wts.T, xprime[:,k,n])
        return xens

def enkf_update_modens(xens,hxens,fwdop,model,indxob,obs,oberrs,z,letkf=False):
    """serial potter method or LETKF"""

    nanals, nlevs, ndim = xens.shape; nobs = obs.shape[-1]
    xmean = xens.mean(axis=0); xprime = xens-xmean
    #for nanal in range(nanals):
    #    hxens[nanal] = fwdop(model,xens[nanal],indxob)
    hxmean = hxens.mean(axis=0); hxprime = hxens-hxmean

    # modulation ensemble
    neig = z.shape[0]; nanals2 = neig*nanals
    xprime2 = np.zeros((nanals2,nlevs,ndim),xprime.dtype)
    hxprime2 = np.zeros((nanals2,nobs),xprime.dtype)
    for k in range(2):
        nanal2 = 0
        for j in range(neig):
            for nanal in range(nanals):
                xprime2[nanal2,k,:] = xprime[nanal,k,:]*z[neig-j-1,:]
                nanal2 += 1
    # normalize modulated ensemble so total variance unchanged.
    var = ((xprime**2).sum(axis=0)/(nanals-1)).mean()
    var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    xprime2 = np.sqrt(var/var2)*xprime2
    #xprime2 = np.sqrt(float(nanals2-1)/float(nanals-1))*xprime2
    #var2 = ((xprime2**2).sum(axis=0)/(nanals2-1)).mean()
    #print(var,var2)

    # compute forward operator on modulated ensemble.
    for nanal in range(nanals2):
        hxprime2[nanal] = fwdop(model,xprime2[nanal].reshape(nlevs,model.N,model.N),indxob)
        #hxprime2[nanal] = xprime2[nanal,0,indxob]
    if not letkf:  # serial EnSRF update

        for nob,ob,oberr in zip(np.arange(nobs),obs,oberrs):
            ominusf = ob-hxmean[nob].copy()
            hxens = hxprime2[:,nob].copy()
            hxens_orig = hxprime[:,nob].copy()
            hpbht = (hxens**2).sum()/(nanals2-1)
            gainfact = ((hpbht+oberr)/hpbht*\
                       (1.-np.sqrt(oberr/(hpbht+oberr))))
            # state space update
            for k in range(2):
                pbht = (xprime2[:,k,:].T*hxens).sum(axis=1)/float(nanals2-1)
                kfgain = pbht/(hpbht+oberr)
                #import matplotlib.pyplot as plt
                #plt.contourf(np.arange(model.N),np.arange(model.N),(kfgain*ominusf).reshape(model.N,model.N),np.linspace(-2000,2000,21),extend='both')
                #plt.colorbar()
                #plt.show()
                #raise SystemExit
                xmean[k,:] = xmean[k,:] + kfgain*ominusf
                xprime2[:,k,:] = xprime2[:,k,:] -\
                        gainfact*kfgain*hxens[:,np.newaxis]
                xprime[:,k,:] = xprime[:,k,:] -\
                        gainfact*kfgain*hxens_orig[:,np.newaxis]
            # observation space update
            # only update obs within localization radius
            pbht = (hxprime2.T*hxens).sum(axis=1)/float(nanals2-1)
            kfgain = pbht/(hpbht+oberr)
            hxmean = hxmean + kfgain*ominusf
            hxprime2 = hxprime2 - gainfact*kfgain*hxens[:,np.newaxis]
            hxprime = hxprime - gainfact*kfgain*hxens_orig[:,np.newaxis]
        return xmean + xprime

    else:  # ETKF computation of gain, perturbed obs update for ens perts.
        YbRinv = np.dot(hxprime2,(1./oberrs)*np.eye(nobs))
        pa = (nanals2-1)*np.eye(nanals2)+np.dot(YbRinv,hxprime2.T)
        painv = linalg.cho_solve(linalg.cho_factor(pa),np.eye(nanals2))
        # make sure ob noise has zero mean and correct stdev.
        obnoise =\
        np.sqrt(oberrs)*np.random.standard_normal(size=(nanals,nobs))
        obnoise_var =\
        ((obnoise-obnoise.mean(axis=0))**2).sum(axis=0)/(nanals-1)
        obnoise = np.sqrt(oberrs)*obnoise/np.sqrt(obnoise_var)
        hxprime = hxprime + obnoise - obnoise.mean(axis=0) 
        for k in range(2):
            kfgain = np.dot(xprime2[:,k,:].T,np.dot(painv,YbRinv))
            xmean[k] = xmean[k] + np.dot(kfgain, obs-hxmean).T
            xprime[:,k,:] = xprime[:,k,:] - np.dot(kfgain,hxprime.T).T
        return xmean + xprime
