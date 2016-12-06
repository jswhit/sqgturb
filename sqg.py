"""
constant PV f-plane QG turbulence (a.k.a surface QG turbulence).
Doubly periodic geometry with sin(2*pi/L) jet basic state.
References:
http://journals.ametsoc.org/doi/pdf/10.1175/2008JAS2921.1 (section 3)
http://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281978%29035%3C0774%3AUPVFPI%3E2.0.CO%3B2

includes Ekman damping, linear thermal relaxation back
to equilibrium jet, and hyperdiffusion.

pv has units of meters per second.
scale by f*theta0/g to convert to temperature.

FFT spectral collocation method with 4th order Runge Kutta
time stepping (dealiasing with 2/3 rule, hyperdiffusion treated implicitly).

Jeff Whitaker December, 2016 <jeffrey.s.whitaker@noaa.gov>
"""
from __future__ import print_function
import os
import numpy as np
try: # pyfftw is *much* faster
    from pyfftw.interfaces import numpy_fft, cache
    #print('# using pyfftw...')
    cache.enable()
    rfft2 = numpy_fft.rfft2; irfft2 = numpy_fft.irfft2
except ImportError: # fall back on numpy fft.
    print('# WARNING: using numpy fft (install pyfftw for better performance)...')
    def rfft2(*args, **kwargs):
        kwargs.pop('threads',None)
        return np.fft.rfft2(*args,**kwargs)
    def irfft2(*args, **kwargs):
        kwargs.pop('threads',None)
        return np.fft.irfft2(*args,**kwargs)

class SQG:

    def __init__(self,pv,f=1.e-4,nsq=1.e-4,L=20.e6,H=10.e3,U=30.,\
                 r=0.,tdiab=10.*86400,diff_order=8,diff_efold=None,
                 symmetric=True,dt=None,dealias=True,threads=1):
        # initialize SQG model.
        if pv.shape[0] != 2:
            raise ValueError('1st dim of pv should be 2')
        N = pv.shape[1] # number of grid points in each direction
        # N should be even
        if N%2:
            raise ValueError('N must be even (powers of 2 are fastest)')
        if dt is None: # time step must be specified
            raise ValueError('must specify time step')
        if diff_efold is None: # efolding time scale for diffusion must be specified
            raise ValueError('must specify efolding time scale for diffusion')
        # number of openmp threads to use for FFTs (only for pyfftw)
        self.threads = threads
        self.N = N
        # force arrays to be float32 (ffts are twice as fast)
        self.nsq = np.array(nsq,np.float32) # Brunt-Vaisalla (buoyancy) freq squared
        self.f = np.array(f,np.float32) # coriolis
        self.H = np.array(H,np.float32) # height of upper boundary
        self.U = np.array(U,np.float32) # basic state velocity at z = H
        self.L = np.array(L,np.float32) # size of square domain.
        self.dt = np.array(dt,np.float32) # time step (seconds)
        self.dealias = dealias  # if True, dealiasing applied using 2/3 rule.
        if r < 1.e-10:
            self.ekman = False
        else:
            self.ekman = True
        self.r = np.array(r,np.float32) # Ekman damping (at z=0)
        self.tdiab = np.array(tdiab,np.float32) # thermal relaxation damping.
        self.t = 0 # initialize time counter
        # setup basic state pv (for thermal relaxation)
        self.symmetric = symmetric # symmetric jet, or jet with U=0 at sfc.
        y = np.arange(0,L,L/N,dtype=np.float32)
        pvbar = np.zeros((2,N),np.float32)
        pi = np.array(np.pi,np.float32)
        l = 2.*pi/L; mu = l*np.sqrt(nsq)*H/f
        if symmetric:
            # symmetric version, no difference between upper and lower
            # boundary.
            # l = 2.*pi/L and mu = l*N*H/f
            # u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
            # theta = (f*theta0/g)*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*np.cos(l*y)/np.sinh(0.5*mu) + \
            #         theta0 + (theta0*nsq*z/g)
            pvbar[:] = -(mu*0.5*U/(l*H))*np.cosh(0.5*mu)*np.cos(l*y)/np.sinh(0.5*mu)
        else:
            # asymmetric version, equilibrium state has no flow at surface and
            # temp gradient slightly weaker at sfc.
            # l = 2.*pi/L and mu = l*N*H/f
            # u = U*np.sin(l*y)*np.sinh(mu*z/H)*np.sin(l*y)/np.sinh(mu)
            # theta = (f*theta0/g)*(U*mu/(l*H))*np.cosh(mu*z/H)*np.cos(l*y)/np.sinh(mu) +\
            # theta0 + (theta0*nsq*z/g)
            pvbar[:]   = -(mu*U/(l*H))*np.cos(l*y)/np.sinh(mu)
            pvbar[1,:] = pvbar[0,:]*np.cosh(mu)
        pvbar.shape = (2,N,1)
        pvbar = pvbar*np.ones((2,N,N),np.float32)
        self.pvbar = pvbar
        self.pvspec_eq = rfft2(pvbar) # state to relax to with timescale tdiab
        self.pvspec = rfft2(pv) # initial pv field (spectral)
        # spectral stuff
        k = (N*np.fft.fftfreq(N))[0:(N/2)+1]
        l = N*np.fft.fftfreq(N)
        k,l = np.meshgrid(k,l)
        k = k.astype(np.float32); l = l.astype(np.float32)
        # dimensionalize wavenumbers.
        k = 2.*pi*k/self.L; l = 2.*pi*l/self.L
        ksqlsq = k**2+l**2
        self.k = k; self.l = l; self.ksqlsq = ksqlsq
        self.ik = (1.j*k).astype(np.complex64)
        self.il = (1.j*l).astype(np.complex64)
        if dealias: # arrays needed for dealiasing nonlinear Jacobian
            k_pad = ((3*N/2)*np.fft.fftfreq(3*N/2))[0:(3*N/4)+1]
            l_pad = (3*N/2)*np.fft.fftfreq(3*N/2)
            k_pad,l_pad = np.meshgrid(k_pad,l_pad)
            k_pad = k_pad.astype(np.float32); l_pad = l_pad.astype(np.float32)
            k_pad = 2.*pi*k_pad/self.L; l_pad = 2.*pi*l_pad/self.L
            # add factor of (3/2)**2 to account for rescaling
            # of padded inverse transform (inverse transform is normalized
            # by 1/N in each direction).
            self.ik_pad = (1.j*k_pad).astype(np.complex64)
            self.il_pad = (1.j*l_pad).astype(np.complex64)
        mu = np.sqrt(ksqlsq)*np.sqrt(self.nsq)*self.H/self.f
        mu = mu.clip(np.finfo(mu.dtype).eps) # clip to avoid NaN
        self.Hovermu = self.H/mu
        mu = mu.astype(np.float64) # cast to avoid overflow in sinh
        self.tanhmu = np.tanh(mu).astype(np.float32) # cast back to float32
        self.sinhmu = np.sinh(mu).astype(np.float32)
        self.diff_order = np.array(diff_order,np.float32) # hyperdiffusion order
        self.diff_efold = np.array(diff_efold,np.float32) # hyperdiff time scale
        ktot = np.sqrt(ksqlsq)
        ktotcutoff = np.array(pi*N/self.L, np.float32)
        # integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff =\
        np.exp((-self.dt/self.diff_efold)*(ktot/ktotcutoff)**self.diff_order)
        # number of timesteps to advance each call to 'advance' method.
        self.timesteps = 1

    def invert(self,pvspec=None):
        if pvspec is None: pvspec = self.pvspec
        # invert boundary pv to get streamfunction
        psispec = np.empty((2,self.N,self.N/2+1),dtype=pvspec.dtype)
        psispec[0] = self.Hovermu*((pvspec[1]/self.sinhmu) -\
                                   (pvspec[0]/self.tanhmu))
        psispec[1] = self.Hovermu*((pvspec[1]/self.tanhmu) -\
                                   (pvspec[0]/self.sinhmu))
        return psispec

    def advance(self,pv=None):
        # given total pv on grid, advance forward
        # number of timesteps given by 'timesteps' instance var.
        # if pv not specified, use pvspec instance variable.
        if pv is not None:
            self.pvspec = rfft2(pv,threads=self.threads)
        for n in range(self.timesteps):
            self.timestep()
        return irfft2(self.pvspec,threads=self.threads)

    def specpad(self, specarr):
        # pad spectral arrays with zeros to get
        # interpolation to 3/2 larger grid using inverse fft.
        # take care of normalization factor for inverse transform.
        specarr_pad = np.zeros((2, 3*self.N/2, 3*self.N/4+1), specarr.dtype)
        specarr_pad[:,0:self.N/2,0:self.N/2] = 2.25*specarr[:,0:self.N/2,0:self.N/2]
        specarr_pad[:,-self.N/2:,0:self.N/2] = 2.25*specarr[:,-self.N/2:,0:self.N/2]
        # include negative Nyquist frequency.
        specarr_pad[:,0:self.N/2,self.N/2]=np.conjugate(2.25*specarr[:,0:self.N/2,-1])
        specarr_pad[:,-self.N/2:,self.N/2]=np.conjugate(2.25*specarr[:,-self.N/2:,-1])
        return specarr_pad

    def spectrunc(self, specarr):
        # truncate spectral array using 2/3 rule.
        specarr_trunc = np.zeros((2, self.N, (self.N/2)+1), specarr.dtype)
        specarr_trunc[:,0:self.N/2,0:self.N/2] = specarr[:,0:self.N/2,0:self.N/2]
        specarr_trunc[:,-self.N/2:,0:self.N/2] = specarr[:,-self.N/2:,0:self.N/2]
        return specarr_trunc

    def gettend(self,pvspec=None):
        # compute tendencies of pv on z=0,H
        # invert pv to get streamfunction
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        # nonlinear jacobian and thermal relaxation
        if not self.dealias:
            u = irfft2(-self.il*psispec,threads=self.threads)
            v = irfft2(self.ik*psispec,threads=self.threads)
            pvx = irfft2(self.ik*pvspec,threads=self.threads)
            pvy = irfft2(self.il*pvspec,threads=self.threads)
        else: # pad spectral coeffs with zeros for dealiased jacobian
            psispec_pad = self.specpad(psispec)
            pvspec_pad  = self.specpad(pvspec)
            u = irfft2(-self.il_pad*psispec_pad,threads=self.threads)
            v = irfft2(self.ik_pad*psispec_pad,threads=self.threads)
            pvx = irfft2(self.ik_pad*pvspec_pad,threads=self.threads)
            pvy = irfft2(self.il_pad*pvspec_pad,threads=self.threads)
        advection = u*pvx + v*pvy
        jacobianspec = rfft2(advection,threads=self.threads)
        if self.dealias: # 2/3 rule: truncate spectral coefficients of jacobian
            jacobianspec = self.spectrunc(jacobianspec)
        dpvspecdt = (1./self.tdiab)*(self.pvspec_eq-pvspec)-jacobianspec
        # Ekman damping at boundaries.
        if self.ekman:
            dpvspecdt[0] += self.r*self.ksqlsq*psispec[0]
            # for asymmetric jet (U=0 at sfc), no Ekman layer at lid
            if self.symmetric:
                dpvspecdt[1] -= self.r*self.ksqlsq*psispec[1]
        # save wind field
        self.u = u; self.v = v
        return dpvspecdt

    def timestep(self):
        # update pv using 4th order runge-kutta time step with
        # implicit "integrating factor" treatment of hyperdiffusion.
        k1 = self.dt*self.gettend(self.pvspec)
        k2 = self.dt*self.gettend(self.pvspec + 0.5*k1)
        k3 = self.dt*self.gettend(self.pvspec + 0.5*k2)
        k4 = self.dt*self.gettend(self.pvspec + k3)
        self.pvspec = self.hyperdiff*(self.pvspec + (k1+2.*k2+2.*k3+k4)/6.)
        self.t += self.dt # increment time
