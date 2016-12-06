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
time stepping (dealiasing with 3/2 rule, hyperdiffusion treated implicitly).

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
        self.nsq = np.array(nsq,np.float32) # Brunt-Vaisalla (buoyancy) freq squared
        self.f = np.array(f,np.float32) # coriolis
        self.H = np.array(H,np.float32) # height of upper boundary
        self.U = np.array(U,np.float32) # basic state velocity at z = H
        self.L = np.array(L,np.float32) # size of square domain.
        self.dt = np.array(dt,np.float32) # time step (seconds)
        self.dealias = dealias  # if True, dealiasing applied using 3/2 rule.
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
            self.ik_pad = 2.25*(1.j*k_pad).astype(np.complex64)
            self.il_pad = 2.25*(1.j*l_pad).astype(np.complex64)
        self.mu = np.sqrt(ksqlsq)*np.sqrt(self.nsq)*self.H/self.f
        self.mu = self.mu.clip(np.finfo(self.mu.dtype).eps) # clip to avoid NaN
        self.Hovermu = self.H/self.mu
        self.tanhmu = np.tanh(self.mu)
        self.sinhmu = np.sinh(self.mu)
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
            psispec_pad = np.zeros((2,)+self.ik_pad.shape, psispec.dtype)
            pvspec_pad = np.zeros((2,)+self.ik_pad.shape, pvspec.dtype)
            psispec_pad[:,0:self.N/2,0:self.N/2] = psispec[:,0:self.N/2,0:self.N/2]
            psispec_pad[:,-self.N/2:,0:self.N/2] = psispec[:,-self.N/2:,0:self.N/2]
            # include negative Nyquist frequency.
            psispec_pad[:,0:self.N/2,self.N/2]=np.conjugate(psispec[:,0:self.N/2,-1])
            psispec_pad[:,-self.N/2:,self.N/2]=np.conjugate(psispec[:,-self.N/2:,-1])
            pvspec_pad[:,0:self.N/2,0:self.N/2] = pvspec[:,0:self.N/2,0:self.N/2]
            pvspec_pad[:,-self.N/2:,0:self.N/2] = pvspec[:,-self.N/2:,0:self.N/2]
            # include negative Nyquist frequency.
            pvspec_pad[:,0:self.N/2,self.N/2]=np.conjugate(pvspec[:,0:self.N/2,-1])
            pvspec_pad[:,-self.N/2:,self.N/2]=np.conjugate(pvspec[:,-self.N/2:,-1])
            u = irfft2(-self.il_pad*psispec_pad,threads=self.threads)
            v = irfft2(self.ik_pad*psispec_pad,threads=self.threads)
            pvx = irfft2(self.ik_pad*pvspec_pad,threads=self.threads)
            pvy = irfft2(self.il_pad*pvspec_pad,threads=self.threads)
        advection = u*pvx+v*pvy
        tmpspec = rfft2(advection,threads=self.threads)
        if self.dealias: # truncate spectral coefficients of jacobian.
            advspec = np.zeros(pvspec.shape, pvspec.dtype)
            advspec[:,0:self.N/2,0:self.N/2] = tmpspec[:,0:self.N/2,0:self.N/2]
            advspec[:,-self.N/2:,0:self.N/2] = tmpspec[:,-self.N/2:,0:self.N/2]
            # include negative Nyquist frequency.
            advspec[:,0:self.N/2,-1] = np.conjugate(tmpspec[:,0:self.N/2,self.N/2])
            advspec[:,-self.N/2:,-1] = np.conjugate(tmpspec[:,-self.N/2:,self.N/2])
        else:
            advspec = tmpspec
        dpvspecdt = (1./self.tdiab)*(self.pvspec_eq-pvspec)-advspec
        # Ekman damping at boundaries.
        if self.ekman:
            dpvspecdt[0] += self.r*self.ksqlsq*psispec[0]
            # for asymmetric jet (U=0 at sfc), no Ekman layer at lid
            if self.symmetric:
                dpvspecdt[1] -= self.r*self.ksqlsq*psispec[1]
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # run SQG turbulence simulation, plotting results to screen and/or saving to
    # netcdf file.

    # model parameters.
    N = 256 # number of grid points in each direction (waves=N/2)
    dt = 240 # time step in seconds
    # efolding time scale (seconds) for smallest wave (N/2) in del**norder hyperdiffusion
    norder = 8; diff_efold = 2400
    dealias = True # dealiased with 3/2 rule?
    # Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
    # Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
    # for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
    # corresponding to ekman depth of 141-316 m over ocean.
    # spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
    # H=10km, f=0.0001, dek=100m.
    dek = 0. # applied only at surface if symmetric=False
    nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
    H = 10.e3 # lid height
    r = dek*nsq/f
    U = 30 # jet speed
    Lr = np.sqrt(nsq)*H/f # Rossby radius
    L = 20.*Lr
    # thermal relaxation time scale
    tdiab = 10.*86400 # in seconds
    symmetric = True # (asymmetric equilibrium jet with zero wind at sfc)
    # parameter used to scale PV to temperature units.
    scalefact = f*theta0/g

    # create random noise
    pv = np.random.normal(0,100.,size=(2,N,N)).astype(np.float32)
    # add isolated blob on lid
    nexp = 20
    x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
    x,y = np.meshgrid(x,y)
    x = x.astype(np.float32); y = y.astype(np.float32)
    pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
    # remove area mean from each level.
    for k in range(2):
        pv[k] = pv[k] - pv[k].mean()

    # get OMP_NUM_THREADS (threads to use) from environment.
    threads = int(os.getenv('OMP_NUM_THREADS','1'))

    # initialize qg model instance
    model = SQG(pv,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
                diff_order=norder,diff_efold=diff_efold,
                dealias=dealias,symmetric=symmetric,threads=threads)

    #  initialize figure.
    outputinterval = 3600. # interval between frames in seconds
    tmin = 100.*86400. # time to start saving data (in days)
    tmax = 200.*86400. # time to stop (in days)
    nsteps = int(tmax/outputinterval) # number of time steps to animate
    # set number of timesteps to integrate for each call to model.advance
    model.timesteps = int(outputinterval/model.dt)
    if dealias:
        savedata = 'data/sqg_N%s_dealiased.nc' % N # save data plotted in a netcdf file.
    else:
        savedata = 'data/sqg_N%s_aliased.nc' % N # save data plotted in a netcdf file.
    #savedata = None # don't save data
    plot = True # animate data as model is running?

    if savedata is not None:
        from netCDF4 import Dataset
        nc = Dataset(savedata, mode='w', format='NETCDF4_CLASSIC')
        nc.r = model.r
        nc.f = model.f
        nc.U = model.U
        nc.L = model.L
        nc.H = model.H
        nc.g = g; nc.theta0 = theta0
        nc.nsq = model.nsq
        nc.tdiab = model.tdiab
        nc.dt = model.dt
        nc.diff_efold = model.diff_efold
        nc.diff_order = model.diff_order
        nc.symmetric = int(model.symmetric)
        nc.dealias = int(model.dealias)
        x = nc.createDimension('x',N)
        y = nc.createDimension('y',N)
        z = nc.createDimension('z',2)
        t = nc.createDimension('t',None)
        pvvar =\
        nc.createVariable('pv',np.float32,('t','z','y','x'),zlib=True)
        pvvar.units = 'K'
        # pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
        xvar = nc.createVariable('x',np.float32,('x',))
        xvar.units = 'meters'
        yvar = nc.createVariable('y',np.float32,('y',))
        yvar.units = 'meters'
        zvar = nc.createVariable('z',np.float32,('z',))
        zvar.units = 'meters'
        tvar = nc.createVariable('t',np.float32,('t',))
        tvar.units = 'seconds'
        xvar[:] = np.arange(0,model.L,model.L/N)
        yvar[:] = np.arange(0,model.L,model.L/N)
        zvar[0] = 0; zvar[1] = model.H

    levplot = 1; nout = 0
    if plot:
        fig = plt.figure(figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0.0, top=1., right=1.)
        vmin = scalefact*model.pvbar[levplot].min()
        vmax = scalefact*model.pvbar[levplot].max()
        def initfig():
            global im
            ax = fig.add_subplot(111)
            ax.axis('off')
            pv = irfft2(model.pvspec[levplot])  # spectral to grid
            im = ax.imshow(scalefact*pv,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
            return im,
        def updatefig(*args):
            global nout
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            hr = t/3600.
            spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
            print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
            im.set_data(scalefact*pv[levplot])
            if savedata is not None and t >= tmin:
                print('saving data at t = t = %g hours' % hr)
                pvvar[nout,:,:,:] = pv
                tvar[nout] = t
                nc.sync()
                if t >= tmax: nc.close()
                nout = nout + 1
            return im,

        # interval=0 means draw as fast as possible
        ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
              init_func=initfig,interval=0,blit=True)
        plt.show()
    else:
        t = 0.0
        while t < tmax:
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            hr = t/3600.
            spd = np.sqrt(model.u[levplot]**2+model.v[levplot]**2)
            print(hr,spd.max(),scalefact*pv.min(),scalefact*pv.max())
            if savedata is not None and t >= tmin:
                print('saving data at t = t = %g hours' % hr)
                pvvar[nout,:,:,:] = pv
                tvar[nout] = t
                nc.sync()
                if t >= tmax: nc.close()
                nout = nout + 1
