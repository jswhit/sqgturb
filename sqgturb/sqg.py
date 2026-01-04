import numpy as np

from pyfftw.interfaces import numpy_fft, cache
cache.enable()
rfft2 = numpy_fft.rfft2
irfft2 = numpy_fft.irfft2

class SQG:
    def __init__(
        self,
        pv,
        f=1.0e-4,
        nsq=1.0e-4,
        L=20.0e6,
        H=10.0e3,
        U=30.0,
        r=0.0,
        tdiab=10.0 * 86400,
        diff_order=8,
        diff_efold=None,
        theta0=300,
        g=9.8,
        dt=None,
        threads=1,
        precision="single",
        tstart=0,
    ):
        # initialize SQG model.
        if pv.shape[0] != 2:
            raise ValueError("1st dim of pv should be 2")
        N = pv.shape[1]  # number of grid points in each direction
        # N should be even
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:  # time step must be specified
            raise ValueError("must specify time step")
        if diff_efold is None:  # efolding time scale for diffusion must be specified
            raise ValueError("must specify efolding time scale for diffusion")
        # number of openmp threads to use for FFTs (only for pyfftw)
        self.threads = threads
        self.N = N
        if precision == "single":
            # ffts in single precision (faster)
            dtype = np.float32
        elif precision == "double":
            # ffts in double precision
            dtype = np.float64
        else:
            msg = "precision must be 'single' or 'double'"
            raise ValueError(msg)
        # force arrays to be float32 for precision='single' (ffts are twice as fast)
        self.nsq = np.array(nsq, dtype)  # Brunt-Vaisalla (buoyancy) freq squared
        self.f = np.array(f, dtype)  # coriolis
        self.H = np.array(H, dtype)  # height of upper boundary
        self.U = np.array(U, dtype)  # basic state velocity 
        self.L = np.array(L, dtype)  # size of square domain
        # theta0,g only used to convert pv to temp units (K).
        self.theta0 = np.array(theta0, dtype) # mean temp
        self.g = np.array(g, dtype) # gravity
        self.dt = np.array(dt, dtype)  # time step (seconds)
        self.r = np.empty(2, dtype)
        self.r[0]=r; self.r[1]=-r  # Ekman damping 
        self.tdiab = np.array(tdiab, dtype)  # thermal relaxation damping.
        self.t = tstart  # initialize time counter
        # setup basic state pv (for thermal relaxation)
        y = np.arange(0, L, L / N, dtype=dtype)
        pvbar = np.zeros((2, N), dtype)
        pi = np.array(np.pi, dtype)
        l = 2.0 * pi / L
        mu = l * np.sqrt(nsq) * H / f
        # symmetric state, no difference between upper and lower
        # boundary.
        # l = 2.*pi/L and mu = l*N*H/f
        # u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
        # theta = (f*theta0/g)*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*
        # np.cos(l*y)/np.sinh(0.5*mu)
        # + theta0 + (theta0*nsq*z/g)
        pvbar[:] = (
            -(mu * 0.5 * U / (l * H))
            * np.cosh(0.5 * mu)
            * np.cos(l * y)
            / np.sinh(0.5 * mu)
        )
        pvbar.shape = (2, N, 1)
        pvbar = pvbar * np.ones((2, N, N), dtype)
        self.pvbar = pvbar
        self.pvspec_eq = rfft2(pvbar)  # state to relax to with timescale tdiab
        self.pvspec = rfft2(pv)  # initial pv field (spectral)
        # spectral stuff
        k = N * np.fft.rfftfreq(N)
        l = N * np.fft.fftfreq(N)
        kk, ll = np.meshgrid(k, l)
        k = kk.astype(dtype)
        l = ll.astype(dtype)
        # dimensionalize wavenumbers.
        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L
        ksqlsq = k ** 2 + l ** 2
        self.k = k
        self.l = l
        self.ksqlsq = ksqlsq
        self.ik = (1.0j * k).astype(np.complex64)
        self.il = (1.0j * l).astype(np.complex64)
        self.wavenums = np.sqrt(kk**2+ll**2)
        k_pad = (3 * N // 2) * np.fft.rfftfreq(3 * N // 2)
        l_pad = (3 * N // 2) * np.fft.fftfreq(3 * N // 2)
        k_pad, l_pad = np.meshgrid(k_pad, l_pad)
        k_pad = k_pad.astype(dtype)
        l_pad = l_pad.astype(dtype)
        k_pad = 2.0 * pi * k_pad / self.L
        l_pad = 2.0 * pi * l_pad / self.L
        self.ik_pad = (1.0j * k_pad).astype(np.complex64)
        self.il_pad = (1.0j * l_pad).astype(np.complex64)
        mu = np.sqrt(ksqlsq) * np.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(np.finfo(mu.dtype).eps)  # clip to avoid NaN
        self.Hovermu = self.H / mu
        mu = mu.astype(np.float64)  # cast to avoid overflow in sinh
        self.tanhmu = np.tanh(mu).astype(dtype)  # cast back to original type
        self.sinhmu = np.sinh(mu).astype(dtype)
        self.diff_order = np.array(diff_order, dtype)  # hyperdiffusion order
        self.diff_efold = np.array(diff_efold, dtype)  # hyperdiff time scale
        ktot = np.sqrt(ksqlsq)
        ktotcutoff = np.array(pi * N / self.L, dtype)
        # integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff = np.exp(
            (-self.dt / self.diff_efold) * (ktot / ktotcutoff) ** self.diff_order
        )

    def invert(self, pvspec=None):
        if pvspec is None:
            pvspec = self.pvspec
        # invert boundary pv to get streamfunction
        psispec = np.empty((2, self.N, self.N // 2 + 1), dtype=pvspec.dtype)
        psispec[0] = self.Hovermu * (
            (pvspec[1] / self.sinhmu) - (pvspec[0] / self.tanhmu)
        )
        psispec[1] = self.Hovermu * (
            (pvspec[1] / self.tanhmu) - (pvspec[0] / self.sinhmu)
        )
        return psispec

    def advance(self, timesteps=1, pv=None):
        # given total pv on grid, advance forward timesteps time steps.
        # if pv not specified, use pvspec instance variable.
        if pv is not None:
            self.pvspec = rfft2(pv, threads=self.threads)
        for n in range(timesteps):
            self.timestep()
        return irfft2(self.pvspec, threads=self.threads)

    def specpad(self, specarr):
        # pad spectral arrays with zeros to get
        # interpolation to 3/2 larger grid using inverse fft.
        # take care of normalization factor for inverse transform.
        specarr_pad = np.zeros((2, 3 * self.N // 2, 3 * self.N // 4 + 1), specarr.dtype)
        specarr_pad[:, 0 : self.N // 2, 0 : self.N // 2] = (
            2.25 * specarr[:, 0 : self.N // 2, 0 : self.N // 2]
        )
        specarr_pad[:, -self.N // 2 :, 0 : self.N // 2] = (
            2.25 * specarr[:, -self.N // 2 :, 0 : self.N // 2]
        )
        # include negative Nyquist frequency.
        specarr_pad[:, 0 : self.N // 2, self.N // 2] = np.conjugate(
            2.25 * specarr[:, 0 : self.N // 2, -1]
        )
        specarr_pad[:, -self.N // 2 :, self.N // 2] = np.conjugate(
            2.25 * specarr[:, -self.N // 2 :, -1]
        )
        return specarr_pad

    def spectrunc(self, specarr):
        # truncate spectral array using 2/3 rule.
        specarr_trunc = np.zeros((2, self.N, self.N // 2 + 1), specarr.dtype)
        specarr_trunc[:, 0 : self.N // 2, 0 : self.N // 2] = specarr[
            :, 0 : self.N // 2, 0 : self.N // 2
        ]
        specarr_trunc[:, -self.N // 2 :, 0 : self.N // 2] = specarr[
            :, -self.N // 2 :, 0 : self.N // 2
        ]
        return specarr_trunc

    def xyderiv(self, specarr):
        # pad spectral coeffs with zeros for dealiased jacobian
        specarr_pad = self.specpad(specarr)
        xderiv = irfft2(self.ik_pad * specarr_pad, threads=self.threads)
        yderiv = irfft2(self.il_pad * specarr_pad, threads=self.threads)
        return xderiv, yderiv

    def gettend(self, pvspec=None):
        # compute tendencies of pv on z=0,H
        # invert pv to get streamfunction
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        # nonlinear jacobian and thermal relaxation
        psix, psiy = self.xyderiv(psispec)
        pvx, pvy = self.xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = rfft2(jacobian, threads=self.threads)
        # 2/3 rule: truncate spectral coefficients of jacobian
        jacobianspec = self.spectrunc(jacobianspec)
        dpvspecdt = (1.0 / self.tdiab) * (self.pvspec_eq - pvspec) - jacobianspec + self.r[:,np.newaxis,np.newaxis] * self.ksqlsq * psispec
        # save wind field
        self.u = -psiy
        self.v = psix
        return dpvspecdt

    def timestep(self):
        # update pv using 4th order runge-kutta time step with
        # implicit "integrating factor" treatment of hyperdiffusion.
        self.rkstep = 0
        k1 = self.dt * self.gettend(self.pvspec)
        self.rkstep = 1
        k2 = self.dt * self.gettend(self.pvspec + 0.5 * k1)
        self.rkstep = 2
        k3 = self.dt * self.gettend(self.pvspec + 0.5 * k2)
        self.rkstep = 3
        k4 = self.dt * self.gettend(self.pvspec + k3)
        pvspecnew = self.pvspec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.pvspec = self.hyperdiff * pvspecnew
        self.t += self.dt  # increment time

if __name__ == "__main__":
    import matplotlib, os
    matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    N = 96 # size of domain 
    dt = 900 # time step in seconds
    diff_efold = 86400./2. # hyperdiffusion dampling time scale on shortest wave
    norder = 8 # order of hyperdiffusion
    r = 0 # Ekman damping 
    nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
    H = 10.e3 # lid height
    U = 20 # jet speed
    L = 20.e6
    # thermal relaxation time scale
    tdiab = 10.*86400 # in seconds
    # parameter used to scale PV to temperature units.
    scalefact = f*theta0/g
    
    # create random noise
    rs = np.random.RandomState(42) # fixed seed
    pv = rs.normal(0,0.,size=(2,N,N)).astype(np.float32)
    # add isolated blob on lid
    nexp = 20
    x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
    x,y = np.meshgrid(x,y)
    pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
    # remove area mean from each level.
    for k in range(2):
        pv[k] = pv[k] - pv[k].mean()
    
    # get OMP_NUM_THREADS (threads to use) from environment.
    threads = int(os.getenv('OMP_NUM_THREADS','1'))
    
    # single or double precision
    precision='single' # pyfftw FFTs twice as fast as double
    
    # initialize qg model instance
    model = SQG(pv,nsq=nsq,f=f,U=U,L=L,H=H,r=r,tdiab=tdiab,dt=dt,
                diff_order=norder,diff_efold=diff_efold,
                threads=threads,
                precision=precision,tstart=0)
    
    #  initialize figure.
    outputinterval = 6.*3600. # interval between frames in seconds
    tmax = 300.*86400. # time to stop (in days)
    nsteps = int(tmax/outputinterval) # number of time steps to animate
    # set number of timesteps to integrate for each call to model.advance
    ntimesteps = int(outputinterval/model.dt)
    
    nout = 0 
    fig = plt.figure(figsize=(14,8))
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
    vmin = scalefact*model.pvbar.min()
    vmax = scalefact*model.pvbar.max()
    def initfig():
        global im1,im2
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        pv = irfft2(model.pvspec[0])  # spectral to grid
        im1 = ax1.imshow(scalefact*pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        pv = irfft2(model.pvspec[1])  
        im2 = ax2.imshow(scalefact*pv,cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
        return im1,im2,
    def updatefig(*args):
        global nout
        model.advance(timesteps=ntimesteps)
        pv = irfft2(model.pvspec[0])
        print(model.t/3600.,scalefact*pv.min(),scalefact*pv.max())
        im1.set_data(scalefact*pv)
        pv = irfft2(model.pvspec[1]) 
        im2.set_data(scalefact*pv)
        return im1,im2,
    
    # interval=0 means draw as fast as possible
    ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
          init_func=initfig,interval=0,blit=True)
    plt.show()
