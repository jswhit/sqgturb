import numpy as np

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

def newDistArrayGrid(FFT):
    distarr = newDistArray(FFT, False) 
    distarr = np.tile(distarr, (2,1,1))
    return distarr

def newDistArraySpec(FFT):
    distarr = newDistArray(FFT, True) 
    distarr = np.tile(distarr, (2,1,1))
    return distarr

def fft_forward(FFT, distarr):
    distarr_spec = newDistArraySpec(FFT)
    for k in range(2):
        distarr_spec[k] = FFT.forward(distarr[k], distarr_spec[k])
    return distarr_spec

def fft_backward(FFT, distarr_spec):
    distarr = newDistArrayGrid(FFT)
    for k in range(2):
        distarr[k] = FFT.backward(distarr_spec[k])
    return distarr

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
        precision="single",
        backend="fftw",
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
        self.N = N
        if precision == 'single':
            # ffts in single precision (faster)
            dtype = np.float32
            self.FFT = PFFT(comm, [N,N], dtype=dtype, collapse=False, axes=(0,1), backend=backend)
            self.FFT_pad = PFFT(comm, [N,N], dtype=dtype, padding=[1.5,1.5], axes=(0,1), backend=backend)
        elif precision == 'double':
            # ffts in double precision
            dtype = np.float64
            self.FFT = PFFT(comm, [N,N], dtype=dtype, collapse=False, axes=(0,1), backend=backend)
            self.FFT_pad = PFFT(comm, [N,N], dtype=dtype, padding=[1.5,1.5], axes=(0,1), backend=backend)
        else:
            msg = "precision must be 'single' or 'double'"
            raise ValueError(msg)
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

        # distributed pv on grid.
        pv_dist = newDistArrayGrid(self.FFT) 
        for k in range(2):
            pv_dist[k,...] = pv[k][pv_dist.local_slice()]
        self.pvspec = fft_forward(self.FFT, pv_dist)

        self.pvbar = newDistArrayGrid(self.FFT) 
        for k in range(2):
            self.pvbar[k,...] = pvbar[k][self.pvbar.local_slice()]
        self.pvspec_eq = fft_forward(self.FFT, self.pvbar)

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

        self.k = k[self.pvspec.local_slice()]
        self.l = l[self.pvspec.local_slice()]

        self.ksqlsq = ksqlsq[self.pvspec.local_slice()]
        self.ik = (1.0j * self.k).astype(np.complex64)
        self.il = (1.0j * self.l).astype(np.complex64)

        mu = np.sqrt(self.ksqlsq) * np.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(np.finfo(mu.dtype).eps)  # clip to avoid NaN
        self.Hovermu = self.H / mu
        mu = mu.astype(np.float64)  # cast to avoid overflow in sinh
        self.tanhmu = np.tanh(mu).astype(dtype)  # cast back to original type
        self.sinhmu = np.sinh(mu).astype(dtype)
        self.diff_order = np.array(diff_order, dtype)  # hyperdiffusion order
        self.diff_efold = np.array(diff_efold, dtype)  # hyperdiff time scale
        ktot = np.sqrt(self.ksqlsq)
        ktotcutoff = np.array(pi * N / self.L, dtype)
        # integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff = np.exp(
            (-self.dt / self.diff_efold) * (ktot / ktotcutoff) ** self.diff_order
        )

    def invert(self,pvspec=None):
        # invert boundary pv to get streamfunction
        if pvspec is None:
            pvspec = self.pvspec
        psispec = np.empty_like(pvspec)
        psispec[0] = self.Hovermu * (
            (pvspec[1] / self.sinhmu) - (pvspec[0] / self.tanhmu)
        )
        psispec[1] = self.Hovermu * (
            (pvspec[1] / self.tanhmu) - (pvspec[0] / self.sinhmu)
        )
        return psispec

    def advance(self, timesteps=1, pv=None):
        # given pv on global grid, advance forward timesteps time steps.
        # if pv not specified, use pvspec instance variable.
        # return updated PV on global grid (on all ranks)
        if pv is not None:
            pv_dist = newDistArrayGrid(self.FFT) 
            for k in range(2):
                pv_dist[k,...] = pv[k][pv_dist.local_slice()]
            # distributed pv spectal coeffs.
            self.pvspec = fft_forward(self.FFT, pv_dist)
        for n in range(timesteps):
            self.timestep()
        return self.pv()

    def pv(self):
        # return global pv grid on all tasks
        pv = np.zeros((2,self.N,self.N),self.pvbar.dtype)
        pv_dist = fft_backward(self.FFT, self.pvspec)
        for k in range(2):
            pv[k][pv_dist.local_slice()] = pv_dist[k,...]
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE,pv,op=MPI.SUM)
        return pv

    def xyderiv(self, specarr):
        xderiv_spec = self.ik * specarr
        yderiv_spec = self.il * specarr
        xderiv = fft_backward(self.FFT_pad, xderiv_spec)
        yderiv = fft_backward(self.FFT_pad, yderiv_spec)
        return xderiv, yderiv

    def gettend(self,pvspec=None):
        if pvspec is None:
            pvspec = self.pvspec
        # compute tendencies of pv on z=0,H
        # invert pv to get streamfunction
        psispec = self.invert(pvspec)
        # nonlinear jacobian and thermal relaxation
        psix, psiy = self.xyderiv(psispec)
        pvx, pvy = self.xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = fft_forward(self.FFT_pad, jacobian)
        dpvspecdt = (1.0 / self.tdiab) * (self.pvspec_eq - pvspec) - jacobianspec + self.r[:,np.newaxis,np.newaxis] * self.ksqlsq * psispec
        dpvdt = fft_backward(self.FFT, dpvspecdt)
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
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()
    
    N = 96 # size of domain 
    dt = 900 # time step in seconds
    diff_efold = 86400./2. # hyperdiffusion dampling time scale on shortest wave
    norder = 8 # order of hyperdiffusion
    r = 0 # Ekman damping 
    nsq = 1.e-4; f=1.e-4; g = 9.8; theta0 = 300
    H = 10.e3 # lid height
    U = 16 # jet speed
    L = 20.e6
    # thermal relaxation time scale
    tdiab = 10.*86400 # in seconds
    # parameter used to scale PV to temperature units.
    scalefact = f*theta0/g
    
    # create initial pv
    if rank == 0:
        rs = np.random.RandomState(42) # fixed seed
        pv = rs.normal(0,100.,size=(2,N,N)).astype(np.float32)
        # add isolated blob on lid
        nexp = 20
        x = np.arange(0,2.*np.pi,2.*np.pi/N); y = np.arange(0.,2.*np.pi,2.*np.pi/N)
        x,y = np.meshgrid(x,y)
        pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
        # remove area mean from each level.
        for k in range(2):
            pv[k] = pv[k] - pv[k].mean()
    else:
        pv = np.zeros((2,N,N),dtype=np.float32)
    comm.Bcast(pv,root=0)

    # single or double precision
    precision='single' # pyfftw FFTs twice as fast as double

    # initialize qg model instance
    model = SQG(pv,nsq=nsq,f=f,U=U,H=H,r=r,tdiab=tdiab,dt=dt,
                diff_order=norder,diff_efold=diff_efold,
                precision=precision,tstart=0)
    
    outputinterval = 6.*3600. # interval between frames in seconds
    tmax = 300.*86400. # time to stop (in days)
    nsteps = int(tmax/outputinterval) # number of time steps to animate
    # set number of timesteps to integrate for each call to model.advance
    ntimesteps = int(outputinterval/model.dt)

    if num_processes > 1:
        while model.t < tmax:
            pv = model.advance(timesteps=ntimesteps)
            if rank==0:
                hr = model.t/3600.
                print(hr,scalefact*pv.min(),scalefact*pv.max())
    else:
        import matplotlib, os
        matplotlib.use('qtagg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        nout = 0 
        fig = plt.figure(figsize=(14,8))
        fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
        vmin = comm.reduce(scalefact*model.pvbar.min(),op=MPI.MIN)
        vmax = comm.reduce(scalefact*model.pvbar.max(),op=MPI.MAX)
        def initfig():
            global im1,im2
            ax1 = fig.add_subplot(121)
            ax1.axis('off')
            pv = model.advance(timesteps=0)
            im1 = ax1.imshow(scalefact*pv[0],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
            ax2 = fig.add_subplot(122)
            ax2.axis('off')
            im2 = ax2.imshow(scalefact*pv[1],cmap=plt.cm.jet,interpolation='nearest',origin='lower',vmin=vmin,vmax=vmax)
            return im1,im2,
        def updatefig(*args):
            global nout
            pv = model.advance(timesteps=ntimesteps)
            print(model.t/3600.,scalefact*pv.min(),scalefact*pv.max())
            im1.set_data(scalefact*pv[0])
            im2.set_data(scalefact*pv[1])
            return im1,im2,
        
        # interval=0 means draw as fast as possible
        ani = animation.FuncAnimation(fig, updatefig, frames=nsteps, repeat=False,\
              init_func=initfig,interval=0,blit=True)
        plt.show()
