import numpy as np
from .pyfft import Fouriert

class SQG:
    def __init__(
        self,
        ft,
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
        symmetric=True,
        dt=None,
        threads=1,
        precision="single",
        tstart=0,
    ):
        # initialize SQG model.
        if pv.shape[0] != 2:
            raise ValueError("1st dim of pv should be 2")
        Nt = pv.shape[1]  # number of grid points in each direction
        # Nt should be even
        if Nt % 2:
            raise ValueError("grid must be even (powers of 2 are fastest)")
        if dt is None:  # time step must be specified
            raise ValueError("must specify time step")
        if diff_efold is None:  # efolding time scale for diffusion must be specified
            raise ValueError("must specify efolding time scale for diffusion")
        # number of openmp threads to use for FFTs (only for pyfftw)
        self.threads = threads
        self.N = 2*Nt//3
        self.Nt = Nt
        self.ft = ft
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
        self.U = np.array(U, dtype)  # basic state velocity at z = H
        self.L = np.array(L, dtype)  # size of square domain.
        self.dt = np.array(dt, dtype)  # time step (seconds)
        if r < 1.0e-10:
            self.ekman = False
        else:
            self.ekman = True
        self.r = np.array(r, dtype)  # Ekman damping (at z=0)
        self.tdiab = np.array(tdiab, dtype)  # thermal relaxation damping.
        self.t = tstart  # initialize time counter
        # setup basic state pv (for thermal relaxation)
        self.symmetric = symmetric  # symmetric jet, or jet with U=0 at sfc.
        y = np.arange(0, self.L, self.L / self.Nt, dtype=dtype)
        pvbar = np.zeros((2, self.Nt), dtype)
        pi = np.array(np.pi, dtype)
        l = 2.0 * pi / self.L
        mu = l * np.sqrt(nsq) * H / f
        if symmetric:
            # symmetric version, no difference between upper and lower
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
        else:
            # asymmetric version, equilibrium state has no flow at surface and
            # temp gradient slightly weaker at sfc.
            # u = U*np.sin(l*y)*np.sinh(mu*z/H)*np.sin(l*y)/np.sinh(mu)
            # theta = (f*theta0/g)*(U*mu/(l*H))*np.cosh(mu*z/H)*
            # np.cos(l*y)/np.sinh(mu)
            # + theta0 + (theta0*nsq*z/g)
            pvbar[:] = -(mu * U / (l * H)) * np.cos(l * y) / np.sinh(mu)
            pvbar[1, :] = pvbar[0, :] * np.cosh(mu)
        pvbar.shape = (2, self.Nt, 1)
        pvbar = pvbar * np.ones((2, self.Nt, self.Nt), dtype)
        self.pvbar = pvbar
        self.pvspec_eq = self.ft.grdtospec(pvbar)  # state to relax to with timescale tdiab
        self.pvspec = self.ft.grdtospec(pv)  # initial pv field (spectral)
        mu = np.sqrt(self.ft.ksqlsq) * np.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(np.finfo(mu.dtype).eps)  # clip to avoid NaN
        self.Hovermu = self.H / mu
        mu = mu.astype(np.float64)  # cast to avoid overflow in sinh
        self.tanhmu = np.tanh(mu).astype(dtype)  # cast back to original type
        self.sinhmu = np.sinh(mu).astype(dtype)
        self.diff_order = np.array(diff_order, dtype)  # hyperdiffusion order
        self.diff_efold = np.array(diff_efold, dtype)  # hyperdiff time scale
        ktot = np.sqrt(self.ft.ksqlsq)
        ktotcutoff = np.array(pi * self.N / self.L, dtype)
        # integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff = np.exp(
            (-self.dt / self.diff_efold) * (ktot / ktotcutoff) ** self.diff_order
        )
        # number of timesteps to advance each call to 'advance' method.
        self.timesteps = 1

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

    def invert_inverse(self, psispec=None):
        if psispec is None:
            psispec = self.invert(self.pvspec)
        # given streamfunction, return PV
        pvspec = np.empty((2, self.N, self.N // 2 + 1), dtype=psispec.dtype)
        alpha = self.Hovermu
        th = self.tanhmu
        sh = self.sinhmu
        tmp1 = 1.0 / sh ** 2 - 1.0 / th ** 2
        tmp1[0, 0] = 1.0
        pvspec[0] = ((psispec[0] / th) - (psispec[1] / sh)) / (alpha * tmp1)
        pvspec[1] = ((psispec[0] / sh) - (psispec[1] / th)) / (alpha * tmp1)
        pvspec[:, 0, 0] = 0.0  # area mean PV not determined by streamfunction
        return pvspec

    def advance(self, pv=None):
        # given total pv on grid, advance forward
        # number of timesteps given by 'timesteps' instance var.
        # if pv not specified, use pvspec instance variable.
        if pv is not None:
            self.pvspec = self.ft.grdtospec(pv)
        for n in range(self.timesteps):
            self.timestep()
        return self.ft.spectogrd(self.pvspec)

    def gettend(self, pvspec=None):
        # compute tendencies of pv on z=0,H
        # invert pv to get streamfunction
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        # nonlinear jacobian and thermal relaxation
        psix, psiy = self.ft.getgrad(psispec)
        pvx, pvy = self.ft.getgrad(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = self.ft.grdtospec(jacobian)
        dpvspecdt = (1.0 / self.tdiab) * (self.pvspec_eq - pvspec) - jacobianspec
        # Ekman damping at boundaries.
        if self.ekman:
            dpvspecdt[0] += self.r * self.ft.ksqlsq * psispec[0]
            # for asymmetric jet (U=0 at sfc), no Ekman layer at lid
            if self.symmetric:
                dpvspecdt[1] -= self.r * self.ft.ksqlsq * psispec[1]
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
