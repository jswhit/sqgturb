import jax
import jax.numpy as jnp
from typing import NamedTuple

class SQGState(NamedTuple):
    """Immutable state for the SQG model time integration."""
    pvspec: jnp.ndarray  # spectral PV, shape (2, N, N//2+1)
    t: float             # current time (seconds)

class SQG:
    """
    Surface Quasi-Geostrophic (SQG) model implemented in JAX.

    All time-stepping functions are pure (no side effects). Evolving state
    (pvspec, t) is carried in an SQGState named tuple and returned explicitly
    rather than mutated in place.
    """

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
        tstart=0,
    ):
        # --- validation -------------------------------------------------------
        if pv.shape[0] != 2:
            raise ValueError("1st dim of pv should be 2")
        N = pv.shape[1]
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:
            raise ValueError("must specify time step")
        if diff_efold is None:
            raise ValueError("must specify efolding time scale for diffusion")

        # --- dtype ------------------------------------------------------------
        if precision == "single":
            dtype = jnp.float32
        elif precision == "double":
            dtype = jnp.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")

        self.N = N
        self.dtype = dtype

        # --- scalar parameters (stored as 0-d JAX arrays) ---------------------
        self.nsq    = jnp.array(nsq,    dtype)
        self.f      = jnp.array(f,      dtype)
        self.H      = jnp.array(H,      dtype)
        self.U      = jnp.array(U,      dtype)
        self.L      = jnp.array(L,      dtype)
        self.theta0 = jnp.array(theta0, dtype)
        self.g      = jnp.array(g,      dtype)
        self.dt     = jnp.array(dt,     dtype)
        self.tdiab  = jnp.array(tdiab,  dtype)

        # Ekman damping: r[0]=+r at z=0, r[1]=-r at z=H
        self.r = jnp.array([r, -r], dtype)

        # --- basic-state PV (relaxation target) --------------------------------
        pi     = jnp.array(jnp.pi, dtype)
        l_wave = 2.0 * pi / self.L
        mu_bar = l_wave * jnp.sqrt(self.nsq) * self.H / self.f

        y = jnp.arange(0, float(L), float(L) / N, dtype=dtype)  # (N,)
        pvbar_1d = (
            -(mu_bar * 0.5 * self.U / (l_wave * self.H))
            * jnp.cosh(0.5 * mu_bar)
            * jnp.cos(l_wave * y)
            / jnp.sinh(0.5 * mu_bar)
        )  # shape (N,)

        # broadcast to (2, N, N)
        pvbar = jnp.broadcast_to(pvbar_1d[jnp.newaxis, :, jnp.newaxis],
                                  (2, N, N)).copy()
        self.pvbar = pvbar
        self.pvspec_eq = jnp.fft.rfft2(pvbar)  # spectral relaxation target

        # --- spectral wavenumbers ---------------------------------------------
        k1d = (N * jnp.fft.fftfreq(N))[: N // 2 + 1]
        l1d =  N * jnp.fft.fftfreq(N)
        kk, ll = jnp.meshgrid(k1d, l1d)           # (N, N//2+1)
        k = kk.astype(dtype)
        l = ll.astype(dtype)

        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L

        self.k      = k
        self.l      = l
        self.ksqlsq = k ** 2 + l ** 2
        self.ik     = (1.0j * k).astype(jnp.complex64)
        self.il     = (1.0j * l).astype(jnp.complex64)
        self.wavenums = jnp.sqrt(kk ** 2 + ll ** 2)

        # --- padded wavenumbers (3/2 rule, dealiasing) ------------------------
        Npad   = 3 * N // 2
        k1d_p  = (Npad * jnp.fft.fftfreq(Npad))[: 3 * N // 4 + 1]
        l1d_p  =  Npad * jnp.fft.fftfreq(Npad)
        kk_p, ll_p = jnp.meshgrid(k1d_p, l1d_p)
        k_pad  = (2.0 * pi * kk_p / self.L).astype(dtype)
        l_pad  = (2.0 * pi * ll_p / self.L).astype(dtype)
        self.ik_pad = (1.0j * k_pad).astype(jnp.complex64)
        self.il_pad = (1.0j * l_pad).astype(jnp.complex64)

        # --- inversion helpers ------------------------------------------------
        mu = jnp.sqrt(self.ksqlsq) * jnp.sqrt(self.nsq) * self.H / self.f
        mu = jnp.clip(mu, jnp.finfo(dtype).eps)
        self.Hovermu = self.H / mu
        mu64 = mu.astype(jnp.float64)   # avoid overflow in sinh/tanh
        self.tanhmu = jnp.tanh(mu64).astype(dtype)
        self.sinhmu = jnp.sinh(mu64).astype(dtype)

        # --- hyper-diffusion --------------------------------------------------
        self.diff_order = jnp.array(diff_order, dtype)
        self.diff_efold = jnp.array(diff_efold, dtype)
        ktot        = jnp.sqrt(self.ksqlsq)
        ktotcutoff  = jnp.array(pi * N / self.L, dtype)
        self.hyperdiff = -(1.0 / self.diff_efold) * (ktot / ktotcutoff) ** self.diff_order

        # --- initial state (SQGState) -----------------------------------------
        self.initial_state = SQGState(
            pvspec=jnp.fft.rfft2(jnp.array(pv, dtype)),
            t=float(tstart),
        )

    # ------------------------------------------------------------------
    # Pure helper methods (no side effects)
    # ------------------------------------------------------------------

    def invert(self, pvspec: jnp.ndarray) -> jnp.ndarray:
        """Invert boundary PV to get the streamfunction (spectral)."""
        psispec_0 = self.Hovermu * (
            pvspec[1] / self.sinhmu - pvspec[0] / self.tanhmu
        )
        psispec_1 = self.Hovermu * (
            pvspec[1] / self.tanhmu - pvspec[0] / self.sinhmu
        )
        return jnp.stack([psispec_0, psispec_1], axis=0)

    def specpad(self, specarr: jnp.ndarray) -> jnp.ndarray:
        """
        Zero-pad spectral array for interpolation to the 3/2 grid
        (anti-aliasing).  Returns a new array; input is unchanged.
        """
        N     = self.N
        Npad  = 3 * N // 2
        Nhalf = 3 * N // 4 + 1

        specarr_pad = jnp.zeros((2, Npad, Nhalf), dtype=specarr.dtype)

        # lower-left block
        specarr_pad = specarr_pad.at[:, :N // 2, :N // 2].set(
            2.25 * specarr[:, :N // 2, :N // 2]
        )
        # upper-left block
        specarr_pad = specarr_pad.at[:, -N // 2 :, :N // 2].set(
            2.25 * specarr[:, -N // 2 :, :N // 2]
        )
        # negative Nyquist frequency
        specarr_pad = specarr_pad.at[:, :N // 2, N // 2].set(
            jnp.conjugate(2.25 * specarr[:, :N // 2, -1])
        )
        specarr_pad = specarr_pad.at[:, -N // 2 :, N // 2].set(
            jnp.conjugate(2.25 * specarr[:, -N // 2 :, -1])
        )
        return specarr_pad

    def spectrunc(self, specarr: jnp.ndarray) -> jnp.ndarray:
        """Truncate spectral array using the 2/3 de-aliasing rule."""
        N = self.N
        specarr_trunc = jnp.zeros((2, N, N // 2 + 1), dtype=specarr.dtype)
        specarr_trunc = specarr_trunc.at[:, :N // 2, :N // 2].set(
            specarr[:, :N // 2, :N // 2]
        )
        specarr_trunc = specarr_trunc.at[:, -N // 2 :, :N // 2].set(
            specarr[:, -N // 2 :, :N // 2]
        )
        return specarr_trunc

    def xyderiv(self, specarr: jnp.ndarray):
        """
        Compute physical-space x- and y-derivatives via padded FFT.
        Returns (xderiv, yderiv), both on the 3/2 grid.
        """
        specarr_pad = self.specpad(specarr)
        xderiv = jnp.fft.irfft2(self.ik_pad * specarr_pad)
        yderiv = jnp.fft.irfft2(self.il_pad * specarr_pad)
        return xderiv, yderiv

    def gettend(self, pvspec: jnp.ndarray) -> jnp.ndarray:
        """
        Compute PV tendency (spectral) from current spectral PV.
        Pure function — returns tendency array, does not mutate anything.
        """
        psispec = self.invert(pvspec)

        psix, psiy = self.xyderiv(psispec)
        pvx,  pvy  = self.xyderiv(pvspec)

        jacobian     = psix * pvy - psiy * pvx
        jacobianspec = self.spectrunc(jnp.fft.rfft2(jacobian))

        dpvspecdt = (
            (1.0 / self.tdiab) * (self.pvspec_eq - pvspec)
            - jacobianspec
            + self.r[:, jnp.newaxis, jnp.newaxis] * self.ksqlsq * psispec
            + self.hyperdiff[jnp.newaxis, ...] * pvspec
        )
        return dpvspecdt

    def timestep(self, state: SQGState) -> SQGState:
        """
        Advance state by one time step using 4th-order Runge-Kutta.
        Pure function — returns a new SQGState.
        """
        pvspec = state.pvspec
        dt     = self.dt

        k1 = self.gettend(pvspec)
        k2 = self.gettend(pvspec + 0.5 * dt * k1)
        k3 = self.gettend(pvspec + 0.5 * dt * k2)
        k4 = self.gettend(pvspec + dt * k3)

        new_pvspec = pvspec + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        new_t      = state.t + float(dt)

        return SQGState(pvspec=new_pvspec, t=new_t)

    def advance(self, state: SQGState, timesteps: int = 1, pv=None) -> jnp.ndarray:
        """
        Advance the model forward by `timesteps` steps.

        Parameters
        ----------
        state      : SQGState — current model state
        timesteps  : number of RK4 steps to take
        pv         : optional physical-space PV array to restart from

        Returns
        -------
        pv_out - the physical-space PV after integration.
        """
        if pv is not None:
            state = SQGState(
                pvspec=jnp.fft.rfft2(jnp.array(pv, self.dtype)),
                t=state.t,
            )

        for _ in range(timesteps):
            state = self.timestep(state)

        return jnp.fft.irfft2(state.pvspec)
