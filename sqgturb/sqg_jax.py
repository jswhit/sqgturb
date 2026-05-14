"""
sqg_jax.py — Surface Quasi-Geostrophic model in JAX (optimised).
 
Key speed improvements over the naive translation:
  1. @jax.jit on gettend / timestep  → single XLA kernel, no Python overhead.
  2. jax.lax.scan in advance         → compiled loop, no per-step re-tracing.
  3. Fused irfft2 for x/y derivs    → one irfft2 on a stacked (2,…) array
                                       instead of two sequential calls.
  4. spectrunc via slicing           → no scatter (zeros + at.set).
  5. Pre-broadcast `r` at init       → avoids repeated broadcasting in gettend.
  6. Pre-fused linear operator       → hyperdiff + Newtonian relaxation combined
                                       into `linop` so each RK4 stage does one
                                       multiply instead of two separate terms.
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple
 
class SQGState(NamedTuple):
    """Immutable state carried through time integration."""
    pvspec: jnp.ndarray   # spectral PV, shape (2, N, N//2+1)
    t: float              # current model time (seconds)
 
class SQG:
    """
    Optimised JAX implementation of the SQG model.
 
    All time-stepping methods are pure functions (no side effects).
    The full RK4 loop is JIT-compiled and runs as a single XLA program.
    """
 
    def __init__(
        self,
        N,
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
        precision="single"
    ):
        # ── validation ────────────────────────────────────────────────────────
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:
            raise ValueError("must specify time step")
        if diff_efold is None:
            raise ValueError("must specify efolding time scale for diffusion")
 
        # ── dtype ─────────────────────────────────────────────────────────────
        if precision == "single":
            dtype = jnp.float32
        elif precision == "double":
            dtype = jnp.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")
 
        self.N     = N
        self.dtype = dtype
 
        # ── scalar parameters ─────────────────────────────────────────────────
        nsq_   = jnp.array(nsq,   dtype)
        f_     = jnp.array(f,     dtype)
        H_     = jnp.array(H,     dtype)
        U_     = jnp.array(U,     dtype)
        L_     = jnp.array(L,     dtype)
        dt_    = jnp.array(dt,    dtype)
        tdiab_ = jnp.array(tdiab, dtype)
        self.nsq = nsq_
        self.f = f_
        self.H = H_
        self.U = U_
        self.L = L_
        self.dt    = dt_
        self.tdiab = tdiab_
 
        # ── basic-state PV (Newtonian relaxation target) ──────────────────────
        pi     = jnp.array(jnp.pi, dtype)
        l_wave = 2.0 * pi / L_
        mu_bar = l_wave * jnp.sqrt(nsq_) * H_ / f_
 
        y = jnp.arange(0, float(L), float(L) / N, dtype=dtype)
        pvbar_1d = (
            -(mu_bar * 0.5 * U_ / (l_wave * H_))
            * jnp.cosh(0.5 * mu_bar)
            * jnp.cos(l_wave * y)
            / jnp.sinh(0.5 * mu_bar)
        )
        pvbar          = jnp.broadcast_to(pvbar_1d[None, :, None], (2, N, N))
        self.pvspec_eq = jnp.fft.rfft2(pvbar)   # (2, N, N//2+1) — fixed target
 
        # ── spectral wavenumbers ──────────────────────────────────────────────
        k1d = (N * jnp.fft.fftfreq(N))[: N // 2 + 1]
        l1d =  N * jnp.fft.fftfreq(N)
        kk, ll = jnp.meshgrid(k1d, l1d)          # (N, N//2+1)
        k = (2.0 * pi * kk / L_).astype(dtype)
        l = (2.0 * pi * ll / L_).astype(dtype)
 
        ksqlsq = k ** 2 + l ** 2
        self.ksqlsq = ksqlsq
 
        # ── padded wavenumbers (3/2 dealiasing grid) ──────────────────────────
        Npad  = 3 * N // 2
        k1d_p = (Npad * jnp.fft.fftfreq(Npad))[: 3 * N // 4 + 1]
        l1d_p =  Npad * jnp.fft.fftfreq(Npad)
        kk_p, ll_p = jnp.meshgrid(k1d_p, l1d_p)
        k_pad = (2.0 * pi * kk_p / L_).astype(dtype)
        l_pad = (2.0 * pi * ll_p / L_).astype(dtype)
 
        # Stack ik_pad / il_pad → (2, Npad, 3N/4+1) complex.
        # Used to compute both derivatives with a single irfft2 call.
        cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        self.ikl_pad = jnp.stack(
            [(1j * k_pad).astype(cdtype), (1j * l_pad).astype(cdtype)], axis=0
        )
 
        # ── inversion helpers ─────────────────────────────────────────────────
        mu = jnp.sqrt(ksqlsq) * jnp.sqrt(nsq_) * H_ / f_
        mu = jnp.clip(mu, jnp.finfo(dtype).eps)
        self.Hovermu = H_ / mu
        mu64         = mu.astype(jnp.float64)
        self.tanhmu  = jnp.tanh(mu64).astype(dtype)
        self.sinhmu  = jnp.sinh(mu64).astype(dtype)
 
        # ── Ekman damping: pre-broadcast to (2, N, N//2+1) ───────────────────
        # Avoids re-broadcasting r inside every gettend call.
        r_vec       = jnp.array([r, -r], dtype)
        self.r = r_vec
        self.r_bc   = r_vec[:, None, None] * ksqlsq[None, ...]  # (2, N, N//2+1)
 
        # ── Pre-fused linear operator ─────────────────────────────────────────
        # Combines hyperdiffusion and the -pvspec/tdiab part of relaxation:
        #   linop * pvspec  =  hyperdiff*pvspec  -  pvspec/tdiab
        # The +pvspec_eq/tdiab term is added separately (it doesn't involve pvspec).
        ktot       = jnp.sqrt(ksqlsq)
        ktotcutoff = jnp.array(pi * N / L_, dtype)
        hyperdiff  = -(1.0 / jnp.array(diff_efold, dtype)) * (ktot / ktotcutoff) ** jnp.array(diff_order, dtype)
        self.diff_efold = diff_efold
        self.diff_order = diff_order
        self.linop = hyperdiff - (1.0 / tdiab_)   # (N, N//2+1)
 
        # Pre-compute the constant forcing term:  pvspec_eq / tdiab
        self.relax_forcing = self.pvspec_eq / tdiab_  # (2, N, N//2+1)
 
        self.N2   = N // 2
        self.Npad = Npad
 
        # ── JIT-compile the hot path once at construction time ────────────────
        self.gettend  = jax.jit(self._gettend)
        self.timestep = jax.jit(self._timestep)
 
    # ── pure spectral helpers ─────────────────────────────────────────────────
 
    def _invert(self, pvspec: jnp.ndarray) -> jnp.ndarray:
        """Invert boundary PV → streamfunction (spectral). Pure."""
        psi0 = self.Hovermu * (pvspec[1] / self.sinhmu - pvspec[0] / self.tanhmu)
        psi1 = self.Hovermu * (pvspec[1] / self.tanhmu - pvspec[0] / self.sinhmu)
        return jnp.stack([psi0, psi1], axis=0)
 
    def _specpad(self, specarr: jnp.ndarray) -> jnp.ndarray:
        """
        Zero-pad spectral coefficients onto the 3/2 dealiasing grid.
        Input shape: (2, N, N//2+1) → output: (2, 3N/2, 3N/4+1).  Pure.
        """
        N, N2, Npad = self.N, self.N2, self.Npad
        Nhalf = 3 * N // 4 + 1
 
        s   = 2.25 * specarr
        pad = jnp.zeros((2, Npad, Nhalf), dtype=specarr.dtype)
        pad = pad.at[:, :N2,  :N2].set(s[:, :N2,  :N2])
        pad = pad.at[:, -N2:, :N2].set(s[:, -N2:, :N2])
        pad = pad.at[:, :N2,  N2 ].set(jnp.conjugate(s[:, :N2,  -1]))
        pad = pad.at[:, -N2:, N2 ].set(jnp.conjugate(s[:, -N2:, -1]))
        return pad
 
    def _spectrunc(self, specarr: jnp.ndarray) -> jnp.ndarray:
        """
        Truncate padded spectral array back to (2, N, N//2+1).
 
        Optimisation: pure slicing instead of zeros + scatter (.at.set).
        Slices map directly to XLA DynamicSlice ops — no write-back.
        """
        N, N2 = self.N, self.N2
        top    = specarr[:, :N2,  :N2]                    # (2, N/2, N/2)
        bottom = specarr[:, -N2:, :N2]                    # (2, N/2, N/2)
        block  = jnp.concatenate([top, bottom], axis=1)   # (2, N,   N/2)
        # Append a zero Nyquist column → (2, N, N//2+1)
        zero_col = jnp.zeros((*block.shape[:2], 1), dtype=specarr.dtype)
        return jnp.concatenate([block, zero_col], axis=2)
 
    def _xyderiv(self, specarr: jnp.ndarray):
        """
        Compute x- and y-derivatives on the dealiased 3/2 grid.
 
        Optimisation: pad once, then broadcast-multiply with the stacked
        [ik, il] tensor and run a *single* irfft2 over all 4 fields at once
        (2 boundaries × 2 directions).  Halves FFT calls vs two separate
        irfft2s, and lets XLA batch them efficiently.
 
        Returns
        -------
        xderiv, yderiv — each shape (2, Npad, Npad)
        """
        N2pad  = 3 * self.N // 4 + 1
        s_pad  = self._specpad(specarr)                    # (2, Npad, N2pad)
 
        # ikl_pad: (2, Npad, N2pad) — axis-0 indexes [x-dir, y-dir]
        # s_pad:   (2, Npad, N2pad) — axis-0 indexes [boundary 0, boundary 1]
        # product: (2, 2, Npad, N2pad) → [deriv, boundary, y, x]
        prod    = self.ikl_pad[:, None, :, :] * s_pad[None, :, :, :]
        merged  = prod.reshape(4, self.Npad, N2pad)
        phys    = jnp.fft.irfft2(merged)                  # (4, Npad, Npad)
        phys    = phys.reshape(2, 2, self.Npad, self.Npad)
        return phys[0], phys[1]                            # xderiv, yderiv
 
    def _gettend(self, pvspec: jnp.ndarray) -> jnp.ndarray:
        """
        Compute spectral PV tendency.  Pure; JIT-compiled via self.gettend.
        """
        psispec = self._invert(pvspec)
 
        psix, psiy = self._xyderiv(psispec)
        pvx,  pvy  = self._xyderiv(pvspec)
 
        jacobian     = psix * pvy - psiy * pvx
        jacobianspec = self._spectrunc(jnp.fft.rfft2(jacobian))
 
        # Pre-fused linear term:  linop*pvspec + relax_forcing
        #   where linop = hyperdiff - 1/tdiab  (avoids two separate array ops)
        linear = self.linop[None, ...] * pvspec + self.relax_forcing
 
        return linear - jacobianspec + self.r_bc * psispec
 
    def _timestep(self, state: SQGState) -> SQGState:
        """One RK4 step. Pure; JIT-compiled via self.timestep."""
        pvspec = state.pvspec
        dt     = self.dt
        gt     = self._gettend
 
        k1 = gt(pvspec)
        k2 = gt(pvspec + 0.5 * dt * k1)
        k3 = gt(pvspec + 0.5 * dt * k2)
        k4 = gt(pvspec + dt * k3)
 
        new_pvspec = pvspec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return SQGState(pvspec=new_pvspec, t=state.t + float(dt))
 
    # ── public API ────────────────────────────────────────────────────────────
 
    def advance(self, state: SQGState, timesteps: int = 1, pv=None) -> tuple:
        """
        Advance the model forward by `timesteps` RK4 steps.
 
        Uses jax.lax.scan so the entire loop is compiled into a single XLA
        program with no Python interpreter overhead between steps.
 
        Parameters
        ----------
        state      : SQGState  — current model state
        timesteps  : int       — number of RK4 steps to take
        pv         : ndarray   — optional restart from physical-space PV
 
        Returns
        -------
        (new_state, pv_out) : (SQGState, jnp.ndarray)
        """
        if pv is not None:
            state = SQGState(
                pvspec=jnp.fft.rfft2(jnp.array(pv, self.dtype)),
                t=state.t,
            )
 
        def scan_body(carry: SQGState, _):
            return self._timestep(carry), None
 
        # Entire loop compiled once by XLA — no Python overhead per step
        final_state, _ = jax.lax.scan(scan_body, state, None, length=timesteps)
 
        pv_out = jnp.fft.irfft2(final_state.pvspec)
        return final_state, pv_out
