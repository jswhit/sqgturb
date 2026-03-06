# sqg.pyx
# -----------------------------------------------------------------------
#  Cython wrapper for the C SQG turbulence model (sqg.c / sqg.h).
#
#  Exposes a single Python class ``SQG`` whose interface deliberately
#  mirrors the original sqg.py so that existing scripts need only swap
#  the import line:
#
#      # before
#      from sqg import SQG
#      # after
#      from sqg_cy import SQG          # or whatever the installed name is
#
#  All heavy work is delegated to the C library.  NumPy arrays with
#  dtype=float32 and C-contiguous layout are used throughout to avoid
#  copies on the boundary between Python and C.
#
#  Build with setup.py (see below) or directly:
#      cython --3str sqg.pyx
#      gcc -O3 -shared -fPIC $(python3-config --includes) \
#          sqg.c sqg_cy.c -lfftw3f -lm -o sqg_cy$(python3-config --extension-suffix)
# -----------------------------------------------------------------------

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
cimport numpy as cnp
import  numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from sqg_c cimport (
    SQG as CSQG,
    real_t, cplx_t,
    sqg_create, sqg_destroy,
    sqg_invert, sqg_specpad, sqg_spectrunc,
    sqg_xyderiv, sqg_gettend,
    sqg_timestep, sqg_advance,
    sqg_exec_rfft2, sqg_exec_irfft2, sqg_exec_rfft2_pad,
)

cnp.import_array()

# -----------------------------------------------------------------------
#  Small private helpers
# -----------------------------------------------------------------------

cdef inline object _f32c(arr):
    """Return arr as a C-contiguous float32 NumPy array (copy only if needed)."""
    return np.ascontiguousarray(arr, dtype=np.float32)

cdef inline void _check_model(CSQG *s) except *:
    if s == NULL:
        raise RuntimeError("SQG model is not initialised (sqg_create returned NULL)")


# -----------------------------------------------------------------------
#  SQG Python class
# -----------------------------------------------------------------------

cdef class SQG:
    """
    Surface Quasi-Geostrophic turbulence model.

    Wraps the C implementation in sqg.c / sqg.h.

    Parameters
    ----------
    pv : array_like, shape (2, N, N), dtype float32
        Initial potential vorticity field.
    f : float, optional
        Coriolis parameter (default 1e-4).
    nsq : float, optional
        Brunt-Väisälä frequency squared (default 1e-4).
    L : float, optional
        Domain size in metres (default 20e6).
    H : float, optional
        Lid height in metres (default 10e3).
    U : float, optional
        Basic-state jet speed m/s (default 30).
    r : float, optional
        Ekman damping coefficient (default 0).
    tdiab : float, optional
        Thermal relaxation time scale in seconds (default 10*86400).
    diff_order : int, optional
        Hyperdiffusion order (default 8).
    diff_efold : float
        Hyperdiffusion e-folding time for the shortest wave (seconds).
        **Must** be specified (> 0).
    theta0 : float, optional
        Reference potential temperature K (default 300).
    g : float, optional
        Gravity m/s² (default 9.8).
    dt : float
        Time step in seconds.  **Must** be specified (!= 0).
    tstart : float, optional
        Initial model time in seconds (default 0).
    """

    # Hold the opaque C pointer.  Declared as void* in Cython so that the
    # .pxd SQG typedef is only needed when we actually dereference it.
    cdef CSQG *_c

    # Cache N so Python code can read it without touching the C struct.
    cdef readonly int N

    # ------------------------------------------------------------------
    def __cinit__(self,
                  object pv,
                  float  f          = 1.0e-4,
                  float  nsq        = 1.0e-4,
                  float  L          = 20.0e6,
                  float  H          = 10.0e3,
                  float  U          = 30.0,
                  float  r          = 0.0,
                  float  tdiab      = 10.0 * 86400.0,
                  int    diff_order = 8,
                  float  diff_efold = -1.0,
                  float  theta0     = 300.0,
                  float  g          = 9.8,
                  float  dt         = 0.0,
                  double tstart     = 0.0):

        self._c = NULL

        # ---- validate & prepare initial PV ----------------------
        pv_arr = _f32c(pv)
        if pv_arr.ndim != 3 or pv_arr.shape[0] != 2:
            raise ValueError("pv must have shape (2, N, N)")
        cdef int N_in = pv_arr.shape[1]
        if pv_arr.shape[2] != N_in:
            raise ValueError("pv must have shape (2, N, N) with equal N")
        if N_in % 2 != 0:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt == 0.0:
            raise ValueError("must specify time step dt")
        if diff_efold <= 0.0:
            raise ValueError("must specify diff_efold > 0")

        # Flatten to 1-D: C expects a flat (2*N*N) row-major array
        cdef cnp.ndarray[real_t, ndim=1, mode='c'] flat = pv_arr.reshape(-1)

        cdef const real_t *pv_ptr = <const real_t *>flat.data

        with nogil:
            self._c = sqg_create(pv_ptr, N_in,
                                 f, nsq, L, H, U, r, tdiab,
                                 diff_order, diff_efold,
                                 theta0, g, dt, tstart)

        if self._c == NULL:
            raise RuntimeError("sqg_create failed; check stderr for details")

        self.N = self._c.N

    # ------------------------------------------------------------------
    def __dealloc__(self):
        if self._c != NULL:
            sqg_destroy(self._c)
            self._c = NULL

    # ==================================================================
    #  Read-only properties (mirror sqg.py attributes)
    # ==================================================================

    @property
    def t(self):
        """Current model time (seconds, float64)."""
        _check_model(self._c)
        return self._c.t

    @property
    def f(self):
        _check_model(self._c)
        return self._c.f

    @property
    def nsq(self):
        _check_model(self._c)
        return self._c.nsq

    @property
    def L(self):
        _check_model(self._c)
        return self._c.L

    @property
    def H(self):
        _check_model(self._c)
        return self._c.H

    @property
    def U(self):
        _check_model(self._c)
        return self._c.U

    @property
    def dt(self):
        _check_model(self._c)
        return self._c.dt

    @property
    def tdiab(self):
        _check_model(self._c)
        return self._c.tdiab

    @property
    def diff_efold(self):
        _check_model(self._c)
        return self._c.diff_efold

    @property
    def diff_order(self):
        _check_model(self._c)
        return self._c.diff_order

    @property
    def r(self):
        """Ekman damping array [r_bottom, -r_lid], shape (2,), float32."""
        _check_model(self._c)
        out = np.empty(2, dtype=np.float32)
        out[0] = self._c.r[0]
        out[1] = self._c.r[1]
        return out

    # ------------------------------------------------------------------
    #  Spectral operator arrays (returned as 2-D NumPy views, shape N×Nc)
    # ------------------------------------------------------------------

    cdef _make_real_view(self, real_t *ptr, int length):
        """Wrap a C real_t* as a read-only 1-D float32 NumPy array."""
        cdef cnp.npy_intp dims[1]
        dims[0] = length
        arr = cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_FLOAT, ptr)
        # Mark read-only so the user cannot accidentally corrupt C memory
        arr.flags.writeable = False
        return arr

    @property
    def ksqlsq(self):
        """k²+l² spectral array, shape (N, Nc), float32."""
        _check_model(self._c)
        return self._make_real_view(self._c.ksqlsq,
                                    self._c.N * self._c.Nc
                                    ).reshape(self._c.N, self._c.Nc)

    @property
    def wavenums(self):
        """Total wavenumber array, shape (N, Nc), float32."""
        _check_model(self._c)
        return self._make_real_view(self._c.wavenums,
                                    self._c.N * self._c.Nc
                                    ).reshape(self._c.N, self._c.Nc)

    @property
    def hyperdiff(self):
        """Hyperdiffusion coefficient array, shape (N, Nc), float32."""
        _check_model(self._c)
        return self._make_real_view(self._c.hyperdiff,
                                    self._c.N * self._c.Nc
                                    ).reshape(self._c.N, self._c.Nc)

    @property
    def pvbar(self):
        """Basic-state PV, shape (2, N, N), float32."""
        _check_model(self._c)
        return self._make_real_view(self._c.pvbar,
                                    2 * self._c.N * self._c.N
                                    ).reshape(2, self._c.N, self._c.N)

    @property
    def pvspec(self):
        """
        Current spectral PV as a complex64 NumPy array, shape (2, N, Nc).

        The data lives in the C struct; this is a read-only view.
        Use set_pvspec() to modify it safely.
        """
        _check_model(self._c)
        cdef cnp.npy_intp dims[1]
        dims[0] = 2 * self._c.N * self._c.Nc
        # cplx_t is float[2] — same memory layout as np.complex64
        arr = cnp.PyArray_SimpleNewFromData(
            1, dims, cnp.NPY_COMPLEX64,
            <void *>self._c.pvspec)
        arr.flags.writeable = False
        return arr.reshape(2, self._c.N, self._c.Nc)

    # ==================================================================
    #  Core API  (mirror sqg.py method signatures)
    # ==================================================================

    def advance(self, int timesteps=1, pv=None):
        """
        Advance the model by *timesteps* RK4 steps.

        Parameters
        ----------
        timesteps : int
            Number of time steps to take.
        pv : array_like (2, N, N), float32, optional
            If given, reset the spectral state to rfft2(pv) first.

        Returns
        -------
        pv_out : ndarray, shape (2, N, N), float32
            Physical PV after the time steps.
        """
        _check_model(self._c)

        cdef int N = self._c.N

        # Output array
        cdef cnp.ndarray[real_t, ndim=1, mode='c'] pv_out = \
            np.empty(2 * N * N, dtype=np.float32)

        cdef const real_t *pv_in_ptr = NULL
        cdef cnp.ndarray[real_t, ndim=1, mode='c'] pv_in_flat

        if pv is not None:
            pv_in_flat = _f32c(pv).reshape(-1)
            if pv_in_flat.size != 2 * N * N:
                raise ValueError(f"pv must have 2*N*N={2*N*N} elements")
            pv_in_ptr = <const real_t *>pv_in_flat.data

        with nogil:
            sqg_advance(self._c, timesteps, pv_in_ptr,
                        <real_t *>pv_out.data)

        return pv_out.reshape(2, N, N)

    # ------------------------------------------------------------------
    def timestep(self):
        """Take a single 4th-order Runge-Kutta time step."""
        _check_model(self._c)
        with nogil:
            sqg_timestep(self._c)

    # ------------------------------------------------------------------
    def invert(self, pvspec_in=None):
        """
        Invert boundary PV to obtain the streamfunction (spectral).

        Parameters
        ----------
        pvspec_in : array_like (2, N, Nc), complex64, optional
            Spectral PV to invert.  Uses the model's current pvspec
            if not supplied.

        Returns
        -------
        psispec : ndarray, shape (2, N, Nc), complex64
        """
        _check_model(self._c)
        cdef int N  = self._c.N
        cdef int Nc = self._c.Nc

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] psispec = \
            np.empty(2 * N * Nc, dtype=np.complex64)

        cdef const cplx_t *in_ptr = NULL
        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] pv_flat

        if pvspec_in is not None:
            pv_flat = np.ascontiguousarray(pvspec_in,
                                           dtype=np.complex64).reshape(-1)
            if pv_flat.size != 2 * N * Nc:
                raise ValueError(f"pvspec_in must have shape (2, N, Nc)")
            in_ptr = <const cplx_t *>pv_flat.data

        with nogil:
            sqg_invert(self._c, in_ptr,
                       <cplx_t *>psispec.data)

        return psispec.reshape(2, N, Nc)

    # ------------------------------------------------------------------
    def specpad(self, specarr):
        """
        Zero-pad spectral array to the 3/2 dealiasing grid (×2.25 scale).

        Parameters
        ----------
        specarr : array_like, shape (2, N, Nc), complex64

        Returns
        -------
        out : ndarray, shape (2, N_pad, Nc_pad), complex64
        """
        _check_model(self._c)
        cdef int N      = self._c.N
        cdef int Nc     = self._c.Nc
        cdef int N_pad  = self._c.N_pad
        cdef int Nc_pad = self._c.Nc_pad

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] inp = \
            np.ascontiguousarray(specarr, dtype=np.complex64).reshape(-1)
        if inp.size != 2 * N * Nc:
            raise ValueError("specarr must have shape (2, N, Nc)")

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] out = \
            np.zeros(2 * N_pad * Nc_pad, dtype=np.complex64)

        with nogil:
            sqg_specpad(self._c,
                        <const cplx_t *>inp.data,
                        <cplx_t *>out.data)

        return out.reshape(2, N_pad, Nc_pad)

    # ------------------------------------------------------------------
    def spectrunc(self, specarr):
        """
        Truncate a padded spectral array back to the regular N grid.

        Parameters
        ----------
        specarr : array_like, shape (2, N_pad, Nc_pad), complex64

        Returns
        -------
        out : ndarray, shape (2, N, Nc), complex64
        """
        _check_model(self._c)
        cdef int N      = self._c.N
        cdef int Nc     = self._c.Nc
        cdef int N_pad  = self._c.N_pad
        cdef int Nc_pad = self._c.Nc_pad

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] inp = \
            np.ascontiguousarray(specarr, dtype=np.complex64).reshape(-1)
        if inp.size != 2 * N_pad * Nc_pad:
            raise ValueError("specarr must have shape (2, N_pad, Nc_pad)")

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] out = \
            np.zeros(2 * N * Nc, dtype=np.complex64)

        with nogil:
            sqg_spectrunc(self._c,
                          <const cplx_t *>inp.data,
                          <cplx_t *>out.data)

        return out.reshape(2, N, Nc)

    # ------------------------------------------------------------------
    def xyderiv(self, specarr):
        """
        Compute x- and y-derivatives on the dealiased padded grid.

        Parameters
        ----------
        specarr : array_like, shape (2, N, Nc), complex64

        Returns
        -------
        xderiv : ndarray, shape (2, N_pad, N_pad), float32
        yderiv : ndarray, shape (2, N_pad, N_pad), float32
        """
        _check_model(self._c)
        cdef int N      = self._c.N
        cdef int Nc     = self._c.Nc
        cdef int N_pad  = self._c.N_pad

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] inp = \
            np.ascontiguousarray(specarr, dtype=np.complex64).reshape(-1)
        if inp.size != 2 * N * Nc:
            raise ValueError("specarr must have shape (2, N, Nc)")

        cdef cnp.ndarray[real_t, ndim=1, mode='c'] xd = \
            np.empty(2 * N_pad * N_pad, dtype=np.float32)
        cdef cnp.ndarray[real_t, ndim=1, mode='c'] yd = \
            np.empty(2 * N_pad * N_pad, dtype=np.float32)

        with nogil:
            sqg_xyderiv(self._c,
                        <const cplx_t *>inp.data,
                        <real_t *>xd.data,
                        <real_t *>yd.data)

        return (xd.reshape(2, N_pad, N_pad),
                yd.reshape(2, N_pad, N_pad))

    # ------------------------------------------------------------------
    def gettend(self, pvspec_in=None):
        """
        Compute spectral PV tendency d(pvspec)/dt.

        Parameters
        ----------
        pvspec_in : array_like (2, N, Nc), complex64, optional
            Spectral PV to compute tendency for.  Uses model's current
            pvspec if not supplied.

        Returns
        -------
        dpvdt : ndarray, shape (2, N, Nc), complex64
        """
        _check_model(self._c)
        cdef int N  = self._c.N
        cdef int Nc = self._c.Nc

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] dpvdt = \
            np.empty(2 * N * Nc, dtype=np.complex64)

        cdef const cplx_t *in_ptr = NULL
        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] pv_flat

        if pvspec_in is not None:
            pv_flat = np.ascontiguousarray(pvspec_in,
                                           dtype=np.complex64).reshape(-1)
            if pv_flat.size != 2 * N * Nc:
                raise ValueError("pvspec_in must have shape (2, N, Nc)")
            in_ptr = <const cplx_t *>pv_flat.data

        with nogil:
            sqg_gettend(self._c, in_ptr,
                        <cplx_t *>dpvdt.data)

        return dpvdt.reshape(2, N, Nc)

    # ------------------------------------------------------------------
    def set_pvspec(self, pvspec):
        """
        Overwrite the model's internal spectral PV.

        Parameters
        ----------
        pvspec : array_like, shape (2, N, Nc), complex64
        """
        _check_model(self._c)
        cdef int N  = self._c.N
        cdef int Nc = self._c.Nc
        cdef size_t nbytes = <size_t>(2 * N * Nc) * sizeof(cplx_t)

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] src = \
            np.ascontiguousarray(pvspec, dtype=np.complex64).reshape(-1)
        if src.size != 2 * N * Nc:
            raise ValueError("pvspec must have shape (2, N, Nc)")

        memcpy(self._c.pvspec, src.data, nbytes)

    # ------------------------------------------------------------------
    #  Convenience FFT helpers  (mirror Python rfft2 / irfft2 calls)
    # ------------------------------------------------------------------

    def rfft2(self, grid):
        """
        Forward 2-D r2c FFT of a (2, N, N) physical array.

        Returns
        -------
        spec : ndarray, shape (2, N, Nc), complex64
        """
        _check_model(self._c)
        cdef int N  = self._c.N
        cdef int Nc = self._c.Nc

        cdef cnp.ndarray[real_t, ndim=1, mode='c'] inp = \
            _f32c(grid).reshape(-1)
        if inp.size != 2 * N * N:
            raise ValueError("grid must have shape (2, N, N)")

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] out = \
            np.empty(2 * N * Nc, dtype=np.complex64)

        with nogil:
            sqg_exec_rfft2(self._c,
                           <const real_t *>inp.data,
                           N, N, Nc,
                           <cplx_t *>out.data)

        return out.reshape(2, N, Nc)

    # ------------------------------------------------------------------
    def irfft2(self, spec):
        """
        Backward 2-D c2r FFT of a (2, N, Nc) spectral array (normalised).

        Returns
        -------
        grid : ndarray, shape (2, N, N), float32
        """
        _check_model(self._c)
        cdef int N  = self._c.N
        cdef int Nc = self._c.Nc

        cdef cnp.ndarray[cnp.complex64_t, ndim=1, mode='c'] inp = \
            np.ascontiguousarray(spec, dtype=np.complex64).reshape(-1)
        if inp.size != 2 * N * Nc:
            raise ValueError("spec must have shape (2, N, Nc)")

        cdef cnp.ndarray[real_t, ndim=1, mode='c'] out = \
            np.empty(2 * N * N, dtype=np.float32)

        with nogil:
            sqg_exec_irfft2(self._c,
                            <const cplx_t *>inp.data,
                            N, N, Nc,
                            <real_t *>out.data)

        return out.reshape(2, N, N)

    # ------------------------------------------------------------------
    def __repr__(self):
        if self._c == NULL:
            return "SQG(uninitialised)"
        return (f"SQG(N={self._c.N}, dt={self._c.dt:.0f}s, "
                f"t={self._c.t/3600:.2f}hr)")
