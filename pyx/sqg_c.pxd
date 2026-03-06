# sqg_c.pxd
# -----------------------------------------------------------------------
#  Cython extern declarations that mirror sqg.h exactly.
#  Other .pyx files cimport from here; nothing is compiled from this file.
# -----------------------------------------------------------------------

from libc.stddef cimport size_t

cdef extern from "sqg.h":

    # ------------------------------------------------------------------
    #  Scalar types
    # ------------------------------------------------------------------
    ctypedef float         real_t
    ctypedef float[2]      cplx_t      # fftwf_complex

    # ------------------------------------------------------------------
    #  SQG struct  (we only declare fields we need to read from Python;
    #  Cython does not need to know about FFTW plan handles)
    # ------------------------------------------------------------------
    ctypedef struct SQG:
        int    N
        int    Nc
        int    N_pad
        int    Nc_pad

        real_t f
        real_t nsq
        real_t L
        real_t H
        real_t U
        real_t r[2]
        real_t tdiab
        real_t diff_efold
        real_t theta0
        real_t g
        real_t dt
        int    diff_order
        double t

        real_t *k_arr
        real_t *l_arr
        real_t *ksqlsq
        real_t *wavenums
        real_t *Hovermu
        real_t *tanhmu
        real_t *sinhmu
        real_t *hyperdiff
        cplx_t *ik
        cplx_t *il
        cplx_t *ik_pad
        cplx_t *il_pad

        cplx_t *pvspec
        cplx_t *pvspec_eq
        real_t *pvbar

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    SQG *sqg_create(const real_t *pv, int N,
                    real_t f, real_t nsq, real_t L, real_t H, real_t U,
                    real_t r, real_t tdiab, int diff_order, real_t diff_efold,
                    real_t theta0, real_t g, real_t dt, double tstart) nogil

    void sqg_destroy(SQG *s) nogil

    void sqg_invert  (const SQG *s, const cplx_t *pvspec_in,
                      cplx_t *psispec) nogil

    void sqg_specpad (const SQG *s, const cplx_t *inp,
                      cplx_t *out) nogil

    void sqg_spectrunc(const SQG *s, const cplx_t *inp,
                       cplx_t *out) nogil

    void sqg_xyderiv (SQG *s, const cplx_t *specarr,
                      real_t *xderiv, real_t *yderiv) nogil

    void sqg_gettend (SQG *s, const cplx_t *pvspec_in,
                      cplx_t *dpvdt) nogil

    void sqg_timestep(SQG *s) nogil

    void sqg_advance (SQG *s, int ntimesteps,
                      const real_t *pv_in, real_t *pv_out) nogil

    void sqg_exec_rfft2    (SQG *s, const real_t *inp,
                            int rows, int cols, int Nc,
                            cplx_t *out) nogil
    void sqg_exec_irfft2   (SQG *s, const cplx_t *inp,
                            int rows, int cols, int Nc,
                            real_t *out) nogil
    void sqg_exec_rfft2_pad(SQG *s, const real_t *inp,
                            cplx_t *out) nogil
