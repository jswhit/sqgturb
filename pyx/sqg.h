/*
 * sqg.h  -  Surface Quasi-Geostrophic turbulence model
 * C translation of sqg.hpp (C++/FFTW3)
 *
 * Key differences from C++
 * ------------------------
 * - The C++ class SQG becomes a plain struct SQG with all member
 *   data fields exposed directly.
 * - C++ constructors/destructors become sqg_create() / sqg_destroy().
 * - C++ methods become free functions prefixed sqg_* that take
 *   SQG* as their first argument.
 * - std::vector<T> is replaced by malloc'd T* arrays with an
 *   explicit length stored alongside, or heap-allocated in/out
 *   buffers that the caller owns and must free.
 * - std::complex<float> is replaced by fftwf_complex (a float[2]
 *   typedef provided by <fftw3.h>: [0]=real, [1]=imag).
 * - RAII wrappers FftwPlan / AlignedBuf are replaced by raw
 *   fftwf_plan / fftwf_malloc + explicit destroy/free calls.
 * - C++ exceptions are replaced by returning NULL / -1 and
 *   printing an error message to stderr.
 *
 * Dependencies: FFTW3 single-precision  (-lfftw3f)
 *
 * Compile:
 *   gcc -O3 -march=native sqg.c sqg_main.c -lfftw3f -lm -o sqg
 */

#ifndef SQG_H
#define SQG_H

#include <fftw3.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Scalar types                                                       */
/* ------------------------------------------------------------------ */
typedef float          real_t;
typedef fftwf_complex  cplx_t;   /* float[2]: [0]=real, [1]=imag     */

/* ------------------------------------------------------------------ */
/*  Convenience complex arithmetic macros                             */
/*  C99 <complex.h> _Complex could also be used, but fftwf_complex    */
/*  is already what FFTW expects so we stay with it throughout.       */
/* ------------------------------------------------------------------ */
#define C_RE(c)        ((c)[0])
#define C_IM(c)        ((c)[1])

/* c = a + b */
#define CADD(c, a, b)  do { (c)[0]=(a)[0]+(b)[0]; (c)[1]=(a)[1]+(b)[1]; } while(0)
/* c = a - b */
#define CSUB(c, a, b)  do { (c)[0]=(a)[0]-(b)[0]; (c)[1]=(a)[1]-(b)[1]; } while(0)
/* c = s * a  (s real) */
#define CSCALE(c, a, s) do { (c)[0]=(a)[0]*(s); (c)[1]=(a)[1]*(s); } while(0)
/* c += a */
#define CADDTO(c, a)   do { (c)[0]+=(a)[0]; (c)[1]+=(a)[1]; } while(0)
/* c = (0 + i*k) * a  (multiply by pure-imaginary ik) */
#define CMUL_IK(c, ik, a) \
    do { (c)[0] = -(ik)[1]*(a)[1]; (c)[1] = (ik)[1]*(a)[0]; } while(0)
/* conj(a) */
#define CCONJ(c, a)    do { (c)[0]=(a)[0]; (c)[1]=-(a)[1]; } while(0)

/* ------------------------------------------------------------------ */
/*  SQG model struct                                                   */
/* ------------------------------------------------------------------ */
typedef struct SQG {

    /* ---- grid / transform sizes ----------------------------- */
    int N;        /* global grid size                            */
    int Nc;       /* N/2 + 1                                     */
    int N_pad;    /* 3*N/2  (dealiased)                          */
    int Nc_pad;   /* N_pad/2 + 1                                 */

    /* ---- physical parameters -------------------------------- */
    real_t f, nsq, L, H, U;
    real_t r[2];          /* r[0]=+r (bottom), r[1]=-r (lid)    */
    real_t tdiab, diff_efold, theta0, g, dt;
    int    diff_order;
    double t;             /* model time (double for accumulation)*/

    /* ---- spectral operator arrays  (length N*Nc each) ------- */
    real_t  *k_arr;       /* dimensionalised k wavenumber        */
    real_t  *l_arr;       /* dimensionalised l wavenumber        */
    real_t  *ksqlsq;      /* k^2 + l^2                           */
    real_t  *wavenums;    /* sqrt(k^2+l^2) in grid units         */
    real_t  *Hovermu;     /* H / mu                              */
    real_t  *tanhmu;      /* tanh(mu)                            */
    real_t  *sinhmu;      /* sinh(mu)                            */
    real_t  *hyperdiff;   /* hyperdiffusion coefficient          */
    cplx_t  *ik;          /* i*k (complex)                       */
    cplx_t  *il;          /* i*l (complex)                       */

    /* ---- padded wavenumber operators  (N_pad*Nc_pad) -------- */
    cplx_t  *ik_pad;
    cplx_t  *il_pad;

    /* ---- model state  (length 2*N*Nc each) ------------------ */
    cplx_t  *pvspec;      /* spectral PV (current state)         */
    cplx_t  *pvspec_eq;   /* equilibrium spectral PV             */
    real_t  *pvbar;       /* basic-state PV  (length 2*N*N)      */

    /* ---- FFTW plans ----------------------------------------- */
    fftwf_plan plan_fwd;       /* N x N   r2c (regular)          */
    fftwf_plan plan_bwd;       /* N x N   c2r (regular)          */
    fftwf_plan plan_fwd_pad;   /* N_pad x N_pad  r2c (padded)    */
    fftwf_plan plan_bwd_pad;   /* N_pad x N_pad  c2r (padded)    */

    /* ---- FFTW-aligned persistent work buffers --------------- */
    real_t  *buf_real;         /* length N*N                     */
    cplx_t  *buf_cplx;         /* length N*Nc                    */
    real_t  *buf_real_pad;     /* length N_pad*N_pad             */
    cplx_t  *buf_cplx_pad;     /* length N_pad*Nc_pad            */

} SQG;

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/*
 * sqg_create – allocate and initialise a new SQG model.
 *
 *   pv          – initial PV, flat array of 2*N*N floats (row-major)
 *   N           – grid size (must be even)
 *   f,nsq,L,H,U – physical parameters
 *   r           – Ekman damping coefficient
 *   tdiab       – thermal relaxation time scale (seconds)
 *   diff_order  – hyperdiffusion order
 *   diff_efold  – hyperdiffusion e-folding time (seconds, must be > 0)
 *   theta0,g    – for PV-to-temperature conversion
 *   dt          – time step (seconds, must be != 0)
 *   tstart      – initial model time (seconds)
 *
 * Returns a heap-allocated SQG* on success, NULL on failure.
 * The caller must free it with sqg_destroy().
 */
SQG *sqg_create(const real_t *pv, int N,
                real_t f, real_t nsq, real_t L, real_t H, real_t U,
                real_t r, real_t tdiab, int diff_order, real_t diff_efold,
                real_t theta0, real_t g, real_t dt, double tstart);

/*
 * sqg_destroy – free all resources owned by the model.
 */
void sqg_destroy(SQG *s);

/*
 * sqg_invert – boundary PV -> streamfunction (spectral)
 *
 *   pvspec_in  – input spectral PV  (2*N*Nc), or NULL to use s->pvspec
 *   psispec    – output spectral psi (2*N*Nc), must be pre-allocated
 */
void sqg_invert(const SQG *s, const cplx_t *pvspec_in, cplx_t *psispec);

/*
 * sqg_specpad – zero-pad spectral array to 3/2 grid, scale by 2.25
 *
 *   in   – input  (2*N*Nc)
 *   out  – output (2*N_pad*Nc_pad), must be pre-allocated and zeroed
 */
void sqg_specpad(const SQG *s, const cplx_t *in, cplx_t *out);

/*
 * sqg_spectrunc – truncate padded spectral array back to N
 *
 *   in   – input  (2*N_pad*Nc_pad)
 *   out  – output (2*N*Nc), must be pre-allocated
 */
void sqg_spectrunc(const SQG *s, const cplx_t *in, cplx_t *out);

/*
 * sqg_xyderiv – x/y spatial derivatives on the padded (3/2) grid
 *
 *   specarr  – input spectral field (2*N*Nc)
 *   xderiv   – output x-derivative on padded grid (2*N_pad*N_pad)
 *   yderiv   – output y-derivative on padded grid (2*N_pad*N_pad)
 *   Both output arrays must be pre-allocated by the caller.
 */
void sqg_xyderiv(SQG *s, const cplx_t *specarr,
                 real_t *xderiv, real_t *yderiv);

/*
 * sqg_gettend – compute spectral PV tendency dpvspec/dt
 *
 *   pvspec_in – input spectral PV (2*N*Nc), or NULL to use s->pvspec
 *   dpvdt     – output tendency    (2*N*Nc), must be pre-allocated
 */
void sqg_gettend(SQG *s, const cplx_t *pvspec_in, cplx_t *dpvdt);

/*
 * sqg_timestep – one 4th-order Runge-Kutta step
 */
void sqg_timestep(SQG *s);

/*
 * sqg_advance – integrate ntimesteps RK4 steps, write physical PV
 *
 *   ntimesteps – number of time steps
 *   pv_in      – optional new initial PV (2*N*N), or NULL
 *   pv_out     – physical PV on exit     (2*N*N), must be pre-allocated
 */
void sqg_advance(SQG *s, int ntimesteps,
                 const real_t *pv_in, real_t *pv_out);

/* ------------------------------------------------------------------ */
/*  Internal FFT helpers (exposed so sqg_main.c can call them)        */
/* ------------------------------------------------------------------ */
void sqg_exec_rfft2    (SQG *s, const real_t *in,
                        int rows, int cols, int Nc, cplx_t *out);
void sqg_exec_irfft2   (SQG *s, const cplx_t *in,
                        int rows, int cols, int Nc, real_t *out);
void sqg_exec_rfft2_pad(SQG *s, const real_t *in, cplx_t *out);

#endif /* SQG_H */
