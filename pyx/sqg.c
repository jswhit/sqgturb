/*
 * sqg.c  -  Surface Quasi-Geostrophic turbulence model
 * C translation of sqg.hpp
 *
 * See sqg.h for the full API description.
 */

#include "sqg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>   /* FLT_EPSILON */

/* ================================================================== */
/*  Internal helpers                                                   */
/* ================================================================== */

/* Allocate an FFTW-aligned buffer of n elements of size elem_sz.
   Exits on failure (mirrors C++ bad_alloc behaviour).               */
static void *aligned_alloc_fftw(size_t n, size_t elem_sz)
{
    void *p = fftwf_malloc(n * elem_sz);
    if (!p) {
        fprintf(stderr, "sqg: fftwf_malloc failed (%zu bytes)\n",
                n * elem_sz);
        return NULL;
    }
    return p;
}

/* ================================================================== */
/*  sqg_create                                                        */
/* ================================================================== */
SQG *sqg_create(const real_t *pv, int N_in,
                real_t f,   real_t nsq, real_t L,
                real_t H,   real_t U,   real_t r,
                real_t tdiab, int diff_order, real_t diff_efold,
                real_t theta0, real_t g, real_t dt, double tstart)
{
    /* ---- validate inputs ------------------------------------ */
    if (N_in % 2) {
        fprintf(stderr, "sqg_create: N must be even\n");
        return NULL;
    }
    if (dt == 0.0f) {
        fprintf(stderr, "sqg_create: must specify time step\n");
        return NULL;
    }
    if (diff_efold <= 0.0f) {
        fprintf(stderr, "sqg_create: diff_efold must be > 0\n");
        return NULL;
    }

    /* ---- allocate struct ------------------------------------ */
    SQG *s = (SQG *)calloc(1, sizeof(SQG));
    if (!s) return NULL;

    /* ---- store scalar parameters ---------------------------- */
    s->N          = N_in;
    s->Nc         = N_in / 2 + 1;
    s->N_pad      = (3 * N_in) / 2;
    s->Nc_pad     = s->N_pad / 2 + 1;
    s->f          = f;
    s->nsq        = nsq;
    s->L          = L;
    s->H          = H;
    s->U          = U;
    s->r[0]       =  r;
    s->r[1]       = -r;
    s->tdiab      = tdiab;
    s->diff_order = diff_order;
    s->diff_efold = diff_efold;
    s->theta0     = theta0;
    s->g          = g;
    s->dt         = dt;
    s->t          = tstart;

    const int N      = s->N;
    const int Nc     = s->Nc;
    const int N_pad  = s->N_pad;
    const int Nc_pad = s->Nc_pad;

    const real_t pi = (real_t)M_PI;

    /* ---- allocate FFTW-aligned work buffers ----------------- */
    s->buf_real     = (real_t *)aligned_alloc_fftw((size_t)(N * N),             sizeof(real_t));
    s->buf_cplx     = (cplx_t *)aligned_alloc_fftw((size_t)(N * Nc),            sizeof(cplx_t));
    s->buf_real_pad = (real_t *)aligned_alloc_fftw((size_t)(N_pad * N_pad),     sizeof(real_t));
    s->buf_cplx_pad = (cplx_t *)aligned_alloc_fftw((size_t)(N_pad * Nc_pad),   sizeof(cplx_t));

    if (!s->buf_real || !s->buf_cplx ||
        !s->buf_real_pad || !s->buf_cplx_pad) {
        sqg_destroy(s);
        return NULL;
    }

    /* ---- create FFTW plans ---------------------------------- */
    s->plan_fwd = fftwf_plan_dft_r2c_2d(
        N, N,
        s->buf_real,
        s->buf_cplx,
        FFTW_MEASURE);

    s->plan_bwd = fftwf_plan_dft_c2r_2d(
        N, N,
        s->buf_cplx,
        s->buf_real,
        FFTW_MEASURE);

    s->plan_fwd_pad = fftwf_plan_dft_r2c_2d(
        N_pad, N_pad,
        s->buf_real_pad,
        s->buf_cplx_pad,
        FFTW_MEASURE);

    s->plan_bwd_pad = fftwf_plan_dft_c2r_2d(
        N_pad, N_pad,
        s->buf_cplx_pad,
        s->buf_real_pad,
        FFTW_MEASURE);

    if (!s->plan_fwd || !s->plan_bwd ||
        !s->plan_fwd_pad || !s->plan_bwd_pad) {
        fprintf(stderr, "sqg_create: fftwf_plan failed\n");
        sqg_destroy(s);
        return NULL;
    }

    /* ---- allocate spectral operator arrays ------------------ */
    s->k_arr     = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->l_arr     = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->ksqlsq    = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->wavenums  = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->Hovermu   = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->tanhmu    = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->sinhmu    = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->hyperdiff = (real_t *)malloc((size_t)(N * Nc) * sizeof(real_t));
    s->ik        = (cplx_t *)malloc((size_t)(N * Nc) * sizeof(cplx_t));
    s->il        = (cplx_t *)malloc((size_t)(N * Nc) * sizeof(cplx_t));
    s->ik_pad    = (cplx_t *)malloc((size_t)(N_pad * Nc_pad) * sizeof(cplx_t));
    s->il_pad    = (cplx_t *)malloc((size_t)(N_pad * Nc_pad) * sizeof(cplx_t));
    s->pvspec    = (cplx_t *)malloc((size_t)(2 * N * Nc) * sizeof(cplx_t));
    s->pvspec_eq = (cplx_t *)malloc((size_t)(2 * N * Nc) * sizeof(cplx_t));
    s->pvbar     = (real_t *)malloc((size_t)(2 * N * N)  * sizeof(real_t));

    if (!s->k_arr || !s->l_arr || !s->ksqlsq || !s->wavenums ||
        !s->Hovermu || !s->tanhmu || !s->sinhmu || !s->hyperdiff ||
        !s->ik || !s->il || !s->ik_pad || !s->il_pad ||
        !s->pvspec || !s->pvspec_eq || !s->pvbar) {
        sqg_destroy(s);
        return NULL;
    }

    /* ---- fill wavenumber / operator arrays ------------------ */
    /*  Row-major layout: index = i*Nc + j
        i = row   (l / y direction),  j = column (k / x direction)
        l[i] = fftfreq,  k[j] = rfftfreq                            */
    const real_t ktotcutoff = pi * (real_t)N / L;
    const double eps_d      = (double)FLT_EPSILON;

    for (int i = 0; i < N; ++i) {
        int    li = (i <= N / 2) ? i : i - N;
        real_t lv = 2.0f * pi * (real_t)li / L;

        for (int j = 0; j < Nc; ++j) {
            const int    idx = i * Nc + j;
            const real_t kv  = 2.0f * pi * (real_t)j / L;

            s->k_arr[idx]  = kv;
            s->l_arr[idx]  = lv;
            s->ksqlsq[idx] = kv * kv + lv * lv;

            /* ik = (0 + i*kv) */
            s->ik[idx][0]  = 0.0f;
            s->ik[idx][1]  = kv;

            /* il = (0 + i*lv) */
            s->il[idx][0]  = 0.0f;
            s->il[idx][1]  = lv;

            s->wavenums[idx] = sqrtf((float)(j * j + li * li));

            /* mu in double to avoid sinh overflow */
            double mu_d = sqrt((double)s->ksqlsq[idx])
                        * sqrt((double)nsq)
                        * (double)H / (double)f;
            if (mu_d < eps_d) mu_d = eps_d;

            s->Hovermu[idx]  = (real_t)((double)H / mu_d);
            s->tanhmu[idx]   = (real_t)tanh(mu_d);
            s->sinhmu[idx]   = (real_t)sinh(mu_d);

            real_t ktot       = sqrtf(s->ksqlsq[idx]);
            s->hyperdiff[idx] = -(1.0f / diff_efold)
                                * powf(ktot / ktotcutoff, (real_t)diff_order);
        }
    }

    /* ---- padded wavenumber operators ------------------------ */
    for (int i = 0; i < N_pad; ++i) {
        int    li = (i <= N_pad / 2) ? i : i - N_pad;
        real_t lv = 2.0f * pi * (real_t)li / L;

        for (int j = 0; j < Nc_pad; ++j) {
            const int    idx = i * Nc_pad + j;
            const real_t kv  = 2.0f * pi * (real_t)j / L;
            s->ik_pad[idx][0] = 0.0f;
            s->ik_pad[idx][1] = kv;
            s->il_pad[idx][0] = 0.0f;
            s->il_pad[idx][1] = lv;
        }
    }

    /* ---- basic-state pvbar ---------------------------------- */
    /*  pvbar[k, i, j] = pvbar1d[j]  (same for both k)
        symmetric zonally-symmetric state, no difference between upper and lower boundary.
        -0.5*U*(2*k-1)*sin(2*pi/L) baroclinic jet (k=1 lower boundary, k=2 upper boundary).
        l = 2.*pi/L and mu = l*N*H/f
        u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
        theta = (f*theta0/g)*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*
        np.cos(l*y)/np.sinh(0.5*mu) + theta0 + (theta0*nsq*z/g)
        C++ layout: pvbar_[k*N*N + j*N + i]                         */
    {
        real_t l_fund = 2.0f * pi / L;
        real_t mu0    = l_fund * sqrtf(nsq) * H / f;
        real_t amp    = -(mu0 * 0.5f * U / (l_fund * H))
                         * coshf(0.5f * mu0) / sinhf(0.5f * mu0);

        for (int j = 0; j < N; ++j) {
            real_t y_val     = (real_t)j * L / (real_t)N;
            real_t pvbar1d_j = amp * cosf(l_fund * y_val);

            for (int i = 0; i < N; ++i) {
                s->pvbar[0 * N * N + j * N + i] = pvbar1d_j;  /* k=0 */
                s->pvbar[1 * N * N + j * N + i] = pvbar1d_j;  /* k=1 */
            }
        }
    }

    /* ---- initial spectral state ----------------------------- */
    sqg_exec_rfft2(s, s->pvbar, N, N, Nc, s->pvspec_eq);
    sqg_exec_rfft2(s, pv,       N, N, Nc, s->pvspec);

    return s;
}

/* ================================================================== */
/*  sqg_destroy                                                       */
/* ================================================================== */
void sqg_destroy(SQG *s)
{
    if (!s) return;

    if (s->plan_fwd)     fftwf_destroy_plan(s->plan_fwd);
    if (s->plan_bwd)     fftwf_destroy_plan(s->plan_bwd);
    if (s->plan_fwd_pad) fftwf_destroy_plan(s->plan_fwd_pad);
    if (s->plan_bwd_pad) fftwf_destroy_plan(s->plan_bwd_pad);

    if (s->buf_real)     fftwf_free(s->buf_real);
    if (s->buf_cplx)     fftwf_free(s->buf_cplx);
    if (s->buf_real_pad) fftwf_free(s->buf_real_pad);
    if (s->buf_cplx_pad) fftwf_free(s->buf_cplx_pad);

    free(s->k_arr);
    free(s->l_arr);
    free(s->ksqlsq);
    free(s->wavenums);
    free(s->Hovermu);
    free(s->tanhmu);
    free(s->sinhmu);
    free(s->hyperdiff);
    free(s->ik);
    free(s->il);
    free(s->ik_pad);
    free(s->il_pad);
    free(s->pvspec);
    free(s->pvspec_eq);
    free(s->pvbar);

    free(s);
}

/* ================================================================== */
/*  sqg_invert  -  boundary PV -> streamfunction (spectral)           */
/*  pvspec_in may be NULL (use s->pvspec)                             */
/* ================================================================== */
void sqg_invert(const SQG *s, const cplx_t *pvspec_in, cplx_t *psispec)
{
    const cplx_t *pv = pvspec_in ? pvspec_in : s->pvspec;
    const int M = s->N * s->Nc;

    for (int idx = 0; idx < M; ++idx) {
        const cplx_t *pv0 = pv + idx;
        const cplx_t *pv1 = pv + M + idx;
        const real_t  hom = s->Hovermu[idx];
        const real_t  sm  = s->sinhmu[idx];
        const real_t  tm  = s->tanhmu[idx];

        /* psispec[idx] = hom * (pv1/sm - pv0/tm) */
        psispec[idx][0] = hom * ((*pv1)[0] / sm - (*pv0)[0] / tm);
        psispec[idx][1] = hom * ((*pv1)[1] / sm - (*pv0)[1] / tm);

        /* psispec[M+idx] = hom * (pv1/tm - pv0/sm) */
        psispec[M + idx][0] = hom * ((*pv1)[0] / tm - (*pv0)[0] / sm);
        psispec[M + idx][1] = hom * ((*pv1)[1] / tm - (*pv0)[1] / sm);
    }
}

/* ================================================================== */
/*  sqg_specpad  -  zero-pad to 3/2 grid, scale by 2.25              */
/* ================================================================== */
void sqg_specpad(const SQG *s, const cplx_t *in, cplx_t *out)
{
    const int N      = s->N;
    const int Nc     = s->Nc;
    const int N_pad  = s->N_pad;
    const int Nc_pad = s->Nc_pad;
    const int nh     = N / 2;

    /* zero the output */
    memset(out, 0, (size_t)(2 * N_pad * Nc_pad) * sizeof(cplx_t));

    for (int k = 0; k < 2; ++k) {
        const cplx_t *src = in  + (size_t)(k * N     * Nc);
        cplx_t       *dst = out + (size_t)(k * N_pad * Nc_pad);

        /* positive-l rows: i = 0 .. N/2-1, cols j = 0 .. N/2-1 */
        for (int i = 0; i < nh; ++i)
            for (int j = 0; j < nh; ++j) {
                cplx_t *d = dst + i * Nc_pad + j;
                const cplx_t *p = src + i * Nc + j;
                (*d)[0] = 2.25f * (*p)[0];
                (*d)[1] = 2.25f * (*p)[1];
            }

        /* negative-l rows: i = N-N/2 .. N-1 mapped to N_pad-N/2 .. N_pad-1 */
        for (int i = 0; i < nh; ++i)
            for (int j = 0; j < nh; ++j) {
                cplx_t *d       = dst + (N_pad - nh + i) * Nc_pad + j;
                const cplx_t *p = src + (N     - nh + i) * Nc     + j;
                (*d)[0] = 2.25f * (*p)[0];
                (*d)[1] = 2.25f * (*p)[1];
            }

        /* negative Nyquist column j = N/2 (conjugate mirror) */
        for (int i = 0; i < nh; ++i) {
            /* positive-l rows */
            {
                cplx_t *d       = dst + i * Nc_pad + nh;
                const cplx_t *p = src + i * Nc     + (Nc - 1);
                (*d)[0] =  2.25f * (*p)[0];
                (*d)[1] = -2.25f * (*p)[1];  /* conjg */
            }
            /* negative-l rows */
            {
                cplx_t *d       = dst + (N_pad - nh + i) * Nc_pad + nh;
                const cplx_t *p = src + (N     - nh + i) * Nc     + (Nc - 1);
                (*d)[0] =  2.25f * (*p)[0];
                (*d)[1] = -2.25f * (*p)[1];  /* conjg */
            }
        }
    }
}

/* ================================================================== */
/*  sqg_spectrunc  -  truncate padded spectral array back to N        */
/* ================================================================== */
void sqg_spectrunc(const SQG *s, const cplx_t *in, cplx_t *out)
{
    const int N      = s->N;
    const int Nc     = s->Nc;
    const int N_pad  = s->N_pad;
    const int Nc_pad = s->Nc_pad;
    const int nh     = N / 2;

    memset(out, 0, (size_t)(2 * N * Nc) * sizeof(cplx_t));

    for (int k = 0; k < 2; ++k) {
        const cplx_t *src = in  + (size_t)(k * N_pad * Nc_pad);
        cplx_t       *dst = out + (size_t)(k * N     * Nc);

        /* positive-l rows */
        for (int i = 0; i < nh; ++i)
            for (int j = 0; j < nh; ++j)
                memcpy(dst + i * Nc + j,
                       src + i * Nc_pad + j,
                       sizeof(cplx_t));

        /* negative-l rows */
        for (int i = 0; i < nh; ++i)
            for (int j = 0; j < nh; ++j)
                memcpy(dst + (N     - nh + i) * Nc     + j,
                       src + (N_pad - nh + i) * Nc_pad + j,
                       sizeof(cplx_t));
    }
}

/* ================================================================== */
/*  sqg_xyderiv  -  x/y derivatives on the padded grid                */
/* ================================================================== */
void sqg_xyderiv(SQG *s, const cplx_t *specarr,
                 real_t *xderiv, real_t *yderiv)
{
    const int Nc_pad = s->Nc_pad;
    const int GNc    = s->N_pad * Nc_pad;
    const int Gsize  = s->N_pad * s->N_pad;
    const real_t norm = 1.0f / (real_t)(s->N_pad * s->N_pad);

    /* zero-pad the input spectrum */
    cplx_t *pad = (cplx_t *)malloc((size_t)(2 * GNc) * sizeof(cplx_t));
    if (!pad) {
        fprintf(stderr, "sqg_xyderiv: malloc failed\n");
        return;
    }
    sqg_specpad(s, specarr, pad);

    for (int k = 0; k < 2; ++k) {
        const cplx_t *src = pad + (size_t)(k * GNc);

        /* ---- x-derivative: buf = ik_pad * src, then IFFT ---- */
        for (int idx = 0; idx < GNc; ++idx) {
            /* multiply by pure-imaginary ik_pad: (0+i*kv)*(re+i*im)
               = -kv*im + i*(kv*re)                                  */
            real_t kv = s->ik_pad[idx][1];   /* imaginary part of ik */
            s->buf_cplx_pad[idx][0] = -kv * src[idx][1];
            s->buf_cplx_pad[idx][1] =  kv * src[idx][0];
        }

        fftwf_execute_dft_c2r(s->plan_bwd_pad,
                               s->buf_cplx_pad,
                               s->buf_real_pad);

        real_t *xd = xderiv + (size_t)(k * Gsize);
        for (int i = 0; i < Gsize; ++i)
            xd[i] = s->buf_real_pad[i] * norm;

        /* ---- y-derivative: buf = il_pad * src, then IFFT ---- */
        for (int idx = 0; idx < GNc; ++idx) {
            real_t lv = s->il_pad[idx][1];
            s->buf_cplx_pad[idx][0] = -lv * src[idx][1];
            s->buf_cplx_pad[idx][1] =  lv * src[idx][0];
        }

        fftwf_execute_dft_c2r(s->plan_bwd_pad,
                               s->buf_cplx_pad,
                               s->buf_real_pad);

        real_t *yd = yderiv + (size_t)(k * Gsize);
        for (int i = 0; i < Gsize; ++i)
            yd[i] = s->buf_real_pad[i] * norm;
    }

    free(pad);
}

/* ================================================================== */
/*  sqg_gettend  -  spectral PV tendency d(pvspec)/dt                 */
/* ================================================================== */
void sqg_gettend(SQG *s, const cplx_t *pvspec_in, cplx_t *dpvdt)
{
    const cplx_t *pv = pvspec_in ? pvspec_in : s->pvspec;

    const int Nc     = s->Nc;
    const int Gsize  = s->N_pad * s->N_pad;
    const int GNc    = s->N_pad * s->Nc_pad;
    const int M      = s->N * Nc;

    /* ---- allocate temporaries ------------------------------- */
    cplx_t *psispec   = (cplx_t *)malloc((size_t)(2 * M)      * sizeof(cplx_t));
    real_t *psix      = (real_t *)malloc((size_t)(2 * Gsize)   * sizeof(real_t));
    real_t *psiy      = (real_t *)malloc((size_t)(2 * Gsize)   * sizeof(real_t));
    real_t *pvx       = (real_t *)malloc((size_t)(2 * Gsize)   * sizeof(real_t));
    real_t *pvy       = (real_t *)malloc((size_t)(2 * Gsize)   * sizeof(real_t));
    real_t *jacobian  = (real_t *)malloc((size_t)(2 * Gsize)   * sizeof(real_t));
    cplx_t *jspec_pad = (cplx_t *)malloc((size_t)(2 * GNc)     * sizeof(cplx_t));
    cplx_t *jspec     = (cplx_t *)malloc((size_t)(2 * M)       * sizeof(cplx_t));

    if (!psispec || !psix || !psiy || !pvx || !pvy ||
        !jacobian || !jspec_pad || !jspec) {
        fprintf(stderr, "sqg_gettend: malloc failed\n");
        goto cleanup;
    }

    /* ---- invert PV -> streamfunction ----------------------- */
    sqg_invert(s, pv, psispec);

    /* ---- spatial derivatives on padded grid ---------------- */
    sqg_xyderiv(s, psispec, psix, psiy);
    sqg_xyderiv(s, pv,      pvx,  pvy);

    /* ---- Jacobian J = psi_x * pv_y - psi_y * pv_x --------- */
    for (int i = 0; i < 2 * Gsize; ++i)
        jacobian[i] = psix[i] * pvy[i] - psiy[i] * pvx[i];

    /* ---- forward FFT of Jacobian, then truncate ------------ */
    sqg_exec_rfft2_pad(s, jacobian, jspec_pad);
    sqg_spectrunc(s, jspec_pad, jspec);

    /* ---- assemble tendency ---------------------------------- */
    /*  dpvdt[I] = (pvspec_eq[I] - pv[I]) / tdiab
                  - jspec[I]
                  + r[k] * ksqlsq[idx] * psispec[I]
                  + hyperdiff[idx]     * pvspec[I]    (always uses pvspec_) */
    for (int k = 0; k < 2; ++k) {
        for (int idx = 0; idx < M; ++idx) {
            const int I = k * M + idx;

            real_t rk   = s->r[k];
            real_t ks   = s->ksqlsq[idx];
            real_t hd   = s->hyperdiff[idx];
            real_t td_i = 1.0f / s->tdiab;

            /* thermal relaxation: (pvspec_eq - pv) / tdiab */
            real_t tend_re = (s->pvspec_eq[I][0] - pv[I][0]) * td_i;
            real_t tend_im = (s->pvspec_eq[I][1] - pv[I][1]) * td_i;

            /* subtract Jacobian */
            tend_re -= jspec[I][0];
            tend_im -= jspec[I][1];

            /* Ekman damping: r[k]*ksqlsq * psispec  (real coeff * complex) */
            tend_re += rk * ks * psispec[I][0];
            tend_im += rk * ks * psispec[I][1];

            /* hyperdiffusion: hyperdiff * pvspec_  (uses current pvspec_) */
            tend_re += hd * s->pvspec[I][0];
            tend_im += hd * s->pvspec[I][1];

            dpvdt[I][0] = tend_re;
            dpvdt[I][1] = tend_im;
        }
    }

cleanup:
    free(psispec);
    free(psix); free(psiy);
    free(pvx);  free(pvy);
    free(jacobian);
    free(jspec_pad);
    free(jspec);
}

/* ================================================================== */
/*  sqg_timestep  -  4th-order Runge-Kutta                           */
/* ================================================================== */
void sqg_timestep(SQG *s)
{
    const size_t sz = (size_t)(2 * s->N * s->Nc);

    cplx_t *k1  = (cplx_t *)malloc(sz * sizeof(cplx_t));
    cplx_t *k2  = (cplx_t *)malloc(sz * sizeof(cplx_t));
    cplx_t *k3  = (cplx_t *)malloc(sz * sizeof(cplx_t));
    cplx_t *k4  = (cplx_t *)malloc(sz * sizeof(cplx_t));
    cplx_t *tmp = (cplx_t *)malloc(sz * sizeof(cplx_t));

    if (!k1 || !k2 || !k3 || !k4 || !tmp) {
        fprintf(stderr, "sqg_timestep: malloc failed\n");
        goto done;
    }

    /* k1 = gettend(pvspec) */
    sqg_gettend(s, s->pvspec, k1);

    /* tmp = pvspec + 0.5*dt*k1 */
    for (size_t i = 0; i < sz; ++i) {
        tmp[i][0] = s->pvspec[i][0] + 0.5f * s->dt * k1[i][0];
        tmp[i][1] = s->pvspec[i][1] + 0.5f * s->dt * k1[i][1];
    }
    sqg_gettend(s, tmp, k2);

    /* tmp = pvspec + 0.5*dt*k2 */
    for (size_t i = 0; i < sz; ++i) {
        tmp[i][0] = s->pvspec[i][0] + 0.5f * s->dt * k2[i][0];
        tmp[i][1] = s->pvspec[i][1] + 0.5f * s->dt * k2[i][1];
    }
    sqg_gettend(s, tmp, k3);

    /* tmp = pvspec + dt*k3 */
    for (size_t i = 0; i < sz; ++i) {
        tmp[i][0] = s->pvspec[i][0] + s->dt * k3[i][0];
        tmp[i][1] = s->pvspec[i][1] + s->dt * k3[i][1];
    }
    sqg_gettend(s, tmp, k4);

    /* pvspec += (dt/6) * (k1 + 2*k2 + 2*k3 + k4) */
    {
        const real_t c = s->dt / 6.0f;
        for (size_t i = 0; i < sz; ++i) {
            s->pvspec[i][0] += c * (k1[i][0] + 2.0f*k2[i][0]
                                    + 2.0f*k3[i][0] + k4[i][0]);
            s->pvspec[i][1] += c * (k1[i][1] + 2.0f*k2[i][1]
                                    + 2.0f*k3[i][1] + k4[i][1]);
        }
    }

    s->t += (double)s->dt;

done:
    free(k1); free(k2); free(k3); free(k4); free(tmp);
}

/* ================================================================== */
/*  sqg_advance                                                       */
/* ================================================================== */
void sqg_advance(SQG *s, int ntimesteps,
                 const real_t *pv_in, real_t *pv_out)
{
    if (pv_in)
        sqg_exec_rfft2(s, pv_in, s->N, s->N, s->Nc, s->pvspec);

    for (int n = 0; n < ntimesteps; ++n)
        sqg_timestep(s);

    sqg_exec_irfft2(s, s->pvspec, s->N, s->N, s->Nc, pv_out);
}

/* ================================================================== */
/*  FFT helpers                                                       */
/* ================================================================== */

/* forward r2c, regular grid: in(2*rows*cols) -> out(2*rows*Nc) */
void sqg_exec_rfft2(SQG *s, const real_t *in,
                    int rows, int cols, int Nc, cplx_t *out)
{
    const int stride_r = rows * cols;
    const int stride_c = rows * Nc;

    for (int k = 0; k < 2; ++k) {
        memcpy(s->buf_real,
               in + (size_t)(k * stride_r),
               (size_t)stride_r * sizeof(real_t));

        fftwf_execute_dft_r2c(s->plan_fwd,
                               s->buf_real,
                               s->buf_cplx);

        memcpy(out + (size_t)(k * stride_c),
               s->buf_cplx,
               (size_t)stride_c * sizeof(cplx_t));
    }
}

/* backward c2r, regular grid, normalised: in(2*rows*Nc) -> out(2*rows*cols) */
void sqg_exec_irfft2(SQG *s, const cplx_t *in,
                     int rows, int cols, int Nc, real_t *out)
{
    const real_t norm    = 1.0f / (real_t)(rows * cols);
    const int    stride_r = rows * cols;
    const int    stride_c = rows * Nc;

    for (int k = 0; k < 2; ++k) {
        memcpy(s->buf_cplx,
               in + (size_t)(k * stride_c),
               (size_t)stride_c * sizeof(cplx_t));

        fftwf_execute_dft_c2r(s->plan_bwd,
                               s->buf_cplx,
                               s->buf_real);

        real_t *dst = out + (size_t)(k * stride_r);
        for (int i = 0; i < stride_r; ++i)
            dst[i] = s->buf_real[i] * norm;
    }
}

/* forward r2c, padded grid: in(2*N_pad*N_pad) -> out(2*N_pad*Nc_pad) */
void sqg_exec_rfft2_pad(SQG *s, const real_t *in, cplx_t *out)
{
    const int stride_r = s->N_pad * s->N_pad;
    const int stride_c = s->N_pad * s->Nc_pad;

    for (int k = 0; k < 2; ++k) {
        memcpy(s->buf_real_pad,
               in + (size_t)(k * stride_r),
               (size_t)stride_r * sizeof(real_t));

        fftwf_execute_dft_r2c(s->plan_fwd_pad,
                               s->buf_real_pad,
                               s->buf_cplx_pad);

        memcpy(out + (size_t)(k * stride_c),
               s->buf_cplx_pad,
               (size_t)stride_c * sizeof(cplx_t));
    }
}
