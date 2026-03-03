#pragma once

// =============================================================
//  sqg.hpp  –  Surface Quasi-Geostrophic turbulence model
//  Translated from sqg.py.
//
//  Dependencies: FFTW3 single-precision  (libfftw3f)
//  Compile:
//    g++ -I<NETCDF_AND_FFTW_INCDIR> -std=c++17 -O3 -march=native main.cpp -L<NETCDF_AND_FFTW_LIBDIR> -lfftw3f -lm -lnetcdf -o sqg
// =============================================================

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>

using real_t = float;
using cplx_t = std::complex<float>;

// ------------------------------------------------------------------
//  2-D real → complex FFT
//  in:  (ncomp × rows × cols) real, row-major
//  out: (ncomp × rows × (cols/2+1)) complex
// ------------------------------------------------------------------
static std::vector<cplx_t>
rfft2(const std::vector<real_t>& in, int rows, int cols, int ncomp = 2)
{
    const int Nc = cols / 2 + 1;
    std::vector<cplx_t>  out(ncomp * rows * Nc);
    std::vector<real_t>  tmp(in);          // FFTW may overwrite input

    for (int k = 0; k < ncomp; ++k) {
        fftwf_plan p = fftwf_plan_dft_r2c_2d(
            rows, cols,
            tmp.data() + k * rows * cols,
            reinterpret_cast<fftwf_complex*>(out.data() + k * rows * Nc),
            FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
    }
    return out;
}

// ------------------------------------------------------------------
//  2-D complex → real (inverse) FFT, normalised by 1/(rows*cols)
// ------------------------------------------------------------------
static std::vector<real_t>
irfft2(const std::vector<cplx_t>& in, int rows, int cols, int ncomp = 2)
{
    const int Nc = cols / 2 + 1;
    std::vector<real_t>  out(ncomp * rows * cols);
    std::vector<cplx_t>  tmp(in);          // c2r destroys input

    const real_t norm = real_t(1) / real_t(rows * cols);
    for (int k = 0; k < ncomp; ++k) {
        fftwf_plan p = fftwf_plan_dft_c2r_2d(
            rows, cols,
            reinterpret_cast<fftwf_complex*>(tmp.data() + k * rows * Nc),
            out.data() + k * rows * cols,
            FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        real_t* sl = out.data() + k * rows * cols;
        for (int i = 0; i < rows * cols; ++i) sl[i] *= norm;
    }
    return out;
}

// ------------------------------------------------------------------
//  Vector arithmetic helpers
// ------------------------------------------------------------------
static std::vector<cplx_t>
vadd(const std::vector<cplx_t>& a, const std::vector<cplx_t>& b)
{
    std::vector<cplx_t> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

static std::vector<cplx_t>
vscale(const std::vector<cplx_t>& a, real_t s)
{
    std::vector<cplx_t> b(a.size());
    for (size_t i = 0; i < a.size(); ++i) b[i] = a[i] * s;
    return b;
}

// ============================================================
//  SQG model
// ============================================================
class SQG {
public:
    // ----------------------------------------------------------
    //  Constructor
    //    pv     – initial PV, shape (2 × N × N) row-major
    //    N_in   – grid size
    // ----------------------------------------------------------
    SQG(
        const std::vector<real_t>& pv,
        int    N_in,
        real_t f          = 1.0e-4f,
        real_t nsq        = 1.0e-4f,
        real_t L          = 20.0e6f,
        real_t H          = 10.0e3f,
        real_t U          = 30.0f,
        real_t r          = 0.0f,
        real_t tdiab      = 10.0f * 86400.f,
        int    diff_order = 8,
        real_t diff_efold = -1.f,     // must be > 0
        real_t theta0     = 300.f,
        real_t g          = 9.8f,
        real_t dt         = 0.f,      // must be != 0
        double tstart     = 0.0
    )
        : N_(N_in),
          f_(f), nsq_(nsq), L_(L), H_(H), U_(U),
          tdiab_(tdiab), diff_order_(diff_order), diff_efold_(diff_efold),
          theta0_(theta0), g_(g), dt_(dt), t_(tstart)
    {
        if ((int)pv.size() != 2 * N_ * N_)
            throw std::invalid_argument("pv must have 2*N*N elements");
        if (N_ % 2)
            throw std::invalid_argument("N must be even (powers of 2 are fastest)");
        if (dt == 0.f)
            throw std::invalid_argument("must specify time step");
        if (diff_efold <= 0.f)
            throw std::invalid_argument("must specify efolding time scale for diffusion");

        // r[0] = +r (bottom),  r[1] = -r (lid)
        r_[0] =  r;
        r_[1] = -r;

        const real_t pi  = real_t(M_PI);
        const int    Nc  = N_ / 2 + 1;
        N_pad_           = 3 * N_ / 2;
        const int Nc_pad = N_pad_ / 2 + 1;

        // ---- basic-state PV pvbar (2 × N × N) ----------------
        {
            const real_t l_fund = 2.f * pi / L_;
            const real_t mu0    = l_fund * std::sqrt(nsq_) * H_ / f_;
            std::vector<real_t> pvbar1d(N_);
            for (int j = 0; j < N_; ++j) {
                real_t y = real_t(j) * L_ / real_t(N_);
                pvbar1d[j] = -(mu0 * 0.5f * U_ / (l_fund * H_))
                              * std::cosh(0.5f * mu0)
                              * std::cos(l_fund * y)
                              / std::sinh(0.5f * mu0);
            }
            // broadcast: pvbar[k, i, j] = pvbar1d[j]  (for both k)
            pvbar_.assign(2 * N_ * N_, 0.f);
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < N_; ++j)
                    for (int i = 0; i < N_; ++i)
                        pvbar_[k * N_ * N_ + j * N_ + i] = pvbar1d[j];
        }

        // ---- spectral state ----------------------------------
        pvspec_eq_ = rfft2(pvbar_, N_, N_);
        pvspec_    = rfft2(pv,     N_, N_);

        // ---- wavenumber arrays (N × Nc) ----------------------
        k_.resize(N_ * Nc);   l_.resize(N_ * Nc);
        ksqlsq_.resize(N_ * Nc);
        ik_.resize(N_ * Nc);  il_.resize(N_ * Nc);
        wavenums_.resize(N_ * Nc);
        Hovermu_.resize(N_ * Nc);
        tanhmu_.resize(N_ * Nc);
        sinhmu_.resize(N_ * Nc);
        hyperdiff_.resize(N_ * Nc);

        const real_t ktotcutoff = pi * real_t(N_) / L_;
        const float  eps_f      = std::numeric_limits<float>::epsilon();
        //std::cout << "FLT_EPSILON: " << eps_f << std::endl;

        for (int i = 0; i < N_; ++i) {
            int    li  = (i <= N_ / 2) ? i : i - N_;
            real_t lv  = 2.f * pi * real_t(li) / L_;

            for (int j = 0; j < Nc; ++j) {
                const int    idx = i * Nc + j;
                const real_t kv  = 2.f * pi * real_t(j) / L_;

                k_[idx]        = kv;
                l_[idx]        = lv;
                ksqlsq_[idx]   = kv * kv + lv * lv;
                ik_[idx]       = cplx_t(0.f, kv);
                il_[idx]       = cplx_t(0.f, lv);
                wavenums_[idx] = std::sqrt(real_t(j * j + li * li));

                double mu_d = std::sqrt((double)ksqlsq_[idx])
                            * std::sqrt((double)nsq_)
                            * (double)H_ / (double)f_;
                if (mu_d < (double)eps_f) mu_d = (double)eps_f;

                Hovermu_[idx] = real_t((double)H_ / mu_d);
                tanhmu_[idx]  = real_t(std::tanh(mu_d));
                sinhmu_[idx]  = real_t(std::sinh(mu_d));

                // additive hyperdiffusion coefficient
                // hyperdiff = -(1/diff_efold) * (ktot/ktotcutoff)^diff_order
                real_t ktot     = std::sqrt(ksqlsq_[idx]);
                hyperdiff_[idx] = -(1.f / diff_efold_)
                                  * std::pow(ktot / ktotcutoff, real_t(diff_order_));
            }
        }

        // ---- padded wavenumber arrays (N_pad × Nc_pad) -------
        ik_pad_.resize(N_pad_ * Nc_pad);
        il_pad_.resize(N_pad_ * Nc_pad);
        for (int i = 0; i < N_pad_; ++i) {
            int    li  = (i <= N_pad_ / 2) ? i : i - N_pad_;
            real_t lv  = 2.f * pi * real_t(li) / L_;
            for (int j = 0; j < Nc_pad; ++j) {
                const int    idx = i * Nc_pad + j;
                const real_t kv  = 2.f * pi * real_t(j) / L_;
                ik_pad_[idx] = cplx_t(0.f, kv);
                il_pad_[idx] = cplx_t(0.f, lv);
            }
        }
    }

    // ----------------------------------------------------------
    //  invert: boundary PV → streamfunction (spectral)
    // ----------------------------------------------------------
    std::vector<cplx_t> invert(const std::vector<cplx_t>* pvspec_in = nullptr) const
    {
        const auto& pvspec = pvspec_in ? *pvspec_in : pvspec_;
        const int   Nc = N_ / 2 + 1;
        std::vector<cplx_t> psispec(2 * N_ * Nc);
        for (int idx = 0; idx < N_ * Nc; ++idx) {
            const cplx_t pv0 = pvspec[           idx];
            const cplx_t pv1 = pvspec[N_ * Nc + idx];
            const real_t hom = Hovermu_[idx];
            const real_t sm  = sinhmu_[idx];
            const real_t tm  = tanhmu_[idx];
            psispec[           idx] = hom * (pv1 / sm - pv0 / tm);
            psispec[N_*Nc + idx] = hom * (pv1 / tm - pv0 / sm);
        }
        return psispec;
    }

    // ----------------------------------------------------------
    //  specpad: zero-pad spectral array to 3/2 grid (× 2.25)
    // ----------------------------------------------------------
    std::vector<cplx_t> specpad(const std::vector<cplx_t>& s) const
    {
        const int Nc     = N_  / 2 + 1;
        const int Nc_pad = N_pad_ / 2 + 1;
        std::vector<cplx_t> out(2 * N_pad_ * Nc_pad, cplx_t(0.f));

        for (int k = 0; k < 2; ++k) {
            const cplx_t* src = s.data()   + k * N_     * Nc;
            cplx_t*       dst = out.data() + k * N_pad_ * Nc_pad;

            // positive-l rows  0 .. N/2-1
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[i * Nc_pad + j] = 2.25f * src[i * Nc + j];

            // negative-l rows  N_pad-N/2 .. N_pad-1
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[(N_pad_ - N_ / 2 + i) * Nc_pad + j] =
                        2.25f * src[(N_ - N_ / 2 + i) * Nc + j];

            // negative-Nyquist column  j == N/2
            for (int i = 0; i < N_ / 2; ++i) {
                dst[i * Nc_pad + N_ / 2] =
                    std::conj(2.25f * src[i * Nc + (Nc - 1)]);
                dst[(N_pad_ - N_ / 2 + i) * Nc_pad + N_ / 2] =
                    std::conj(2.25f * src[(N_ - N_ / 2 + i) * Nc + (Nc - 1)]);
            }
        }
        return out;
    }

    // ----------------------------------------------------------
    //  spectrunc: truncate padded spectral array back to N
    // ----------------------------------------------------------
    std::vector<cplx_t> spectrunc(const std::vector<cplx_t>& s) const
    {
        const int Nc     = N_  / 2 + 1;
        const int Nc_pad = N_pad_ / 2 + 1;
        std::vector<cplx_t> out(2 * N_ * Nc, cplx_t(0.f));
        for (int k = 0; k < 2; ++k) {
            const cplx_t* src = s.data()   + k * N_pad_ * Nc_pad;
            cplx_t*       dst = out.data() + k * N_     * Nc;
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[i * Nc + j] = src[i * Nc_pad + j];
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[(N_ - N_ / 2 + i) * Nc + j] =
                        src[(N_pad_ - N_ / 2 + i) * Nc_pad + j];
        }
        return out;
    }

    // ----------------------------------------------------------
    //  xyderiv: spatial x/y derivatives via padded spectral grid
    // ----------------------------------------------------------
    std::pair<std::vector<real_t>, std::vector<real_t>>
    xyderiv(const std::vector<cplx_t>& specarr) const
    {
        const int Nc_pad = N_pad_ / 2 + 1;
        auto pad = specpad(specarr);
        std::vector<cplx_t> xs(pad.size()), ys(pad.size());
        for (int k = 0; k < 2; ++k)
            for (int idx = 0; idx < N_pad_ * Nc_pad; ++idx) {
                xs[k * N_pad_ * Nc_pad + idx] =
                    ik_pad_[idx] * pad[k * N_pad_ * Nc_pad + idx];
                ys[k * N_pad_ * Nc_pad + idx] =
                    il_pad_[idx] * pad[k * N_pad_ * Nc_pad + idx];
            }
        return {irfft2(xs, N_pad_, N_pad_), irfft2(ys, N_pad_, N_pad_)};
    }

    // ----------------------------------------------------------
    //  gettend: PV tendency dpv/dt
    //
    //  dpvspecdt = (pvspec_eq - pvspec)/tdiab
    //            - jacobianspec
    //            + r[k] * ksqlsq * psispec[k]    (Ekman)
    //            + hyperdiff * pvspec             (hyperdiff)
    // ----------------------------------------------------------
    std::vector<cplx_t> gettend(const std::vector<cplx_t>* pvspec_in = nullptr) const
    {
        const auto& pvspec = pvspec_in ? *pvspec_in : pvspec_;
        const int   Nc = N_ / 2 + 1;

        auto psispec      = invert(&pvspec);
        auto [psix, psiy] = xyderiv(psispec);
        auto [pvx,  pvy ] = xyderiv(pvspec);

        // Jacobian on padded physical grid
        const int G = N_pad_;
        std::vector<real_t> jacobian(2 * G * G);
        for (int i = 0; i < 2 * G * G; ++i)
            jacobian[i] = psix[i] * pvy[i] - psiy[i] * pvx[i];

        auto jacobianspec = spectrunc(rfft2(jacobian, G, G));

        // tendency
        std::vector<cplx_t> dpvspecdt(2 * N_ * Nc);
        for (int k = 0; k < 2; ++k)
            for (int idx = 0; idx < N_ * Nc; ++idx) {
                const int I = k * N_ * Nc + idx;
                dpvspecdt[I] =
                    (pvspec_eq_[I] - pvspec[I]) / tdiab_
                    - jacobianspec[I]
                    + cplx_t(r_[k] * ksqlsq_[idx]) * psispec[I]
                    + cplx_t(hyperdiff_[idx])        * pvspec[I];
            }
        return dpvspecdt;
    }

    // ----------------------------------------------------------
    //  timestep: 4th-order Runge-Kutta
    // ----------------------------------------------------------
    void timestep()
    {
        auto k1   = gettend(&pvspec_);
        auto tmp2 = vadd(pvspec_, vscale(k1, 0.5f * dt_));
        auto k2   = gettend(&tmp2);
        auto tmp3 = vadd(pvspec_, vscale(k2, 0.5f * dt_));
        auto k3   = gettend(&tmp3);
        auto tmp4 = vadd(pvspec_, vscale(k3, dt_));
        auto k4   = gettend(&tmp4);

        const real_t c = dt_ / 6.f;
        for (size_t i = 0; i < pvspec_.size(); ++i)
            pvspec_[i] += c * (k1[i] + 2.f*k2[i] + 2.f*k3[i] + k4[i]);

        t_ += dt_;
    }

    // ----------------------------------------------------------
    //  advance: integrate timesteps steps, return physical PV
    // ----------------------------------------------------------
    std::vector<real_t> advance(int timesteps = 1,
                                 const std::vector<real_t>* pv_in = nullptr)
    {
        if (pv_in)
            pvspec_ = rfft2(*pv_in, N_, N_);
        for (int n = 0; n < timesteps; ++n)
            timestep();
        return irfft2(pvspec_, N_, N_);
    }

    // ----------------------------------------------------------
    //  Accessors
    // ----------------------------------------------------------
    double t()           const { return t_; }
    int    N()           const { return N_; }
    real_t f()           const { return f_; }
    real_t U()           const { return U_; }
    real_t L()           const { return L_; }
    real_t H()           const { return H_; }
    real_t nsq()         const { return nsq_; }
    real_t tdiab()       const { return tdiab_; }
    real_t dt()          const { return dt_; }
    real_t diff_efold()  const { return diff_efold_; }
    int    diff_order()  const { return diff_order_; }
    real_t r(int k)      const { return r_[k]; }

    const std::vector<cplx_t>& pvspec()   const { return pvspec_;   }
    const std::vector<real_t>& wavenums() const { return wavenums_; }

private:
    int    N_, N_pad_;
    real_t f_, nsq_, L_, H_, U_, r_[2];
    real_t tdiab_, diff_efold_, theta0_, g_, dt_;
    int    diff_order_;
    double t_;

    // spectral arrays  (size N × Nc,   Nc = N/2+1)
    std::vector<real_t> k_, l_, ksqlsq_, wavenums_;
    std::vector<double_t> Hovermu_, tanhmu_, sinhmu_;
    std::vector<cplx_t> ik_, il_, hyperdiff_;
    std::vector<cplx_t> ik_pad_, il_pad_;   // N_pad × Nc_pad

    // model state
    std::vector<cplx_t> pvspec_;
    std::vector<cplx_t> pvspec_eq_;
    std::vector<real_t> pvbar_;
};
