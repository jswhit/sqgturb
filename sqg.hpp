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
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using real_t = float;
using cplx_t = std::complex<float>;

// ------------------------------------------------------------------
//  RAII wrapper: owns an fftwf_plan and destroys it on scope exit.
// ------------------------------------------------------------------
struct FftwPlan {
    fftwf_plan p{nullptr};
    FftwPlan() = default;
    explicit FftwPlan(fftwf_plan q) : p(q) {}
    ~FftwPlan() { if (p) fftwf_destroy_plan(p); }
    FftwPlan(const FftwPlan&)            = delete;
    FftwPlan& operator=(const FftwPlan&) = delete;
    FftwPlan(FftwPlan&& o) noexcept : p(o.p) { o.p = nullptr; }
    FftwPlan& operator=(FftwPlan&& o) noexcept {
        if (p) fftwf_destroy_plan(p);
        p = o.p; o.p = nullptr; return *this;
    }
    void execute() const { fftwf_execute(p); }
};

// ------------------------------------------------------------------
//  RAII wrapper: owns an fftwf_malloc'd buffer of T.
//  Provides pointer-like access; never copies data.
// ------------------------------------------------------------------
template<typename T>
struct AlignedBuf {
    T*     ptr{nullptr};
    size_t n{0};

    AlignedBuf() = default;
    explicit AlignedBuf(size_t count)
        : ptr(static_cast<T*>(fftwf_malloc(sizeof(T) * count))), n(count)
    { if (!ptr) throw std::bad_alloc(); }

    ~AlignedBuf() { if (ptr) fftwf_free(ptr); }

    // move-only
    AlignedBuf(const AlignedBuf&)            = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : ptr(o.ptr), n(o.n) { o.ptr=nullptr; o.n=0; }
    AlignedBuf& operator=(AlignedBuf&& o) noexcept {
        if (ptr) fftwf_free(ptr);
        ptr=o.ptr; n=o.n; o.ptr=nullptr; o.n=0; return *this;
    }

    T&       operator[](size_t i)       { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
    size_t   size()                const { return n; }
    void zero() { std::memset(ptr, 0, sizeof(T) * n); }
};

// ------------------------------------------------------------------
//  Vector arithmetic helpers  (std::vector<cplx_t>)
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

        // -------------------------------------------------------
        //  Allocate FFTW-aligned persistent buffers
        // -------------------------------------------------------
        //  Regular grid:  2 components × N × N  (real)
        //                 2 components × N × Nc (complex)
        buf_real_  = AlignedBuf<real_t> (2 * N_     * N_    );
        buf_cplx_  = AlignedBuf<cplx_t >(2 * N_     * Nc    );
        //  Padded grid:   2 components × N_pad × N_pad  (real)
        //                 2 components × N_pad × Nc_pad (complex)
        buf_real_pad_  = AlignedBuf<real_t >(2 * N_pad_ * N_pad_   );
        buf_cplx_pad_  = AlignedBuf<cplx_t >(2 * N_pad_ * Nc_pad   );

        // -------------------------------------------------------
        //  Create plans (FFTW_MEASURE: slower init, faster execute)
        //  Plans are built against the persistent aligned buffers.
        //  "new-array execute" (fftwf_execute_dft_r2c / _c2r) lets
        //  us re-use the same plan on different data with the same
        //  size and alignment.
        // -------------------------------------------------------
        //  Regular forward  r2c  (one component at a time)
        plan_fwd_ = FftwPlan(fftwf_plan_dft_r2c_2d(
            N_, N_,
            buf_real_.ptr,
            reinterpret_cast<fftwf_complex*>(buf_cplx_.ptr),
            FFTW_MEASURE));

        //  Regular backward c2r
        plan_bwd_ = FftwPlan(fftwf_plan_dft_c2r_2d(
            N_, N_,
            reinterpret_cast<fftwf_complex*>(buf_cplx_.ptr),
            buf_real_.ptr,
            FFTW_MEASURE));

        //  Padded forward  r2c
        plan_fwd_pad_ = FftwPlan(fftwf_plan_dft_r2c_2d(
            N_pad_, N_pad_,
            buf_real_pad_.ptr,
            reinterpret_cast<fftwf_complex*>(buf_cplx_pad_.ptr),
            FFTW_MEASURE));

        //  Padded backward c2r
        plan_bwd_pad_ = FftwPlan(fftwf_plan_dft_c2r_2d(
            N_pad_, N_pad_,
            reinterpret_cast<fftwf_complex*>(buf_cplx_pad_.ptr),
            buf_real_pad_.ptr,
            FFTW_MEASURE));

        // -------------------------------------------------------
        //  Wavenumber arrays  (N × Nc)
        // -------------------------------------------------------
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

        for (int i = 0; i < N_; ++i) {
            int    li = (i <= N_ / 2) ? i : i - N_;
            real_t lv = 2.f * pi * real_t(li) / L_;
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

                real_t ktot     = std::sqrt(ksqlsq_[idx]);
                hyperdiff_[idx] = -(1.f / diff_efold_)
                                  * std::pow(ktot / ktotcutoff, real_t(diff_order_));
            }
        }

        // -------------------------------------------------------
        //  Padded wavenumber arrays  (N_pad × Nc_pad)
        // -------------------------------------------------------
        ik_pad_.resize(N_pad_ * Nc_pad);
        il_pad_.resize(N_pad_ * Nc_pad);
        for (int i = 0; i < N_pad_; ++i) {
            int    li = (i <= N_pad_ / 2) ? i : i - N_pad_;
            real_t lv = 2.f * pi * real_t(li) / L_;
            for (int j = 0; j < Nc_pad; ++j) {
                const int    idx = i * Nc_pad + j;
                const real_t kv  = 2.f * pi * real_t(j) / L_;
                ik_pad_[idx] = cplx_t(0.f, kv);
                il_pad_[idx] = cplx_t(0.f, lv);
            }
        }

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

        pvspec_eq_ = exec_rfft2(pvbar_, N_, N_, Nc);
        pvspec_    = exec_rfft2(pv,     N_, N_, Nc);
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
            psispec[N_ * Nc + idx] = hom * (pv1 / tm - pv0 / sm);
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
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[i * Nc_pad + j] = 2.25f * src[i * Nc + j];
            for (int i = 0; i < N_ / 2; ++i)
                for (int j = 0; j < N_ / 2; ++j)
                    dst[(N_pad_ - N_ / 2 + i) * Nc_pad + j] =
                        2.25f * src[(N_ - N_ / 2 + i) * Nc + j];
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
    //  xyderiv: x/y derivatives via padded spectral grid
    //  Uses the cached padded plans + persistent buffers.
    // ----------------------------------------------------------
    std::pair<std::vector<real_t>, std::vector<real_t>>
    xyderiv(const std::vector<cplx_t>& specarr) const
    {
        const int Nc_pad  = N_pad_ / 2 + 1;
        const int Gsize   = N_pad_ * N_pad_;
        const int GNc     = N_pad_ * Nc_pad;
        const real_t norm = real_t(1) / real_t(N_pad_ * N_pad_);

        auto pad = specpad(specarr);

        std::vector<real_t> xderiv(2 * Gsize), yderiv(2 * Gsize);

        for (int k = 0; k < 2; ++k) {
            // --- x-derivative ---
            // multiply pad by ik_pad, store in buf_cplx_pad_
            const cplx_t* src = pad.data() + k * GNc;
            for (int idx = 0; idx < GNc; ++idx)
                buf_cplx_pad_[idx] = ik_pad_[idx] * src[idx];

            // execute cached backward plan via new-array API
            fftwf_execute_dft_c2r(
                plan_bwd_pad_.p,
                reinterpret_cast<fftwf_complex*>(buf_cplx_pad_.ptr),
                buf_real_pad_.ptr);

            real_t* xd = xderiv.data() + k * Gsize;
            for (int i = 0; i < Gsize; ++i) xd[i] = buf_real_pad_[i] * norm;

            // --- y-derivative ---
            for (int idx = 0; idx < GNc; ++idx)
                buf_cplx_pad_[idx] = il_pad_[idx] * src[idx];

            fftwf_execute_dft_c2r(
                plan_bwd_pad_.p,
                reinterpret_cast<fftwf_complex*>(buf_cplx_pad_.ptr),
                buf_real_pad_.ptr);

            real_t* yd = yderiv.data() + k * Gsize;
            for (int i = 0; i < Gsize; ++i) yd[i] = buf_real_pad_[i] * norm;
        }
        return {xderiv, yderiv};
    }

    // ----------------------------------------------------------
    //  gettend: spectral PV tendency dpv/dt
    //
    //  dpvspecdt = (pvspec_eq - pvspec)/tdiab     (thermal relaxation)
    //            - jacobianspec                   (advection)
    //            + r[k] * ksqlsq * psispec[k]     (Ekman damping)
    //            + hyperdiff * pvspec             (hyperdiffusion)
    // ----------------------------------------------------------
    std::vector<cplx_t> gettend(const std::vector<cplx_t>* pvspec_in = nullptr) const
    {
        const auto& pvspec  = pvspec_in ? *pvspec_in : pvspec_;
        const int   Nc  = N_ / 2 + 1;
        const int   Gsize = N_pad_ * N_pad_;
        const int   Nc_pad = N_pad_ / 2 + 1;
        const real_t norm_pad = real_t(1) / real_t(N_pad_ * N_pad_);
        const real_t norm_reg = real_t(1) / real_t(N_ * N_);

        auto psispec      = invert(&pvspec);
        auto [psix, psiy] = xyderiv(psispec);
        auto [pvx,  pvy ] = xyderiv(pvspec);

        // Jacobian on padded physical grid
        std::vector<real_t> jacobian(2 * Gsize);
        for (int i = 0; i < 2 * Gsize; ++i)
            jacobian[i] = psix[i] * pvy[i] - psiy[i] * pvx[i];

        // Forward FFT of Jacobian using cached padded plan
        auto jacobianspec = exec_rfft2_pad(jacobian);
        jacobianspec = spectrunc(jacobianspec);

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
            pvspec_[i] += c * (k1[i] + 2.f * k2[i] + 2.f * k3[i] + k4[i]);

        t_ += dt_;
    }

    // ----------------------------------------------------------
    //  advance: integrate timesteps steps, return physical PV
    // ----------------------------------------------------------
    std::vector<real_t> advance(int timesteps = 1,
                                 const std::vector<real_t>* pv_in = nullptr)
    {
        if (pv_in)
            pvspec_ = exec_rfft2(*pv_in, N_, N_, N_ / 2 + 1);
        for (int n = 0; n < timesteps; ++n)
            timestep();
        return exec_irfft2(pvspec_, N_, N_, N_ / 2 + 1);
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
    // -------------------------------------------------------
    //  exec_rfft2: forward 2-D r2c using the cached regular plan.
    //  Copies input into the aligned buffer, executes, copies out.
    // -------------------------------------------------------
    std::vector<cplx_t>
    exec_rfft2(const std::vector<real_t>& in,
               int rows, int cols, int Nc) const
    {
        std::vector<cplx_t> out(2 * rows * Nc);
        for (int k = 0; k < 2; ++k) {
            // copy input component into aligned buffer
            std::memcpy(buf_real_.ptr,
                        in.data() + k * rows * cols,
                        sizeof(real_t) * rows * cols);

            fftwf_execute_dft_r2c(
                plan_fwd_.p,
                buf_real_.ptr,
                reinterpret_cast<fftwf_complex*>(buf_cplx_.ptr));

            std::memcpy(out.data() + k * rows * Nc,
                        buf_cplx_.ptr,
                        sizeof(cplx_t) * rows * Nc);
        }
        return out;
    }

    // -------------------------------------------------------
    //  exec_irfft2: backward 2-D c2r using the cached regular plan.
    // -------------------------------------------------------
    std::vector<real_t>
    exec_irfft2(const std::vector<cplx_t>& in,
                int rows, int cols, int Nc) const
    {
        const real_t norm = real_t(1) / real_t(rows * cols);
        std::vector<real_t> out(2 * rows * cols);
        for (int k = 0; k < 2; ++k) {
            std::memcpy(buf_cplx_.ptr,
                        in.data() + k * rows * Nc,
                        sizeof(cplx_t) * rows * Nc);

            fftwf_execute_dft_c2r(
                plan_bwd_.p,
                reinterpret_cast<fftwf_complex*>(buf_cplx_.ptr),
                buf_real_.ptr);

            real_t* dst = out.data() + k * rows * cols;
            for (int i = 0; i < rows * cols; ++i)
                dst[i] = buf_real_[i] * norm;
        }
        return out;
    }

    // -------------------------------------------------------
    //  exec_rfft2_pad: forward r2c on padded grid using cached plan.
    // -------------------------------------------------------
    std::vector<cplx_t>
    exec_rfft2_pad(const std::vector<real_t>& in) const
    {
        const int Nc_pad = N_pad_ / 2 + 1;
        std::vector<cplx_t> out(2 * N_pad_ * Nc_pad);
        for (int k = 0; k < 2; ++k) {
            std::memcpy(buf_real_pad_.ptr,
                        in.data() + k * N_pad_ * N_pad_,
                        sizeof(real_t) * N_pad_ * N_pad_);

            fftwf_execute_dft_r2c(
                plan_fwd_pad_.p,
                buf_real_pad_.ptr,
                reinterpret_cast<fftwf_complex*>(buf_cplx_pad_.ptr));

            std::memcpy(out.data() + k * N_pad_ * Nc_pad,
                        buf_cplx_pad_.ptr,
                        sizeof(cplx_t) * N_pad_ * Nc_pad);
        }
        return out;
    }

    // -------------------------------------------------------
    //  Member data
    // -------------------------------------------------------
    int    N_, N_pad_;
    real_t f_, nsq_, L_, H_, U_, r_[2];
    real_t tdiab_, diff_efold_, theta0_, g_, dt_;
    int    diff_order_;
    double t_;

    // spectral arrays  (N × Nc)
    std::vector<real_t> k_, l_, ksqlsq_, wavenums_;
    std::vector<double_t> Hovermu_, tanhmu_, sinhmu_;
    std::vector<cplx_t> ik_, il_, hyperdiff_;
    std::vector<cplx_t> ik_pad_, il_pad_;

    // model state
    std::vector<cplx_t> pvspec_;
    std::vector<cplx_t> pvspec_eq_;
    std::vector<real_t> pvbar_;

    // ---- cached FFTW plans (created once, reused every step) ----
    FftwPlan plan_fwd_;       // N  × N  r2c
    FftwPlan plan_bwd_;       // N  × N  c2r
    FftwPlan plan_fwd_pad_;   // N_pad × N_pad r2c
    FftwPlan plan_bwd_pad_;   // N_pad × N_pad c2r

    // ---- persistent aligned work buffers (one component at a time) ----
    mutable AlignedBuf<real_t>  buf_real_;       // N  × N
    mutable AlignedBuf<cplx_t>  buf_cplx_;       // N  × (N/2+1)
    mutable AlignedBuf<real_t>  buf_real_pad_;   // N_pad × N_pad
    mutable AlignedBuf<cplx_t>  buf_cplx_pad_;   // N_pad × (N_pad/2+1)
};
