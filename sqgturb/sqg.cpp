/**
 * sqg.cpp
 *
 * C++ implementation of the Surface Quasi-Geostrophic (SQG) turbulence model.
 * See sqg.h for documentation.
 */
#include "sqg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

static constexpr float PI = 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SQG::SQG(const float* pv,
         int   N_,
         float f,
         float nsq,
         float L,
         float H,
         float U,
         float r,
         float tdiab,
         int   diff_order,
         float diff_efold,
         float theta0,
         float g,
         bool  symmetric_,
         float dt,
         bool  dealias_,
         int   threads_,
         float tstart)
    : N(N_),
      Nspec(N_ / 2 + 1),
      Npad(3 * N_ / 2),
      Nspec_pad(3 * N_ / 4 + 1),
      f_val(f), nsq_val(nsq), L_val(L), H_val(H), U_val(U), r_val(r),
      tdiab_val(tdiab), theta0_val(theta0), g_val(g), dt_val(dt),
      diff_efold_val(diff_efold), diff_order_val(diff_order),
      symmetric(symmetric_), dealias(dealias_),
      ekman(r >= 1.0e-10f),
      t(static_cast<double>(tstart)),
      timesteps(1),
      threads_val(threads_)
{
    if (N % 2 != 0)
        throw std::invalid_argument("N must be even");
    if (dt <= 0.0f)
        throw std::invalid_argument("must specify a positive time step dt");
    if (diff_efold <= 0.0f)
        throw std::invalid_argument("must specify a positive diff_efold");

    grid_pts = dealias ? (Npad * Npad) : (N * N);

    // ------------------------------------------------------------------
    // Allocate FFTW-aligned workspaces
    // ------------------------------------------------------------------
    fft_r     = fftwf_alloc_real(N * N);
    fft_c     = fftwf_alloc_complex(N * Nspec);
    fft_r_pad = fftwf_alloc_real(Npad * Npad);
    fft_c_pad = fftwf_alloc_complex(Npad * Nspec_pad);

    // ------------------------------------------------------------------
    // Create FFTW plans (FFTW_ESTIMATE: no measurement, safe to reuse arrays)
    // ------------------------------------------------------------------
    plan_r2c     = fftwf_plan_dft_r2c_2d(N,    N,    fft_r,     fft_c,     FFTW_ESTIMATE);
    plan_c2r     = fftwf_plan_dft_c2r_2d(N,    N,    fft_c,     fft_r,     FFTW_ESTIMATE);
    plan_r2c_pad = fftwf_plan_dft_r2c_2d(Npad, Npad, fft_r_pad, fft_c_pad, FFTW_ESTIMATE);
    plan_c2r_pad = fftwf_plan_dft_c2r_2d(Npad, Npad, fft_c_pad, fft_r_pad, FFTW_ESTIMATE);

    // ------------------------------------------------------------------
    // Allocate model arrays
    // ------------------------------------------------------------------
    const int n3c = 2 * N * Nspec;
    const int n3r = 2 * N * N;
    const int n2  = N * Nspec;

    pvspec.resize(n3c,     {0.0f, 0.0f});
    pvspec_eq.resize(n3c,  {0.0f, 0.0f});
    pvbar.resize(n3r,      0.0f);

    k_arr.resize(n2,    0.0f);
    l_arr.resize(n2,    0.0f);
    ksqlsq.resize(n2,   0.0f);
    ik_arr.resize(n2,   {0.0f, 0.0f});
    il_arr.resize(n2,   {0.0f, 0.0f});
    wavenums.resize(n2, 0.0f);
    Hovermu.resize(n2,  0.0f);
    tanhmu.resize(n2,   0.0f);
    sinhmu.resize(n2,   0.0f);
    hyperdiff.resize(n2, 0.0f);

    if (dealias) {
        const int n2p = Npad * Nspec_pad;
        ik_pad.resize(n2p, {0.0f, 0.0f});
        il_pad.resize(n2p, {0.0f, 0.0f});
    }

    // Temporary arrays
    psi_spec.resize(n3c,  {0.0f, 0.0f});
    spec_pad.resize(2 * Npad * Nspec_pad, {0.0f, 0.0f});
    jac_spec.resize(2 * Npad * Nspec_pad, {0.0f, 0.0f});

    psix_phys.resize(2 * grid_pts, 0.0f);
    psiy_phys.resize(2 * grid_pts, 0.0f);
    pvx_phys.resize(2  * grid_pts, 0.0f);
    pvy_phys.resize(2  * grid_pts, 0.0f);
    jac_phys.resize(2  * grid_pts, 0.0f);

    u.resize(2 * grid_pts, 0.0f);
    v.resize(2 * grid_pts, 0.0f);

    pv_rk.resize(n3c, {0.0f, 0.0f});
    k1.resize(n3c,    {0.0f, 0.0f});
    k2.resize(n3c,    {0.0f, 0.0f});
    k3.resize(n3c,    {0.0f, 0.0f});
    k4.resize(n3c,    {0.0f, 0.0f});

    // ------------------------------------------------------------------
    // Build wavenumber arrays
    // ------------------------------------------------------------------
    // k (x direction): 0, 1, ..., N/2          (Nspec values, from rfft)
    // l (y direction): 0, 1, ..., N/2-1, -N/2, ..., -1  (N values, from full fft)
    for (int i = 0; i < N; i++) {
        float l_int = (i < (N + 1) / 2) ? static_cast<float>(i)
                                         : static_cast<float>(i - N);
        for (int j = 0; j < Nspec; j++) {
            float k_int = static_cast<float>(j);
            int   idx   = idx2(i, j);
            float k_dim = 2.0f * PI * k_int / L;
            float l_dim = 2.0f * PI * l_int / L;
            k_arr[idx]  = k_dim;
            l_arr[idx]  = l_dim;
            ksqlsq[idx] = k_dim * k_dim + l_dim * l_dim;
            ik_arr[idx] = {0.0f, k_dim};
            il_arr[idx] = {0.0f, l_dim};
            // wavenums in grid units (integer counts)
            float kk = k_int;
            float ll = (l_int * L) / (2.0f * PI);  // back to integer count
            wavenums[idx] = std::sqrt(kk * kk + ll * ll);
        }
    }

    // Padded wavenumbers
    if (dealias) {
        for (int i = 0; i < Npad; i++) {
            float l_int = (i < (Npad + 1) / 2) ? static_cast<float>(i)
                                                 : static_cast<float>(i - Npad);
            for (int j = 0; j < Nspec_pad; j++) {
                float k_int   = static_cast<float>(j);
                int   idx     = idx2p(i, j);
                ik_pad[idx]   = {0.0f, 2.0f * PI * k_int / L};
                il_pad[idx]   = {0.0f, 2.0f * PI * l_int / L};
            }
        }
    }

    // ------------------------------------------------------------------
    // Compute μ-dependent arrays: Hovermu, tanhmu, sinhmu
    // ------------------------------------------------------------------
    // μ = sqrt(k²+l²) * sqrt(Nsq) * H / f
    const float eps = std::numeric_limits<float>::epsilon();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nspec; j++) {
            int   idx     = idx2(i, j);
            float kl      = std::sqrt(ksqlsq[idx]);
            float mu_f    = kl * std::sqrt(nsq_val) * H_val / f_val;
            if (mu_f < eps) mu_f = eps;           // clip to avoid NaN at k=l=0
            Hovermu[idx]  = H_val / mu_f;
            double mu_d   = static_cast<double>(mu_f);
            tanhmu[idx]   = static_cast<float>(std::tanh(mu_d));
            sinhmu[idx]   = static_cast<float>(std::sinh(mu_d));
        }
    }

    // ------------------------------------------------------------------
    // Hyperdiffusion factor: exp(-dt/diff_efold * (|k|/kcut)^diff_order)
    // ------------------------------------------------------------------
    float kcut = PI * N / L;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nspec; j++) {
            int   idx   = idx2(i, j);
            float ktot  = std::sqrt(ksqlsq[idx]);
            float ratio = ktot / kcut;
            hyperdiff[idx] = std::exp((-dt_val / diff_efold_val)
                                      * std::pow(ratio, static_cast<float>(diff_order_val)));
        }
    }

    // ------------------------------------------------------------------
    // Basic-state pvbar(y): depends only on the y coordinate (row index)
    // ------------------------------------------------------------------
    // l_mode = 2π/L  (fundamental y wavenumber)
    // mu_bs  = l_mode * sqrt(Nsq) * H / f
    float l_mode  = 2.0f * PI / L;
    float mu_bs   = l_mode * std::sqrt(nsq_val) * H_val / f_val;

    std::vector<float> pvbar_y(N);
    for (int i = 0; i < N; i++) {
        float y = static_cast<float>(i) * L / static_cast<float>(N);
        if (symmetric) {
            pvbar_y[i] = -(mu_bs * 0.5f * U_val / (l_mode * H_val))
                          * std::cosh(0.5f * mu_bs)
                          * std::cos(l_mode * y)
                          / std::sinh(0.5f * mu_bs);
        } else {
            pvbar_y[i] = -(mu_bs * U_val / (l_mode * H_val))
                          * std::cos(l_mode * y)
                          / std::sinh(mu_bs);
        }
    }

    // pvbar[lev, i, j] = pvbar_y[i]  (broadcast along x / column axis)
    for (int lev = 0; lev < 2; lev++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                pvbar[idx3r(lev, i, j)] = pvbar_y[i];
            }
        }
    }

    // Asymmetric case: upper boundary has pvbar[1] = pvbar[0] * cosh(mu_bs)
    if (!symmetric) {
        float c = std::cosh(mu_bs);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                pvbar[idx3r(1, i, j)] = pvbar[idx3r(0, i, j)] * c;
            }
        }
    }

    // ------------------------------------------------------------------
    // pvspec_eq = rfft2(pvbar),   pvspec = rfft2(pv)
    // ------------------------------------------------------------------
    for (int lev = 0; lev < 2; lev++) {
        rfft2(pvspec_eq.data() + lev * N * Nspec,
              pvbar.data()     + lev * N * N);
        rfft2(pvspec.data()    + lev * N * Nspec,
              pv               + lev * N * N);
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

SQG::~SQG()
{
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
    fftwf_destroy_plan(plan_r2c_pad);
    fftwf_destroy_plan(plan_c2r_pad);

    fftwf_free(fft_r);
    fftwf_free(fft_c);
    fftwf_free(fft_r_pad);
    fftwf_free(fft_c_pad);
}

// ---------------------------------------------------------------------------
// FFT wrappers
// ---------------------------------------------------------------------------

void SQG::rfft2(std::complex<float>* out, const float* in)
{
    std::memcpy(fft_r, in, static_cast<std::size_t>(N * N) * sizeof(float));
    fftwf_execute_dft_r2c(plan_r2c, fft_r, fft_c);
    std::memcpy(out, fft_c, static_cast<std::size_t>(N * Nspec) * sizeof(fftwf_complex));
}

void SQG::irfft2(float* out, const std::complex<float>* in)
{
    std::memcpy(fft_c, in, static_cast<std::size_t>(N * Nspec) * sizeof(fftwf_complex));
    fftwf_execute_dft_c2r(plan_c2r, fft_c, fft_r);
    const float norm = 1.0f / static_cast<float>(N * N);
    for (int k = 0; k < N * N; k++)
        out[k] = fft_r[k] * norm;
}

void SQG::rfft2_pad(std::complex<float>* out, const float* in)
{
    std::memcpy(fft_r_pad, in, static_cast<std::size_t>(Npad * Npad) * sizeof(float));
    fftwf_execute_dft_r2c(plan_r2c_pad, fft_r_pad, fft_c_pad);
    std::memcpy(out, fft_c_pad,
                static_cast<std::size_t>(Npad * Nspec_pad) * sizeof(fftwf_complex));
}

void SQG::irfft2_pad(float* out, const std::complex<float>* in)
{
    std::memcpy(fft_c_pad, in,
                static_cast<std::size_t>(Npad * Nspec_pad) * sizeof(fftwf_complex));
    fftwf_execute_dft_c2r(plan_c2r_pad, fft_c_pad, fft_r_pad);
    const float norm = 1.0f / static_cast<float>(Npad * Npad);
    for (int k = 0; k < Npad * Npad; k++)
        out[k] = fft_r_pad[k] * norm;
}

// ---------------------------------------------------------------------------
// specpad: zero-pad spectral array [2][N][Nspec] → [2][Npad][Nspec_pad]
// ---------------------------------------------------------------------------
// The 2.25 = (3/2)² scaling factor compensates for the larger irfft2 grid:
// when irfft2 normalises by 1/(Npad²) instead of 1/N², multiplying by
// (Npad/N)² = (3/2)² = 2.25 restores the correct physical-space values.
// ---------------------------------------------------------------------------
void SQG::specpad(std::complex<float>* out, const std::complex<float>* in)
{
    std::fill(out, out + 2 * Npad * Nspec_pad, std::complex<float>(0.0f, 0.0f));

    const float scale = 2.25f;
    const int   hN    = N / 2;

    for (int lev = 0; lev < 2; lev++) {
        // First hN rows of input → first hN rows of output
        for (int i = 0; i < hN; i++) {
            // Columns 0 … hN-1
            for (int j = 0; j < hN; j++)
                out[idx3cp(lev, i, j)] = scale * in[idx3c(lev, i, j)];
            // Negative Nyquist column (column hN in the padded array),
            // = complex conjugate of last column of input (column hN = N/2)
            out[idx3cp(lev, i, hN)] = std::conj(scale * in[idx3c(lev, i, hN)]);
        }

        // Last hN rows of input → last hN rows of output
        for (int i = 0; i < hN; i++) {
            int src_row = hN + i;              // input row:  hN … N-1
            int dst_row = Npad - hN + i;       // output row: Npad-hN … Npad-1
            for (int j = 0; j < hN; j++)
                out[idx3cp(lev, dst_row, j)] = scale * in[idx3c(lev, src_row, j)];
            out[idx3cp(lev, dst_row, hN)] = std::conj(scale * in[idx3c(lev, src_row, hN)]);
        }
    }
}

// ---------------------------------------------------------------------------
// spectrunc: truncate [2][Npad][Nspec_pad] → [2][N][Nspec] (2/3 rule)
// ---------------------------------------------------------------------------
void SQG::spectrunc(std::complex<float>* out, const std::complex<float>* in)
{
    std::fill(out, out + 2 * N * Nspec, std::complex<float>(0.0f, 0.0f));

    const int hN = N / 2;

    for (int lev = 0; lev < 2; lev++) {
        // First hN rows
        for (int i = 0; i < hN; i++)
            for (int j = 0; j < hN; j++)
                out[idx3c(lev, i, j)] = in[idx3cp(lev, i, j)];

        // Last hN rows of input → last hN rows of output
        for (int i = 0; i < hN; i++) {
            int src_row = Npad - hN + i;   // padded rows: Npad-hN … Npad-1
            int dst_row = N    - hN + i;   // output rows: N-hN … N-1
            for (int j = 0; j < hN; j++)
                out[idx3c(lev, dst_row, j)] = in[idx3cp(lev, src_row, j)];
        }
        // Column N/2 is left at zero (dealiasing removes the Nyquist)
    }
}

// ---------------------------------------------------------------------------
// invert: PV → streamfunction
// ---------------------------------------------------------------------------
// From the SQG equations (e.g. Juckes 1994, eqn 4):
//   psispec[0] = (H/μ) * ( pvspec[1]/sinh(μ) − pvspec[0]/tanh(μ) )
//   psispec[1] = (H/μ) * ( pvspec[1]/tanh(μ) − pvspec[0]/sinh(μ) )
// ---------------------------------------------------------------------------
void SQG::invert(std::complex<float>* psispec,
                 const std::complex<float>* pvspec_in)
{
    if (pvspec_in == nullptr) pvspec_in = pvspec.data();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nspec; j++) {
            int   s  = idx2(i, j);
            float hm = Hovermu[s];
            float th = tanhmu[s];
            float sh = sinhmu[s];

            std::complex<float> pv0 = pvspec_in[idx3c(0, i, j)];
            std::complex<float> pv1 = pvspec_in[idx3c(1, i, j)];

            psispec[idx3c(0, i, j)] = hm * (pv1 / sh - pv0 / th);
            psispec[idx3c(1, i, j)] = hm * (pv1 / th - pv0 / sh);
        }
    }
}

// ---------------------------------------------------------------------------
// meantemp: vertical-mean "temperature" spectrum
// ---------------------------------------------------------------------------
void SQG::meantemp(std::complex<float>* pvavspec,
                   const std::complex<float>* pvspec_in)
{
    if (pvspec_in == nullptr) pvspec_in = pvspec.data();
    invert(psi_spec.data(), pvspec_in);
    for (int k = 0; k < N * Nspec; k++)
        pvavspec[k] = (psi_spec[N * Nspec + k] - psi_spec[k]) / H_val;
}

// ---------------------------------------------------------------------------
// invert_inverse: streamfunction → PV  (inverse of invert)
// ---------------------------------------------------------------------------
void SQG::invert_inverse(std::complex<float>* pvspec_out,
                          const std::complex<float>* psispec_in)
{
    // If no psispec supplied, compute it from the current pvspec
    std::vector<std::complex<float>> tmp_psi_local;
    if (psispec_in == nullptr) {
        tmp_psi_local.resize(2 * N * Nspec);
        invert(tmp_psi_local.data(), pvspec.data());
        psispec_in = tmp_psi_local.data();
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nspec; j++) {
            int   s     = idx2(i, j);
            float alpha = Hovermu[s];
            float th    = tanhmu[s];
            float sh    = sinhmu[s];

            // tmp1 = 1/sh² − 1/th²; protect (0,0) from 0/0
            float tmp1;
            if (i == 0 && j == 0) {
                tmp1 = 1.0f;
            } else {
                tmp1 = 1.0f / (sh * sh) - 1.0f / (th * th);
            }
            float denom = alpha * tmp1;

            std::complex<float> psi0 = psispec_in[idx3c(0, i, j)];
            std::complex<float> psi1 = psispec_in[idx3c(1, i, j)];

            pvspec_out[idx3c(0, i, j)] = (psi0 / th - psi1 / sh) / denom;
            pvspec_out[idx3c(1, i, j)] = (psi0 / sh - psi1 / th) / denom;
        }
    }
    // Area-mean PV is not determined by streamfunction
    pvspec_out[idx3c(0, 0, 0)] = {0.0f, 0.0f};
    pvspec_out[idx3c(1, 0, 0)] = {0.0f, 0.0f};
}

// ---------------------------------------------------------------------------
// xyderiv: spectral x- and y-derivatives → physical space
// ---------------------------------------------------------------------------
void SQG::xyderiv(float* xderiv, float* yderiv,
                  const std::complex<float>* specarr)
{
    if (!dealias) {
        // Multiply by ik / il in spectral space, then irfft2
        for (int lev = 0; lev < 2; lev++) {
            const std::complex<float>* src = specarr + lev * N * Nspec;

            // x-derivative: multiply by ik
            for (int k = 0; k < N * Nspec; k++)
                psi_spec[k] = ik_arr[k] * src[k];   // reuse psi_spec as scratch
            irfft2(xderiv + lev * N * N, psi_spec.data());

            // y-derivative: multiply by il
            for (int k = 0; k < N * Nspec; k++)
                psi_spec[k] = il_arr[k] * src[k];
            irfft2(yderiv + lev * N * N, psi_spec.data());
        }
    } else {
        // Pad, multiply by ik_pad / il_pad, irfft2 on padded grid
        specpad(spec_pad.data(), specarr);   // spec_pad: [2][Npad][Nspec_pad]

        for (int lev = 0; lev < 2; lev++) {
            const std::complex<float>* spad = spec_pad.data() + lev * Npad * Nspec_pad;

            // x-derivative
            for (int k = 0; k < Npad * Nspec_pad; k++)
                jac_spec[k] = ik_pad[k] * spad[k];   // reuse jac_spec as scratch
            irfft2_pad(xderiv + lev * Npad * Npad, jac_spec.data());

            // y-derivative
            for (int k = 0; k < Npad * Nspec_pad; k++)
                jac_spec[k] = il_pad[k] * spad[k];
            irfft2_pad(yderiv + lev * Npad * Npad, jac_spec.data());
        }
    }
}

// ---------------------------------------------------------------------------
// gettend: compute PV tendency
// ---------------------------------------------------------------------------
void SQG::gettend(std::complex<float>* dpvdt,
                  const std::complex<float>* pvspec_in)
{
    // 1. Invert PV to get streamfunction (spectral)
    invert(psi_spec.data(), pvspec_in);

    // 2. Compute spatial derivatives: psix, psiy and pvx, pvy
    //    Results are on the (possibly padded) physical grid.
    xyderiv(psix_phys.data(), psiy_phys.data(), psi_spec.data());
    xyderiv(pvx_phys.data(),  pvy_phys.data(),  pvspec_in);

    // 3. Nonlinear Jacobian = psix * pvy − psiy * pvx  (element-wise)
    for (int lev = 0; lev < 2; lev++) {
        for (int k = 0; k < grid_pts; k++) {
            int base        = lev * grid_pts + k;
            jac_phys[base]  = psix_phys[base] * pvy_phys[base]
                            - psiy_phys[base] * pvx_phys[base];
        }
    }

    // 4. FFT Jacobian → spectral; optionally truncate
    if (dealias) {
        for (int lev = 0; lev < 2; lev++)
            rfft2_pad(jac_spec.data() + lev * Npad * Nspec_pad,
                      jac_phys.data() + lev * Npad * Npad);
        spectrunc(dpvdt, jac_spec.data());   // writes [2][N][Nspec]
    } else {
        for (int lev = 0; lev < 2; lev++)
            rfft2(dpvdt + lev * N * Nspec,
                  jac_phys.data() + lev * N * N);
    }

    // 5. Add thermal relaxation; subtract Jacobian
    //    dpvdt = (1/tdiab)*(pvspec_eq − pvspec_in) − jacobianspec
    for (int k = 0; k < 2 * N * Nspec; k++)
        dpvdt[k] = (1.0f / tdiab_val) * (pvspec_eq[k] - pvspec_in[k]) - dpvdt[k];

    // 6. Ekman damping
    if (ekman) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Nspec; j++) {
                float kksq = ksqlsq[idx2(i, j)];
                // Bottom boundary (lev=0): add Ekman
                dpvdt[idx3c(0, i, j)] +=
                    r_val * kksq * psi_spec[idx3c(0, i, j)];
                // Top boundary (lev=1): subtract Ekman (symmetric jet only)
                if (symmetric)
                    dpvdt[idx3c(1, i, j)] -=
                        r_val * kksq * psi_spec[idx3c(1, i, j)];
            }
        }
    }

    // 7. Save wind fields: u = −∂ψ/∂y,  v = +∂ψ/∂x
    for (int k = 0; k < 2 * grid_pts; k++) {
        u[k] = -psiy_phys[k];
        v[k] =  psix_phys[k];
    }
}

// ---------------------------------------------------------------------------
// timestep: 4th-order Runge-Kutta + implicit hyperdiffusion
// ---------------------------------------------------------------------------
void SQG::timestep()
{
    const int n = 2 * N * Nspec;

    // k1 = dt * f(pvspec)
    gettend(k1.data(), pvspec.data());
    for (int i = 0; i < n; i++) k1[i] *= dt_val;

    // k2 = dt * f(pvspec + 0.5*k1)
    for (int i = 0; i < n; i++) pv_rk[i] = pvspec[i] + 0.5f * k1[i];
    gettend(k2.data(), pv_rk.data());
    for (int i = 0; i < n; i++) k2[i] *= dt_val;

    // k3 = dt * f(pvspec + 0.5*k2)
    for (int i = 0; i < n; i++) pv_rk[i] = pvspec[i] + 0.5f * k2[i];
    gettend(k3.data(), pv_rk.data());
    for (int i = 0; i < n; i++) k3[i] *= dt_val;

    // k4 = dt * f(pvspec + k3)
    for (int i = 0; i < n; i++) pv_rk[i] = pvspec[i] + k3[i];
    gettend(k4.data(), pv_rk.data());
    for (int i = 0; i < n; i++) k4[i] *= dt_val;

    // pvspec = hyperdiff * (pvspec + (k1 + 2*k2 + 2*k3 + k4) / 6)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Nspec; j++) {
            float hd = hyperdiff[idx2(i, j)];
            for (int lev = 0; lev < 2; lev++) {
                int ci = idx3c(lev, i, j);
                pvspec[ci] = hd * (pvspec[ci]
                    + (k1[ci] + 2.0f * k2[ci] + 2.0f * k3[ci] + k4[ci]) / 6.0f);
            }
        }
    }

    t += static_cast<double>(dt_val);
}

// ---------------------------------------------------------------------------
// advance: public entry point
// ---------------------------------------------------------------------------
void SQG::advance(float* pv_out, const float* pv_in)
{
    if (pv_in != nullptr) {
        for (int lev = 0; lev < 2; lev++)
            rfft2(pvspec.data() + lev * N * Nspec,
                  pv_in         + lev * N * N);
    }

    for (int step = 0; step < timesteps; step++)
        timestep();

    if (pv_out != nullptr) {
        for (int lev = 0; lev < 2; lev++)
            irfft2(pv_out    + lev * N * N,
                   pvspec.data() + lev * N * Nspec);
    }
}
