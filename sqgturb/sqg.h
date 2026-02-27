/**
 * sqg.h
 *
 * C++ implementation of the Surface Quasi-Geostrophic (SQG) turbulence model.
 *
 * Matches the Python sqg.py class interface:
 *   - Single precision FFTs via FFTW3f
 *   - 4th-order Runge-Kutta time stepping
 *   - Dealiasing with the 2/3 rule (optional)
 *   - Hyperdiffusion treated implicitly
 *   - Ekman damping and thermal relaxation
 *
 * Requires: FFTW3 (link with -lfftw3f)
 */
#pragma once

#include <complex>
#include <vector>
#include <fftw3.h>

class SQG {
public:
    /**
     * Construct and initialise the SQG model.
     *
     * @param pv       Initial potential vorticity [2][N][N], row-major (C order).
     *                 Stored as [level][y][x].
     * @param N        Number of grid points per direction (must be even).
     * @param f        Coriolis parameter (s⁻¹).
     * @param nsq      Brunt-Väisälä frequency squared (s⁻²).
     * @param L        Domain size (m).
     * @param H        Height of upper boundary (m).
     * @param U        Basic-state jet speed at z=H (m/s).
     * @param r        Ekman damping coefficient (m⁻¹ s⁻¹). Zero disables Ekman.
     * @param tdiab    Thermal relaxation time scale (s).
     * @param diff_order  Hyperdiffusion order.
     * @param diff_efold  e-folding time for hyperdiffusion at smallest scale (s).
     *                    Must be specified (positive).
     * @param theta0   Reference potential temperature (K).
     * @param g        Gravitational acceleration (m/s²).
     * @param symmetric  Symmetric jet (true) or asymmetric (false, zero wind at surface).
     * @param dt       Time step (s). Must be specified (positive).
     * @param dealias  Apply 2/3-rule dealiasing to the Jacobian (true by default).
     * @param threads  Number of OpenMP threads for FFTW (currently informational only).
     * @param tstart   Initial time (s).
     */
    SQG(const float* pv,
        int   N,
        float f          = 1.0e-4f,
        float nsq        = 1.0e-4f,
        float L          = 20.0e6f,
        float H          = 10.0e3f,
        float U          = 30.0f,
        float r          = 0.0f,
        float tdiab      = 10.0f * 86400.0f,
        int   diff_order = 8,
        float diff_efold = -1.0f,
        float theta0     = 300.0f,
        float g          = 9.8f,
        bool  symmetric  = true,
        float dt         = -1.0f,
        bool  dealias    = true,
        int   threads    = 1,
        float tstart     = 0.0f);

    ~SQG();

    // -----------------------------------------------------------------------
    // Time integration
    // -----------------------------------------------------------------------

    /**
     * Advance the model by `timesteps` sub-steps.
     *
     * If pv_in != nullptr the model is first loaded from the supplied grid-space
     * PV array [2][N][N].
     *
     * If pv_out != nullptr the current grid-space PV [2][N][N] is written there
     * after all sub-steps complete.
     */
    void advance(float* pv_out, const float* pv_in = nullptr);

    // -----------------------------------------------------------------------
    // Diagnostic helpers (mirror Python methods)
    // -----------------------------------------------------------------------

    /**
     * Invert boundary PV to get streamfunction.
     * psispec [2][N][Nspec] is written.  If pvspec_in is nullptr, this->pvspec is used.
     */
    void invert(std::complex<float>* psispec,
                const std::complex<float>* pvspec_in = nullptr);

    /**
     * Vertical-mean "temperature" spectrum (psi_top - psi_bot) / H.
     * pvavspec [N][Nspec] is written.  If pvspec_in is nullptr, this->pvspec is used.
     */
    void meantemp(std::complex<float>* pvavspec,
                  const std::complex<float>* pvspec_in = nullptr);

    /**
     * Inverse of invert(): given streamfunction, return PV spectra.
     * pvspec_out [2][N][Nspec] is written.  If psispec_in is nullptr, invert() is called.
     */
    void invert_inverse(std::complex<float>* pvspec_out,
                        const std::complex<float>* psispec_in = nullptr);

    // -----------------------------------------------------------------------
    // Public state (matches Python instance attributes)
    // -----------------------------------------------------------------------

    double t;         ///< Current model time (s).
    int    timesteps; ///< Number of sub-steps per advance() call.

    // Grid / spectral dimensions
    const int N;          ///< Grid points per side.
    const int Nspec;      ///< N/2 + 1 (rfft2 output width).
    const int Npad;       ///< 3*N/2 (padded grid for dealiasing).
    const int Nspec_pad;  ///< 3*N/4 + 1 (padded rfft2 output width).

    // Model parameters
    const float f_val, nsq_val, L_val, H_val, U_val, r_val, tdiab_val;
    const float theta0_val, g_val, dt_val, diff_efold_val;
    const int   diff_order_val;
    const bool  symmetric, dealias, ekman;

    // Spectral PV  [2 * N * Nspec]  (stored level-major, then row, then col)
    std::vector<std::complex<float>> pvspec;
    // Equilibrium spectral PV  [2 * N * Nspec]
    std::vector<std::complex<float>> pvspec_eq;
    // Basic-state PV on grid  [2 * N * N]
    std::vector<float> pvbar;

    // Wind fields on the physical grid after the most recent timestep.
    //   If dealias=true  : size = 2 * Npad * Npad
    //   If dealias=false : size = 2 * N * N
    std::vector<float> u;  ///< Zonal wind  u = -dψ/dy
    std::vector<float> v;  ///< Meridional wind  v = +dψ/dx

    // Spectral wavenumber arrays  [N * Nspec]
    std::vector<float>               k_arr;    ///< x wavenumbers (dimensionalised, rad/m)
    std::vector<float>               l_arr;    ///< y wavenumbers (dimensionalised, rad/m)
    std::vector<float>               ksqlsq;   ///< k² + l²
    std::vector<std::complex<float>> ik_arr;   ///< i·k
    std::vector<std::complex<float>> il_arr;   ///< i·l
    std::vector<float>               wavenums; ///< |k| in grid units
    std::vector<float>               Hovermu;  ///< H/μ
    std::vector<float>               tanhmu;   ///< tanh(μ)
    std::vector<float>               sinhmu;   ///< sinh(μ)
    std::vector<float>               hyperdiff;///< exp(-dt/τ_d · (|k|/kcut)^p)

    // Padded wavenumber arrays  [Npad * Nspec_pad]  (only allocated when dealias=true)
    std::vector<std::complex<float>> ik_pad;
    std::vector<std::complex<float>> il_pad;

private:
    int threads_val;
    int grid_pts; ///< Npad*Npad if dealias, else N*N

    // ------------------------------------------------------------------
    // FFTW plans and SIMD-aligned workspaces
    // ------------------------------------------------------------------
    fftwf_plan plan_r2c;      ///< NxN → Nx(N/2+1)
    fftwf_plan plan_c2r;      ///< Nx(N/2+1) → NxN
    fftwf_plan plan_r2c_pad;  ///< NpadxNpad → Npadx(Npad/2+1)
    fftwf_plan plan_c2r_pad;  ///< Npadx(Npad/2+1) → NpadxNpad

    float*         fft_r;     ///< aligned real workspace  [N*N]
    fftwf_complex* fft_c;     ///< aligned complex workspace  [N*Nspec]
    float*         fft_r_pad; ///< aligned real workspace  [Npad*Npad]
    fftwf_complex* fft_c_pad; ///< aligned complex workspace  [Npad*Nspec_pad]

    // ------------------------------------------------------------------
    // Temporary arrays for gettend() and timestep()
    // ------------------------------------------------------------------
    std::vector<std::complex<float>> psi_spec;   // [2*N*Nspec]   streamfunction spectrum
    std::vector<std::complex<float>> spec_pad;   // [2*Npad*Nspec_pad] padded spectrum scratch
    std::vector<float> psix_phys;  // [2*grid_pts]
    std::vector<float> psiy_phys;  // [2*grid_pts]
    std::vector<float> pvx_phys;   // [2*grid_pts]
    std::vector<float> pvy_phys;   // [2*grid_pts]
    std::vector<float> jac_phys;   // [2*grid_pts]
    std::vector<std::complex<float>> jac_spec;   // [2*Npad*Nspec_pad] (truncated in-place to [2*N*Nspec])

    // RK4 work arrays  [2*N*Nspec]
    std::vector<std::complex<float>> pv_rk;  // temporary PV for RK4 stages
    std::vector<std::complex<float>> k1, k2, k3, k4;

    // ------------------------------------------------------------------
    // Private FFT wrappers (single-layer operations)
    // ------------------------------------------------------------------

    /** rfft2: real [N][N] → complex [N][Nspec].  No normalisation. */
    void rfft2(std::complex<float>* out, const float* in);

    /** irfft2: complex [N][Nspec] → real [N][N].  Normalised by 1/(N*N). */
    void irfft2(float* out, const std::complex<float>* in);

    /** rfft2_pad: real [Npad][Npad] → complex [Npad][Nspec_pad].  No normalisation. */
    void rfft2_pad(std::complex<float>* out, const float* in);

    /** irfft2_pad: complex [Npad][Nspec_pad] → real [Npad][Npad].  Normalised by 1/(Npad*Npad). */
    void irfft2_pad(float* out, const std::complex<float>* in);

    // ------------------------------------------------------------------
    // Private model operations
    // ------------------------------------------------------------------

    /** Zero-pad [2][N][Nspec] → [2][Npad][Nspec_pad] with 2.25× scaling. */
    void specpad(std::complex<float>* out, const std::complex<float>* in);

    /** Truncate [2][Npad][Nspec_pad] → [2][N][Nspec] (2/3 rule). */
    void spectrunc(std::complex<float>* out, const std::complex<float>* in);

    /**
     * Compute x and y spatial derivatives of specarr [2][N][Nspec].
     * If dealias: returns arrays on [2][Npad][Npad] grid.
     * Else      : returns arrays on [2][N][N] grid.
     */
    void xyderiv(float* xderiv, float* yderiv, const std::complex<float>* specarr);

    /** Compute PV tendency dpvdt [2][N][Nspec] from pvspec_in [2][N][Nspec]. */
    void gettend(std::complex<float>* dpvdt, const std::complex<float>* pvspec_in);

    /** Advance pvspec one time step with RK4 + implicit hyperdiffusion. */
    void timestep();

    // ------------------------------------------------------------------
    // Flat-array index helpers
    // ------------------------------------------------------------------
    inline int idx2 (int i, int j)           const { return i * Nspec     + j; }
    inline int idx3c(int lev, int i, int j)  const { return lev*N*Nspec   + i*Nspec   + j; }
    inline int idx3r(int lev, int i, int j)  const { return lev*N*N       + i*N       + j; }
    inline int idx2p(int i, int j)           const { return i * Nspec_pad + j; }
    inline int idx3cp(int lev, int i, int j) const { return lev*Npad*Nspec_pad + i*Nspec_pad + j; }
    inline int idx3rp(int lev, int i, int j) const { return lev*Npad*Npad + i*Npad + j; }
};
